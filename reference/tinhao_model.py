import torch.nn as nn
import torch
import numpy as np
from matplotlib import pyplot
import math
input_window = 100  # number of input steps
output_window = 1  # number of prediction steps, in this model its fixed to one
block_len = input_window + output_window  # for one input-output pair
batch_size = 32
train_size = 0.7
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # div_term = torch.exp(
        #     torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        # )
        div_term = 1 / (10000 ** ((2 * np.arange(d_model)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term[0::2])
        pe[:, 1::2] = torch.cos(position * div_term[1::2])
        pe = pe.unsqueeze(0).transpose(0, 1)  # [5000, 1, d_model],so need seq-len <= 5000
        # pe.requires_grad = False
        self.register_buffer('pe', pe)
    def forward(self, x):
        # print(self.pe[:x.size(0), :].repeat(1,x.shape[1],1).shape ,'---',x.shape)
        # dimension 1 maybe inequal batchsize
        return x + self.pe[:x.size(0), :].repeat(1, x.shape[1], 1)
class AIRunner(nn.Module):
    def __init__(self, feature_size=80, num_layers=8, dropout=0.2):
        super(AIRunner, self).__init__()
        self.input_embedding = nn.Linear(80, feature_size)
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(feature_size)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=10, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder_cadence = nn.Sequential(
            nn.Linear(feature_size, feature_size // 2),
            # nn.ReLU(),
            nn.Linear(feature_size // 2, 1)
        )
        self.decoder_cop = nn.Sequential(
            nn.Linear(feature_size, feature_size // 2),
            nn.ReLU(),
            nn.Linear(feature_size // 2, 2)
        )
        self.decoder_pressure = nn.Sequential(
            nn.Linear(feature_size, feature_size // 2),
            nn.ReLU(),
            nn.Linear(feature_size // 2, 4)
        )
    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask
        src = self.input_embedding(src)  # linear transformation before positional embedding
        src = self.pos_encoder(src)
        feature = self.transformer_encoder(src, self.src_mask)  # , self.src_mask)
        cadence = self.decoder_cadence(feature)
        cop = self.decoder_cop(feature)
        # output = torch.sigmoid(output)
        return cadence, cop
    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
if __name__ == '__main__':
    model = AIRunner().to(device)
    src = torch.rand((300, 32, 80)) # (S,N,E)
    out1, out2 = model(src.to(device))
    print('output shape',out1.shape, out2.shape)
11:01
Here is training:
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.model_selection import train_test_split
import pandas as pd
import copy
from sklearn.metrics import mean_squared_error
from scipy.signal import butter, lfilter
import torch.nn.functional as F
import random
import json
from model_yi import *
# from data_processing import *
# import torch
# import copy
from tqdm import tqdm
import sys
torch.manual_seed(0)
np.random.seed(0)
###model
input_window = 100  # number of input steps
output_window = 1  # number of prediction steps, in this model it's fixed to one
block_len = input_window + output_window  # for one input-output pair
batch_size = 32
train_size = 0.7
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
model = AIRunner().to(device)
BATCH_SIZE = 32
def collate_fn(batch):
    mfcc, gct, cop = zip(*batch)
    mfcc = torch.stack(mfcc).permute(1, 0, 2)  # Shape: (sequence length, batch size, feature dimension)
    gct = torch.stack(gct).permute(1, 0)  # Shape: (sequence length, batch size)
    cop = torch.stack(cop).permute(1, 0, 2)  # Shape: (sequence length, batch size)
    return mfcc, gct, cop
### data
def get_data(user):
    mfcc = np.load(f'./data_processed/{user}/mfcc.npy')
    truth_gct = np.load(f'./data_processed/{user}/truth_gct.npy')
    truth_cop = np.load(f'./data_processed/{user}/truth_cop.npy') / 100
    # Splitting data for training, validation, and testing
    mfcc_temp, mfcc_test, gct_temp, gct_test, cop_temp, cop_test = train_test_split(mfcc, truth_gct, truth_cop, test_size=0.1)
    np.save(f'./data_processed/{user}/mfcc_test.npy', mfcc_test)
    np.save(f'./data_processed/{user}/gct_test.npy', gct_test)
    np.save(f'./data_processed/{user}/cop_test.npy', cop_test)
    mfcc_train, mfcc_val, gct_train, gct_val, cop_train, cop_val = train_test_split(mfcc_temp, gct_temp, cop_temp, test_size=0.2)
    train_dataset = TensorDataset(torch.tensor(mfcc_train, dtype=torch.float32),
                                  torch.tensor(gct_train, dtype=torch.float32),
                                  torch.tensor(cop_train, dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_dataset = TensorDataset(torch.tensor(mfcc_val, dtype=torch.float32),
                                  torch.tensor(gct_val, dtype=torch.float32),
                                  torch.tensor(cop_val, dtype=torch.float32))
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)
    test_dataset = TensorDataset(torch.tensor(mfcc_test, dtype=torch.float32),
                                  torch.tensor(gct_test, dtype=torch.float32),
                                  torch.tensor(cop_test, dtype=torch.float32))
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)
    return train_loader, val_loader, test_loader
user = 'irfan_0220'
# Load and process all data
train_loader, val_loader, test_loader = get_data(user)
def train_and_evaluate(model, train_loader, val_loader, epochs=500, lr=0.001):
    # Reproducibility
    # random.seed(1)
    # np.random.seed(1)
    # torch.manual_seed(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # criterion = nn.BCELoss()
    criterion = nn.MSELoss()
    #criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    training_losses = []
    validation_losses = []
    best_val_loss = float('inf')
    best_model_weights = copy.deepcopy(model.state_dict())
    train_output = []
    for epoch in tqdm(range(epochs)):
        model.train()
        gct_loss = 0.0
        cop_loss = 0.0
        for mfcc, gct, cop in train_loader:
            gct = gct.unsqueeze(-1)
            weights = torch.ones_like(gct).to(device)
            weights[gct == 0] = 2
            weights[gct == 1] = 1
            optimizer.zero_grad()
            output_gct, output_cop = model(mfcc.to(device))
            loss_gct = criterion(output_gct.to(device), gct.to(device))
            loss_gct = torch.mean(loss_gct * weights)
            gct_loss += loss_gct.item()
            loss_cop = criterion(output_cop.to(device), cop.to(device))
            cop_loss += loss_cop.item()
            loss = loss_gct + loss_cop
            loss.backward()
            optimizer.step()
        avg_train_loss = (gct_loss + cop_loss) / len(train_loader)
        print(f'Loss for gct: {gct_loss / len(train_loader)}, loss for cop: {cop_loss / len(train_loader)}')
        training_losses.append(avg_train_loss)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for mfcc, gct, cop in val_loader:
                gct = gct.unsqueeze(-1)
                weights = torch.ones_like(gct).to(device)
                weights[gct == 0] = 2
                weights[gct == 1] = 1
                optimizer.zero_grad()
                output_gct, output_cop = model(mfcc.to(device))
                loss_gct = criterion(output_gct.to(device), gct.to(device))
                loss_gct = torch.mean(loss_gct * weights)
                gct_loss += loss_gct.item()
                loss_cop = criterion(output_cop.to(device), cop.to(device))
                cop_loss += loss_cop.item()
                val_loss += (gct_loss + cop_loss)
        avg_val_loss = val_loss / len(val_loader)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_weights = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), './pth/multi_task_yi.pth')  # Save the best model
        validation_losses.append(avg_val_loss)
        print(avg_val_loss)
        # if (epoch + 1) % 10 == 0:
        #     print(f"Epoch {epoch + 1}, Training Loss: {avg_train_loss:.6f}")
        #     print(f"Validation Loss: {avg_val_loss:.6f}")
    model.load_state_dict(best_model_weights)  # Load the best model for final evaluation
    np.save('./train_loss_multi_task.npy', np.array(training_losses))
    np.save('./val_loss_multi_task.npy', np.array(validation_losses))
    #
    # # Plotting
    # plt.plot(training_losses, label='Training Loss')
    # plt.plot(validation_losses, label='Validation Loss')
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.title('Training and Validation Losses')
    # plt.legend()
    # plt.show()
import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
train_and_evaluate(model, train_loader, val_loader, epochs=500, lr=0.0001)