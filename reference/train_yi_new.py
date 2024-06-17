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


def train_and_evaluate(model, train_loader, val_loader, epochs, lr):

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    training_losses = []
    validation_losses = []
    best_val_loss = float('inf')

    for epoch in tqdm(range(epochs)):
        model.train()
        loss_epoch = 0
        for mfcc, gct, cop in train_loader:
            gct = gct.unsqueeze(-1)
            weights = torch.ones_like(gct).to(device)
            gct = gct.to(device)
            weights[gct == 0] = 10
            weights[gct == 1] = 1

            optimizer.zero_grad()
            output_gct = model(mfcc.to(device), gct.to(device))
            loss_gct = criterion(output_gct.to(device), gct.float().to(device))
            loss_gct = torch.mean(loss_gct * weights)
            loss_epoch += loss_gct.item()

            loss_gct.backward()
            optimizer.step()
        avg_train_loss = (loss_epoch) / len(train_loader)

        print(f'Loss for gct: {avg_train_loss}')

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
                output_gct = model(mfcc.to(device), gct.to(device))

                loss_gct = criterion(output_gct.to(device), gct.to(device))
                loss_gct = torch.mean(loss_gct * weights)
                val_loss += loss_gct.item()

        avg_val_loss = val_loss / len(val_loader)
        if avg_train_loss < best_val_loss:
            best_val_loss = avg_train_loss
            best_model_weights = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), './pth/transformer_decoder_gct.pth')  # Save the best model
            print('model saved')
        validation_losses.append(avg_val_loss)
        print(avg_val_loss)

    model.load_state_dict(best_model_weights)  # Load the best model for final evaluation
    np.save('./train_loss_transformer_decoder.npy', np.array(training_losses))
    np.save('./val_loss_transformer_decoder.npy', np.array(validation_losses))
    #

train_and_evaluate(model, train_loader, val_loader, epochs=500, lr=0.00001)

