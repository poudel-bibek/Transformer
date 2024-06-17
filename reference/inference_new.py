import numpy as np
from scipy.signal import convolve
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.model_selection import train_test_split
from model_transformer import *
from tqdm import tqdm
import sys

device = 'cpu'

def inference_transformer_decoder(test_loader, model):
    model.eval()
    max_tgt_length = 375
    generated_sequence = []


    for mfcc, gct, cop in test_loader:
        batch_size = mfcc.shape[1]
        tgt = torch.full((1, batch_size, 1), 2, dtype=torch.float32).to(device)
        with torch.no_grad():
            memory = model.input_embedding(mfcc)
            memory = model.pos_encoder(memory)
            memory = model.transformer_encoder(memory)

            for _ in range(max_tgt_length):
                tgt_embedded = model.output_embedding(tgt)
                tgt_embedded = model.pos_decoder(tgt_embedded)
                output = model.transformer_decoder(tgt_embedded, memory)
                output = model.output_layer(output)
                output = model.sigmoid(output)

                # Get the last predicted token
                predicted_token = output[-1, :, :].unsqueeze(0)  # Shape: [1, batch_size, feature_size]
                generated_sequence.append(predicted_token)

                # Prepare the next target input
                tgt = torch.cat((tgt, predicted_token), dim=0)
                # print(tgt.shape)
        # Convert the list of generated tokens to a tensor
        generated_sequence = torch.cat(generated_sequence, dim=0)

        generated_sequence = generated_sequence.permute(1, 0, 2)
        # Process the output as needed
        # For example, converting output tensor to numpy array
        generated_sequence = generated_sequence.cpu().numpy()
        generated_sequence = generated_sequence.reshape((batch_size, -1))
    return generated_sequence

