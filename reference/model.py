import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np
import json
import math


class Transformer(nn.Module):
        def __init__(
                self, 
                input_size: int, # Number of input variables
                
                dec_seq_len: int, # The length of the input sequence fed to the decoder
                max_seq_len: int, # Length of the longest sequence model will receive
                out_seq_len: int, # The length of the model's output (target sequence length)

                dim_val: int, # Dimension (size) of the value vectors of attention mechanism i.e., representations that model attends to  

                n_encoder_layers: int, # Number of stacked encoder layers
                n_decoder_layers: int, # Number of stacked decoder layers
                n_heads: int, # Number of attention heads (parallel attention layers)

                dropout_encoder : float, # Dropout rate in the encoder
                dropout_decoder : float, # Dropout rate in the decoder
                dropout_pos_enc: float, # Dropout rate in the positional encoding layer

                dim_feedforward_encoder: int, # Dimension of the feedforward network (linear) in the encoder
                dim_feedforward_decoder: int, # Dimension of the feedforward network (linear) in the decoder
                
                # The feedforward projector is only found is some architectures.
                dim_feedforward_projector: int, # Dimension of the feedforward network (linear) in the projector
                num_var: int, # Number of additional input variables in the projector.

                ):

                super().__init__()

                self.dec_seq_len = dec_seq_len
                self.n_heads = n_heads
                self.out_seq_len = out_seq_len
                self.dim_val = dim_val
                
                # Process the input before passing it to actual encoder.
                self.encoder_input_layer = nn.Sequential(
                        nn.Linear(input_size, dim_val),
                        nn.ReLU(), # Instead of Tanh
                        nn.Linear(dim_val, dim_val),
                )

                self.decoder_input_layer = nn.Sequential(
                        nn.Linear(input_size, dim_val),
                        nn.ReLU(), # Instead of Tanh
                        nn.Linear(dim_val, dim_val),
                )

                self.linear_mapping = nn.Sequential(
                        nn.Linear(dim_val, dim_val),
                        nn.ReLU(), # Instead of Tanh
                        nn.Linear(dim_val, input_size),
                )

                self.positional_encoding_layer = PositionalEncoder(d_model = dim_val,
                                                                   dropout= dropout_pos_enc,
                                                                   max_len = max_seq_len)
                
                self.projector 