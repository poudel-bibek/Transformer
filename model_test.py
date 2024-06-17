import torch 
import torch.nn as nn

torch.manual_seed(0)

class SimpleTransformer(nn.Module):
    def __init__(self, 
                 d_model = 512, 
                 nhead = 4,
                 num_encoder_layers = 2,
                 num_decoder_layers = 2,
                 dim_feedforward = 512,
                 dropout = 0.1,):
        
        super(SimpleTransformer, self).__init__()
        self.transformer = nn.Transformer(d_model = d_model,
                                          nhead = nhead,
                                          num_encoder_layers = num_encoder_layers,
                                          num_decoder_layers = num_decoder_layers,
                                          dim_feedforward = dim_feedforward,
                                          dropout = dropout)
        self.linear = nn.Linear(d_model, d_model)
    
    def forward(self, src, tgt):
        x = self.transformer(src, tgt)
        x = self.linear(x)
        return x

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f'Using device: {device}')

    d_model = 128
    model = SimpleTransformer(d_model = d_model).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f'The model has {num_params} parameters')


    # First dimension is sequence length, batch size is the second.
    src = torch.rand(10, 32, d_model).to(device)  # [sequence_length, batch_size, d_model]
    tgt = torch.rand(20, 32, d_model).to(device)  # [sequence_length, batch_size, d_model]

    output = model(src, tgt)

    print(output.shape)

if __name__ == '__main__':
    main()