import torch.nn as nn
import torch



class PositionalEncoding(nn.Module):
    def __init__(self, feature_size, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, feature_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, feature_size, 2).float() * (-torch.log(torch.tensor(10000.0)) / feature_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

class AIRunner_Transformer(nn.Module):
    def __init__(self, feature_size=80, num_layers=8, dropout=0.2):
        super(AIRunner_Transformer, self).__init__()
        self.input_embedding = nn.Linear(80, feature_size)
        self.pos_encoder = PositionalEncoding(feature_size)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=10, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

        self.output_embedding = nn.Embedding(3, feature_size)
        self.pos_decoder = PositionalEncoding(feature_size)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=feature_size, nhead=10, dropout=dropout)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(feature_size, 1)
        self.device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
        self.sigmoid = nn.Sigmoid()

    def forward(self, src, tgt):
        device = self.device

        tgt_mask = self._generate_square_subsequent_mask(len(tgt)).to(device)

        src = self.input_embedding(src)
        src = self.pos_encoder(src)
        feature = self.transformer_encoder(src)

        tgt = self.output_embedding(tgt).squeeze(-2)
        tgt = self.pos_decoder(tgt)
        output = self.transformer_decoder(tgt, feature, tgt_mask = tgt_mask, memory_mask = None)
        output = self.output_layer(output)
        return self.sigmoid(output)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


if __name__ == '__main__':

    model = AIRunner_Transformer()
    src = torch.rand((300, 32, 80)) # (S,N,E)
    tgt = torch.rand((301, 32, 1))
    tgt = torch.randint(0, 3, (376, 32, 1), dtype=torch.long)
    out1 = model(src, tgt)
    print('output shape',out1.shape)
