import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    # Reference : https://discuss.pytorch.org/t/how-to-modify-the-positional-encoding-in-torch-nn-transformer/104308/3
    # For positional Encoding
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class BaseModel(nn.Module):
    def __init__(self, vocab_size=10000, n_layers=1, d_model=512, nhead=8):
        super(BaseModel, self).__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=n_layers)
        self.pos_embed = PositionalEncoding(d_model=d_model)

    def forward(self, x):
        """
        First Layer : Embedding -> input (N, L) -> output (N, L, D)
        Second Layer : Positional Encoding
        Third Layer : Transformer Encoder
        """
        x = self.embed(x)
        x = self.pos_embed(x)
        x = self.encoder(x)

        return x