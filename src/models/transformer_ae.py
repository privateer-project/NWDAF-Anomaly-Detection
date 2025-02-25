import torch
from torch import nn

import math

from src.config import TransformerADConfig, AutoEncoderConfig
from src.models import AutoEncoder


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 300):
        super(PositionalEncoding, self).__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(1), :]

class TransformerAD(nn.Module):
    def __init__(self, config: TransformerADConfig):
        super().__init__()
        self.config = config
        self.input_size = self.config.input_size
        self.d_model = self.config.d_model
        self.num_heads = self.config.num_heads
        self.num_transformer_layers = self.config.num_transformer_layers
        self.seq_len = self.config.seq_len
        self.dropout = self.config.dropout
        self.d_ff = self.d_model * 4
        assert self.d_model % self.num_heads == 0, f"d_model ({self.input_size}) must be divisible by n_head ({self.num_heads})"

        self.input_embedding = nn.Linear(self.input_size, self.d_model)  # out d_model
        self.pos_encoder = PositionalEncoding(self.d_model, self.seq_len) # out d_model

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.num_heads,
            dim_feedforward=self.d_ff,
            dropout=self.dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.num_transformer_layers,
            norm=nn.LayerNorm(self.d_model)
        )
        self.autoencoder = AutoEncoder(AutoEncoderConfig(input_size=self.d_model, dropout_rate=self.dropout))

    def forward(self, src):
        x = self.input_embedding(src)
        x = self.pos_encoder(x)
        transformer_output = self.transformer_encoder(x)
        ae_output = self.autoencoder(transformer_output)
        return {'transformer_output':transformer_output, 'ae_output':ae_output}

    def detect_anomalies(self, src, threshold=None):
        self.eval()
        with torch.no_grad():
            output, _, = self(src)
            reconstruction_error = torch.mean((src - output) ** 2, dim=(1, 2))
            if threshold is None:
                return reconstruction_error
            return reconstruction_error > threshold
