import torch
from torch import nn

from src.config import TransformerADConfig, AutoEncoderConfig
from src.models import AutoEncoder
from src.models.custom_layers.positional_encoding import PositionalEncoding


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
        self.d_ff = self.d_model // 2 #* 4
        assert self.d_model % self.num_heads == 0, f"d_model ({self.input_size}) must be divisible by n_head ({self.num_heads})"

        self.input_embedding = nn.Linear(self.input_size, self.d_model)  # out d_model
        self.pos_encoder = PositionalEncoding(self.d_model, self.seq_len) # out d_model

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.num_heads,
            dim_feedforward=self.d_ff,
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
        return {'transformer_output':transformer_output.clone().detach(), 'ae_output':ae_output}

    def detect_anomalies(self, src, threshold=None):
        self.eval()
        with torch.no_grad():
            output = self(src)
            reconstruction_error = torch.mean((output['transformer_output'] - output['ae_output']) ** 2, dim=(1, 2))
            if threshold is None:
                return reconstruction_error
            return reconstruction_error > threshold
