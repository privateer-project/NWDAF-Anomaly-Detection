import torch
import torch.nn as nn
from torch import Tensor

from src.config import NewTransformerConfig
from src.models.transformer_ae import PositionalEncoding


class NewTransformer(nn.Module):
    def __init__(
            self,config: NewTransformerConfig,
    ):
        super().__init__()
        self.config= config
        self.input_dim = self.config.input_dim
        self.seq_len = self.config.seq_len
        self.d_model = self.config.d_model
        self.nhead = self.config.nhead
        self.num_layers = self.config.num_layers
        self.dropout = self.config.dropout
        self.d_model = self.config.d_model

        # Input embedding (no MLP)
        self.input_embedding = nn.Linear(self.input_dim, self.d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model=self.d_model, max_len=self.seq_len)

        # Transformer encoder (fewer layers)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dropout=self.dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)

        # Reconstruction head (simple linear layer)
        self.decoder = nn.Linear(self.d_model, self.input_dim)

    def forward(self, x: Tensor) -> Tensor:
        # Input shape: [batch, seq_len, input_dim]
        x_embed = self.input_embedding(x)  # [batch, seq_len, d_model]
        x_embed = self.pos_encoder(x_embed)
        encoded = self.transformer(x_embed)
        reconstructed = self.decoder(encoded)  # [batch, seq_len, input_dim]
        return reconstructed

    def detect_anomalies(self, x: Tensor, threshold: float) -> Tensor:
        self.eval()
        with torch.no_grad():
            reconstructed = self(x)
            # Compute MSE per sequence
            error = torch.mean((x - reconstructed) ** 2, dim=(1, 2))  # [batch]
            return error > threshold