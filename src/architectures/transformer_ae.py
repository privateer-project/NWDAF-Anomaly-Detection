import torch
from torch import nn
import math
from dataclasses import dataclass


@dataclass
class TransformerADConfig:
    d_input: int = 4
    seq_len: int = 100
    d_model: int = 128
    n_head: int = 4
    n_layers: int = 2
    dropout: float = 0.1


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 300):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, d_model]
        """
        return x + self.pe[:x.size(1), :]


class TimeStepAutoencoder(nn.Module):
    def __init__(self, d_model, dropout_rate=0.1):
        super().__init__()
        hidden_size = d_model // 2
        bottleneck_size = d_model // 4

        self.encoder = nn.Sequential(
            nn.Linear(d_model, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, bottleneck_size),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, d_model)
        )

    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, d_model]
        Returns:
            tuple: (decoded, encoded)
                decoded: [batch_size, seq_len, d_model]
                encoded: [batch_size, seq_len, bottleneck_size]
        """
        # Process each timestep through the autoencoder
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded


class TransformerAD(nn.Module):
    def __init__(self, d_model, num_heads, num_layers, seq_len, d_input, dropout):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = self.d_model * 4
        self.num_layers = num_layers
        self.seq_len = seq_len
        self.d_input = d_input
        self.dropout = dropout
        # batch_size, seq_len, d_input
        assert self.d_model % self.num_heads == 0, f"d_model ({self.d_model}) must be divisible by n_head ({self.num_heads})"

        self.input_embedding = nn.Linear(self.d_input, self.d_model)
        self.pos_encoder = PositionalEncoding(self.d_model, self.seq_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.num_heads,
            dim_feedforward=self.d_ff,
            dropout=self.dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.num_layers
        )

        # Autoencoder that processes each timestep independently
        self.autoencoder = TimeStepAutoencoder(self.d_model, dropout_rate=self.dropout)

        # Output projection back to input dimensions
        self.output_projection = nn.Linear(self.d_model, self.d_input)

    def forward(self, src):
        """
        Args:
            src: Source sequence [batch_size, seq_len, d_input]
        Returns:
            tuple: (output, memory, ae_output, latent)
                - output: Final output [batch_size, seq_len, d_input]
                - memory: Transformer encoder output [batch_size, seq_len, d_model]
                - ae_output: Autoencoder output [batch_size, seq_len, d_model]
                - latent: Encoded representation [batch_size, seq_len, bottleneck_size]
        """
        # Input embedding and positional encoding
        x = self.input_embedding(src)  # [batch_size, seq_len, d_model]
        x = self.pos_encoder(x)  # Add positional encoding

        # Transformer encoding
        memory = self.transformer_encoder(x)  # [batch_size, seq_len, d_model]

        # Autoencoder processing (per timestep)
        ae_output, latent = self.autoencoder(memory)

        # Project back to input dimensions
        output = self.output_projection(ae_output)  # [batch_size, seq_len, d_input]

        return output

    def detect_anomalies(self, src, threshold=None):
        """
        Detect anomalies based on reconstruction error.
        Args:
            src: Input tensor [batch_size, seq_len, d_input]
            threshold: Optional anomaly threshold
        Returns:
            Reconstruction error or anomaly boolean tensor
        """
        self.eval()
        with torch.no_grad():
            output, _, _, _ = self(src)
            reconstruction_error = torch.mean((src - output) ** 2, dim=(1, 2))
            if threshold is None:
                return reconstruction_error
            return reconstruction_error > threshold