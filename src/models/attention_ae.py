import torch
from torch import nn

from src.config.hparams_config import AttentionAutoencoderConfig


class AttentionAutoencoder(nn.Module):
    """Attention-based autoencoder for anomaly detection using PyTorch's MultiheadAttention."""

    def __init__(self, config: AttentionAutoencoderConfig):
        super(AttentionAutoencoder, self).__init__()
        self.config = config
        self.input_size = self.config.input_size
        self.hidden_dim = self.config.hidden_dim
        self.latent_dim = self.config.latent_dim
        self.dropout = self.config.dropout
        self.num_heads = self.config.num_heads
        self.num_layers = self.config.num_layers

        encoder_blocks = []
        in_dim = self.input_size
        for i in range(self.num_layers):
            encoder_blocks.extend([
                nn.Linear(in_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout)
            ])
            in_dim = self.hidden_dim  # After first layer, input dimension is hidden_dim

        self.encoder = nn.Sequential(*encoder_blocks)

        self.attention = nn.MultiheadAttention(embed_dim=self.hidden_dim, num_heads=self.num_heads, dropout=self.dropout)

        self.bottleneck = nn.Sequential(
            nn.Linear(self.hidden_dim, self.latent_dim),
            nn.ReLU()
        )

        self.bottleneck_out = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dim),
            nn.ReLU()
        )

        decoder_blocks = []
        for _ in range(self.num_layers - 1):
            decoder_blocks.extend([
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout)
            ])
        decoder_blocks.append(nn.Linear(self.hidden_dim, self.input_size))

        self.decoder = nn.Sequential(*decoder_blocks)

    def encode(self, x):
        # x shape: [batch_size, seq_len, feature_dim]
        h = self.encoder(x)
        h, _ = self.attention(h, h, h, need_weights=False)
        z = self.bottleneck(h)
        return z

    def decode(self, z):
        # z shape: [batch_size, seq_len, latent_dim]
        h = self.bottleneck_out(z)
        x_recon = self.decoder(h)
        return x_recon

    def forward(self, x):
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon

    def get_reconstruction_error(self, x, x_recon):
        """Calculate the reconstruction error for anomaly detection."""
        # Mean squared error per sample and feature
        # Shape: [batch_size, seq_len, feature_dim]
        squared_error = torch.square(x - x_recon)

        # Mean across features and sequence length
        # Shape: [batch_size]
        reconstruction_error = squared_error.mean(dim=(1, 2))

        return reconstruction_error
