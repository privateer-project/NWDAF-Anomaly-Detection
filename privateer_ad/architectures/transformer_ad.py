from torch import nn
from opacus.layers import DPMultiheadAttention

from privateer_ad.architectures.layers import PositionalEncoding
from privateer_ad.config import ModelConfig


class TransformerAD(nn.Module):
    """Attention-based model for anomaly detection."""

    def __init__(self, config: ModelConfig = None):
        super(TransformerAD, self).__init__()
        self.config = config or ModelConfig()

        self.embed = nn.Linear(self.config.input_size, self.config.embed_dim)
        self.pos_enc = PositionalEncoding(d_model=self.config.embed_dim,
                                          max_seq_length=self.config.seq_len,
                                          dropout=self.config.dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.config.embed_dim,
            nhead=self.config.num_heads,
            dim_feedforward=self.config.latent_dim,
            batch_first=True
        )
        encoder_layer.self_attn = DPMultiheadAttention(
            embed_dim=self.config.embed_dim,
            num_heads=self.config.num_heads,
            dropout=self.config.dropout,
            batch_first=True)

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.config.num_layers,
            norm=nn.LayerNorm(self.config.embed_dim)
        )

        self.compress = nn.Sequential(nn.Linear(self.config.embed_dim, self.config.latent_dim),
                                      nn.ReLU())
        self.output = nn.Linear(self.config.latent_dim, self.config.input_size)

    def forward(self, x):
        x = self.embed(x)
        x = self.pos_enc(x)
        x = self.transformer_encoder(x)
        x = self.compress(x)
        x = self.output(x)
        return x
