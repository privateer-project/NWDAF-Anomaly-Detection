from torch import nn
from src.config.hparams_config import AttentionAutoencoderConfig
from src.models.custom_layers.positional_encoding import PositionalEncoding

class AttentionAutoencoder(nn.Module):
    """Attention-based autoencoder for anomaly detection."""

    def __init__(self, config: AttentionAutoencoderConfig):
        super(AttentionAutoencoder, self).__init__()
        self.config = config
        self.input_size = self.config.input_size
        self.hidden_dim = self.config.hidden_dim
        self.latent_dim = self.config.latent_dim
        self.dropout = self.config.dropout
        self.num_heads = self.config.num_heads
        self.num_layers = self.config.num_layers
        self.seq_len = self.config.seq_len

        self.embed = nn.Linear(self.input_size, self.hidden_dim)
        self.pos_enc = PositionalEncoding(d_model=self.hidden_dim,
                                          max_seq_length=self.seq_len,
                                          dropout=self.dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.num_heads,
            dim_feedforward=self.latent_dim,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.num_layers,
            norm=nn.LayerNorm(self.hidden_dim)
        )

        self.compress = nn.Sequential(nn.Linear(self.hidden_dim, self.latent_dim),
                                    nn.ReLU())
        self.output = nn.Linear(self.latent_dim, self.input_size)

    def forward(self, x):
        x = self.embed(x)
        x = self.pos_enc(x)
        x = self.transformer_encoder(x)
        x = self.compress(x)
        x = self.output(x)
        return x
