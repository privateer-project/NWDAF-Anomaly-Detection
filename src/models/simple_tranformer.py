import torch
import torch.nn as nn
from torch import Tensor

from src.config import SimpleTransformerConfig
from src.models.custom_layers.positional_encoding import PositionalEncoding


class SimpleTransformer(nn.Module):
    def __init__(self, config: SimpleTransformerConfig):
        super().__init__()
        self.config = config
        self.input_embedding = nn.Linear(in_features=self.config.input_size,
                                         out_features=self.config.d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model=self.config.d_model, max_seq_length=self.config.seq_len)
        self.dim_feedforward = self.config.d_model * 4
        self.transformer = nn.Transformer(d_model=self.config.d_model,
                                          nhead=self.config.nhead,
                                          dropout=self.config.dropout,
                                          batch_first=True,
                                          num_encoder_layers=self.config.num_layers,
                                          num_decoder_layers=self.config.num_layers,
                                          dim_feedforward=self.dim_feedforward)
        self.output = nn.Linear(in_features=self.config.d_model,
                                         out_features=self.config.input_size)

    def forward(self, src) -> Tensor:
        emb = self.input_embedding(src)
        pos_enc = self.pos_encoder(emb)
        transformer_output = self.transformer(pos_enc, pos_enc)
        output = self.output(transformer_output)
        return output


    def detect_anomalies(self, x: Tensor, threshold: float) -> Tensor:
        self.eval()
        with torch.no_grad():
            reconstructed = self(x)
            # Compute MSE per sequence
            error = torch.mean((x - reconstructed) ** 2, dim=(1, 2))  # [batch]
            return error > threshold
