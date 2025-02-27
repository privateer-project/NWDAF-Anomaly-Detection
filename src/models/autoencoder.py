import torch.nn as nn

from src.config import AutoEncoderConfig


class AutoEncoder(nn.Module):
    def __init__(self, config: AutoEncoderConfig):
        super().__init__()
        self.config = config
        self.input_size = self.config.input_size
        self.hidden_size = self.config.hidden_size

        self.dropout_rate = self.config.dropout_rate
        self.encoder = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_size // 2, self.hidden_size // 4),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.hidden_size // 4, self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_size // 2, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.input_size)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
