import math

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length=100, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)

        num_even_positions = (d_model + 1) // 2  # Ceiling division for odd d_model
        num_odd_positions = d_model // 2         # Floor division

        div_term_even = torch.exp(torch.arange(0, num_even_positions).float() * (-math.log(10000.0) / d_model))
        div_term_odd = torch.exp(torch.arange(0, num_odd_positions).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term_even)
        pe[:, 1::2] = torch.cos(position * div_term_odd)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
