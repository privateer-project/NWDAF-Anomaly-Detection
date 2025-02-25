from dataclasses import dataclass, field
from typing import Dict

@dataclass
class HParams:
    epochs: int = 20
    learning_rate: float = 1e-4
    batch_size: int = 4096 * 8
    seq_len: int = 10
    model: str = 'TransformerAD'
    loss: str = 'L1Loss'
    target: str = 'val_loss'
    direction: str = 'minimize'

@dataclass
class TransformerADConfig:
    input_size: int =  None
    seq_len: int = None
    d_model: int = 256
    num_heads: int = 8
    num_transformer_layers: int =  4
    dropout: float = 0.1

@dataclass
class AutoEncoderConfig:
    input_size: int = None
    seq_len: int = None
    hidden_size: int = 128
    dropout_rate: float = 0.1

@dataclass
class OptimizerConfig:
    type: str = "Adam"
    params: Dict = field(default_factory=lambda: {"betas": [0.9, 0.999],
                                                  "eps": 1e-8,
                                                  "weight_decay": 0.0})
