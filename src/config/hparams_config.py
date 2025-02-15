from dataclasses import dataclass
from typing import Dict


@dataclass
class HParams:
    epochs: int = 1
    learning_rate: float = 1e-3
    batch_size: int = 1024
    seq_len: int = 32
    model: str = 'TransformerAD'
    loss: str = 'L1Loss'
    target: str = 'val_loss'
    direction: str = 'minimize'

@dataclass
class TransformerADConfig:
    d_input: int =  4
    seq_len: int = 32
    d_model: int = 256
    num_heads: int = 4
    num_layers: int =  2
    dropout: float = 0.12

@dataclass
class LSTMAutoencoderConfig:
   input_size: int = 4
   hidden_size1: int = 64
   hidden_size2: int = 64
   num_layers: int = 2
   dropout: float = 0.1

@dataclass
class OptimizerConfig:
    type: str = "Adam"
    params: Dict = None

    def __post_init__(self):
        if self.params is None:
            self.params = {
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": 0.0
            }
