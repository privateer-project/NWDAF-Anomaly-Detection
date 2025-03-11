from dataclasses import dataclass, field
from typing import Dict

@dataclass
class HParams:
    epochs: int = 100
    learning_rate: float = 1e-4
    batch_size: int = 4096
    seq_len: int = 15
    model: str = 'AttentionAutoencoder'
    loss: str = 'L1Loss'
    target: str = 'val_loss'
    direction: str = 'minimize'
    early_stopping: bool = True
    apply_dp: bool = False

@dataclass
class AttentionAutoencoderConfig:
    input_size: int =  7
    num_layers: int =  4
    hidden_dim: int = 32
    latent_dim: int = 16
    num_heads: int = 8
    dropout: float = 0.1

@dataclass
class AutoEncoderConfig:
    input_size: int = None
    seq_len: int = None
    hidden_size: int = 128
    dropout_rate: float = 0.1

@dataclass
class SimpleTransformerConfig:
    input_size: int = None
    seq_len: int = None
    d_model: int = 256
    nhead: int = 2
    num_layers: int = 4
    dropout: float = 0.1

@dataclass
class EarlyStoppingConfig:
    es_patience_epochs: int = 20
    es_warmup_epochs: int = 20
    es_improvement_threshold: int = 0.0001

@dataclass
class OptimizerConfig:
    type: str = "Adam"
    params: Dict = field(default_factory=lambda: {"betas": [0.9, 0.999],
                                                  "eps": 1e-8,
                                                  "weight_decay": 0.0})
