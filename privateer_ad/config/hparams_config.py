from dataclasses import dataclass, field
from typing import Dict

@dataclass
class HParams:
    model: str = 'AttentionAutoencoder'
    batch_size: int = 4096
    seq_len: int = 12
    learning_rate: float = 0.001
    epochs: int = 1  # 50 epochs * 10 aggs --> 500 epochs \approx 500 epochs used on models without FL
    loss: str = 'L1Loss'
    early_stopping: bool = True
    target: str = 'val_loss'
    direction: str = 'minimize'
    apply_dp: bool = False
    use_pca: bool = False

@dataclass
class AttentionAutoencoderConfig:
    input_size: int =  8
    seq_len: int = 12
    num_layers: int =  1
    hidden_dim: int = 32
    latent_dim: int = 16
    num_heads: int = 1
    dropout: float = 0.1

@dataclass
class EarlyStoppingConfig:
    es_patience_epochs: int = 20
    es_warmup_epochs: int = 20
    es_improvement_threshold: int = 0.0001

@dataclass
class OptimizerConfig:
    name: str = "Adam"
    params: Dict = field(default_factory=lambda: {"betas": [0.9, 0.999],
                                                  "eps": 1e-8,
                                                  "weight_decay": 0.0})
