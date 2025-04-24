from dataclasses import dataclass, field
from typing import Dict, Sequence

from optuna.distributions import CategoricalChoiceType

@dataclass
class HParams:
    model: str = 'AttentionAutoencoder'
    batch_size: int = 4096
    seq_len: int = 12
    learning_rate: float = 0.001
    epochs: int = 500
    loss: str = 'L1Loss'
    early_stopping: bool = True
    target: str = 'val_loss'
    direction: str = 'minimize'
    apply_dp: bool = False
    use_pca: bool = False

@dataclass
class AttentionAutoencoderConfig:
    input_size: int = 8
    seq_len: int = 12
    num_layers: int = 1
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

@dataclass
class AutotuneParam:
    name: str
    type: str
    choices: Sequence[CategoricalChoiceType] = None
    low: int | float | None = None
    high: int | float | None = None
    step: int | float | None = None
    q: float = None
    log: bool = False

class AutotuneParams:
    params: list[AutotuneParam] = [AutotuneParam(name='seq_len', type='categorical', choices=[1, 2, 6, 12, 24, 120]),
                                   AutotuneParam(name='num_layers', type='categorical', choices=[1, 2, 3, 4]),
                                   AutotuneParam(name='hidden_dim', type='categorical', choices=[16, 32, 64, 128]),
                                   AutotuneParam(name='latent_dim', type='categorical', choices=[8, 16, 32, 64]),
                                   AutotuneParam(name='num_heads', type='categorical', choices=[1, 2, 4, 8]),
                                   AutotuneParam(name='dropout', type='float', low=0., high=0.5, step=0.05)]
