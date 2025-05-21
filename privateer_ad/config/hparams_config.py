from dataclasses import dataclass
from typing import  Sequence

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

@dataclass
class EarlyStoppingConfig:
    es_patience_epochs: int = 20
    es_warmup_epochs: int = 10


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
