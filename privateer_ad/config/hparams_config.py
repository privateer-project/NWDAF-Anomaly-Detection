from dataclasses import dataclass, field
from typing import Sequence, List

from optuna.distributions import CategoricalChoiceType

@dataclass
class HParams:
    model: str = 'AttentionAutoencoder'
    batch_size: int = 4096
    seq_len: int = 12
    learning_rate: float = 0.0001
    epochs: int = 500
    loss: str = 'L1Loss'
    early_stopping: bool = True
    target: str = 'val_loss'
    direction: str = 'minimize'
    apply_dp: bool = True
    use_pca: bool = False

@dataclass
class EarlyStoppingConfig:
    es_patience_epochs: int = 20
    es_warmup_epochs: int = 20


@dataclass
class AlertFilterConfig:
    input_dim: int = 16  # Should match the latent dimension of the autoencoder
    hidden_dims: List[int] = field(default_factory=lambda: [32, 16])
    dropout: float = 0.2
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100
    model_type: str = 'autoencoder'  # 'classifier' or 'autoencoder'

@dataclass
class AlertFilterAEConfig:
    input_dim: int = 16  # Should match the latent dimension of the autoencoder
    latent_dim: int = 4  # Compress further
    hidden_dims: List[int] = field(default_factory=lambda: [32, 16])
    dropout: float = 0.01
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100
    reconstruction_threshold: float = None  # Set during training

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
