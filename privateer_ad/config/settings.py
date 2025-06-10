"""
Configuration files
"""

from pathlib import Path
from typing import Optional, Literal
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings

from .metadata import MetadataConfig

# =============================================================================
# CONFIGURATION CLASSES
# =============================================================================

class PathConfig(BaseSettings):
    """All file and directory paths using package-based root resolution"""
    root_dir: Path = Field(default_factory=lambda: PathConfig._get_package_root())
    data_url: str = Field(default="https://zenodo.org/api/records/13900057/files-archive", description="Data URL")

    model_config = {'env_prefix': 'PRIVATEER_PATH_',
                    'env_file': '.env',
                    'extra': 'ignore',
                    'case_sensitive': False}

    @staticmethod
    def _get_package_root() -> Path:
        """Get project root using package location as reference"""
        import importlib.resources as resources
        # Get the src package location
        package_path = resources.files('privateer_ad')
        with resources.as_file(package_path) as pkg_path:
            return pkg_path.parent

    @property
    def data_dir(self) -> Path:
        return self.root_dir.joinpath('data')

    @property
    def raw_dir(self) -> Path:
        return self.data_dir.joinpath('raw')

    @property
    def processed_dir(self) -> Path:
        return self.data_dir.joinpath('processed')

    @property
    def raw_dataset(self) -> Path:
        return self.raw_dir.joinpath('amari_ue_data_merged_with_attack_number.csv')

    @property
    def zip_file(self) -> Path:
        return self.raw_dir.joinpath('nwdaf-data.zip')

    @property
    def experiments_dir(self) -> Path:
        return self.root_dir.joinpath('experiments')

    @property
    def scalers_dir(self) -> Path:
        return self.root_dir.joinpath('scalers')

    @property
    def models_dir(self) -> Path:
        return self.root_dir.joinpath('models')

    @property
    def analysis_dir(self) -> Path:
        return self.root_dir.joinpath('analysis_results')

    @property
    def requirements_file(self) -> Path:
        return self.root_dir.joinpath('requirements.txt')

class DataConfig(BaseSettings):
    """Data processing and loading configuration"""
    train_size: float = Field(default=0.8, gt=0.0, lt=1.0, description='Training set size as a fraction of the dataset. '
                                                                       'Example: "0.8" for 80% of the dataset. Default: "0.8" Test size is calculated as 1 - train_size - val_size')
    val_size: float = Field(default=0.1, gt=0.0, lt=1.0, description='Validation set size as a fraction of the dataset. '
                                                                     'Example: "0.1" for 10% of the dataset. Default: "0.1". Test size is calculated as 1 - train_size - val_size')
    partition_id: int = Field(default=-1, ge=-1, description='Partition ID for data partitioning. Default: "-1"')
    partition_by: str = Field(default='cell', description="Column to partition data by. Default: 'cell'")
    num_partitions: int = Field(default=0, ge=0, description="Number of partitions to split the data into. Default: 1")
    num_classes_per_partition: int = Field(default=1, ge=1, description="Number of classes per partition. Default: 1")

    batch_size: int = Field(default=4096, gt=0)
    seq_len: int = Field(default=12, ge=1)

    num_workers: int = Field(default=16, ge=0)
    prefetch_factor: int | None = Field(default=16, ge=1)
    persistent_workers: bool = Field(default=True)
    pin_memory: bool = Field(default=True)

    @model_validator(mode='after')
    def validate_splits_sum_to_one(self):
        total = self.train_size + self.val_size
        if 1.0 - total < 0.:
            raise ValueError(f'Train and val sizes must be between 0 and 1. Total: {total}')
        return self

    model_config = {'env_prefix': 'PRIVATEER_DATA_',
                    'env_file': '.env',
                    'extra': 'ignore',
                    'case_sensitive': False}


class ModelConfig(BaseSettings):
    """Model architecture and hyperparameters"""
    model_name: str = Field(default='TransformerAD')
    input_size: int = Field(default=9, ge=1)
    num_layers: int = Field(default=1, ge=1)
    embed_dim: int = Field(default=32, ge=1)
    latent_dim: int = Field(default=16, ge=1)
    num_heads: int = Field(default=1, ge=1)
    dropout: float = Field(default=0.2, ge=0.0, le=1.0)
    seq_len: int = Field(default=12, ge=1)

    model_config = {'env_prefix': 'PRIVATEER_MODEL_',
                    'env_file': '.env',
                    'extra': 'ignore',
                    'case_sensitive': False}


class TrainingConfig(BaseSettings):
    """Training process configuration"""
    learning_rate: float = Field(default=0.0001, gt=0.0)
    epochs: int = Field(default=100, gt=0)
    loss_fn_name: str = Field(default='L1Loss')

    # Early stopping
    es_enabled: bool = Field(default=True)
    es_patience: int = Field(default=20, gt=0)
    es_warmup: int = Field(default=10, gt=0)

    # Optimization
    target_metric: str = Field(default='val_balanced_f1-score')
    direction: Literal['minimize', 'maximize'] = Field(default='maximize')

    model_config = {'env_prefix': 'PRIVATEER_TRAIN_',
                    'env_file': '.env',
                    'extra': 'ignore',
                    'case_sensitive': False}


class AutotuningConfig(BaseSettings):
    """Hyperparameter optimization configuration"""
    study_name: str = Field(default="privateer-autotune", description="Name for the Optuna study")
    n_trials: int = Field(default=30, gt=0, description="Number of optimization trials")
    timeout: Optional[int] = Field(default=None, description="Timeout in seconds for optimization")
    target_metric: str = Field(default="val_unbalanced_f1-score", description="Metric to optimize")
    direction: Literal["minimize", "maximize"] = Field(default="maximize", description="Optimization direction")

    storage_url: str = Field(default="sqlite:///optuna_study.db", description="Optuna storage URL")

    enable_pruning: bool = Field(default=True, description="Enable trial pruning")
    pruning_warmup_steps: int = Field(default=5, gt=0, description="Steps before pruning can start")

    # Sampler configuration
    sampler_type: Literal["tpe", "random", "cmaes"] = Field(default="tpe", description="Type of sampler to use")

    model_config = {'env_prefix': 'PRIVATEER_AUTOTUNE_',
                    'env_file': '.env',
                    'extra': 'ignore',
                    'case_sensitive': False}


class FederatedLearningConfig(BaseSettings):
    """Federated learning settings"""

    server_address: str = Field(default="[::]:8081")
    num_rounds: int = Field(default=100, gt=0)
    n_clients: int = Field(default=3, gt=0)
    fraction_fit: float = Field(default=1.0, gt=0.0, le=1.0)
    fraction_evaluate: float = Field(default=1.0, gt=0.0, le=1.0)

    secure_aggregation_enabled: bool = Field(default=True)
    num_shares: int = Field(default=3, gt=1)
    reconstruction_threshold: int = Field(default=2, gt=0)
    secagg_timeout: int = Field(default=300, gt=0)
    secagg_max_weight: int = Field(default=200000, gt=0)

    epochs_per_round: int = Field(default=1, gt=0)
    partition_data: bool = Field(default=True)

    model_config = {'env_prefix': 'PRIVATEER_FL_',
                    'env_file': '.env',
                    'extra': 'ignore',
                    'case_sensitive': False}


class PrivacyConfig(BaseSettings):
    """Privacy and differential privacy settings"""

    dp_enabled: bool = Field(default=True)
    target_epsilon: float = Field(default=0.3, gt=0.0)
    target_delta: float = Field(default=1e-7, gt=0.0)
    max_grad_norm: float = Field(default=.5, gt=0.0)
    secure_mode: bool = Field(default=True)

    anonymization_enabled: bool = Field(default=False)

    model_config = {'env_prefix': 'PRIVATEER_DP_',
                    'env_file': '.env',
                    'extra': 'ignore',
                    'case_sensitive': False}


class MLFlowConfig(BaseSettings):
    """MLFlow experiment tracking configuration"""
    enabled: bool = Field(default=True, description="Enable MLFlow tracking")
    tracking_uri: str = Field(default="http://localhost:5001", description="MLFlow server address")
    experiment_name: str = Field(default="privateer-ad", description="MLFlow experiment name")
    parent_run_id: str | None = Field(default=None, description="Parent run id")
    child_run_id: str | None = Field(default=None, description="Client run id")

    model_config = {'env_prefix': 'PRIVATEER_MLFLOW_',
                    'env_file': '.env',
                    'extra': 'ignore',
                    'case_sensitive': False}

# =============================================================================
# CONFIGURATION VALIDATION
# =============================================================================

def validate_config():
    """Validate the entire configuration for consistency"""
    errors = []

    # Basic validation - individual configs handle their own validation
    try:
        fl_cfg = FederatedLearningConfig()
        if fl_cfg.reconstruction_threshold > fl_cfg.num_shares:
            errors.append(
                f"Reconstruction threshold ({fl_cfg.reconstruction_threshold}) "
                f"cannot be greater than num_shares ({fl_cfg.num_shares})"
            )
    except Exception as e:
        errors.append(f"FL config validation failed: {e}")

    if errors:
        raise ValueError("Configuration validation errors:\n" + "\n".join(f"- {error}" for error in errors))

    return True


if __name__ == '__main__':
    print(MetadataConfig().features)