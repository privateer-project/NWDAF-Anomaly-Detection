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
    data_url: str = "https://zenodo.org/api/records/13900057/files-archive"

    model_config = {'env_prefix': 'PVTR_PATH_',
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
    def raw_dataset(self) -> Path:
        return self.raw_dir.joinpath('amari_ue_data_merged_with_attack_number.csv')


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

    random_seed: int = Field(default=42)
    batch_size: int = Field(default=1024, gt=0)
    seq_len: int = Field(default=12, ge=1)

    num_workers: int = Field(default=4, ge=0)
    pin_memory: bool = Field(default=True)
    prefetch_factor: int | None = Field(default=10000, ge=1)
    persistent_workers: bool = Field(default=True)

    @model_validator(mode='after')
    def validate_splits_sum_to_one(self):
        total = self.train_size + self.val_size
        if 1.0 - total < 0.:
            raise ValueError(f'Train and val sizes must be between 0 and 1. Total: {total}')
        return self

    model_config = {'env_prefix': 'PVTR_DATA_',
                    'env_file': '.env',
                    'extra': 'ignore',
                    'case_sensitive': False}


class ModelConfig(BaseSettings):
    """Model architecture and hyperparameters"""

    model_type: str = Field(default='TransformerAD')
    input_size: int = Field(default=9, ge=1)
    num_layers: int = Field(default=1, ge=1)
    embed_dim: int = Field(default=32, ge=1)
    latent_dim: int = Field(default=16, ge=1)
    num_heads: int = Field(default=1, ge=1)
    dropout: float = Field(default=0.2, ge=0.0, le=1.0)
    seq_len: int = Field(default=12, ge=1)

    model_config = {'env_prefix': 'PVTR_MODEL_',
                    'env_file': '.env',
                    'extra': 'ignore',
                    'case_sensitive': False}


class TrainingConfig(BaseSettings):
    """Training process configuration"""

    learning_rate: float = Field(default=0.0007, gt=0.0)
    epochs: int = Field(default=120, gt=0)
    loss_fn: str = Field(default='L1Loss')

    # Early stopping
    early_stopping_enabled: bool = Field(default=True)
    early_stopping_patience: int = Field(default=20, gt=0)
    early_stopping_warmup: int = Field(default=10, gt=0)

    # Optimization
    target_metric: str = Field(default='val_f1-score')
    direction: Literal['minimize', 'maximize'] = Field(default='maximize')

    model_config = {'env_prefix': 'PVTR_TRAIN_',
                    'env_file': '.env',
                    'extra': 'ignore',
                    'case_sensitive': False}


class AutotuningConfig(BaseSettings):
    """Hyperparameter optimization configuration"""

    study_name: str = Field(default="privateer-autotune", description="Name for the Optuna study")
    n_trials: int = Field(default=30, gt=0, description="Number of optimization trials")
    timeout: Optional[int] = Field(default=None, description="Timeout in seconds for optimization")
    target_metric: str = Field(default="test_f1-score", description="Metric to optimize")
    direction: Literal["minimize", "maximize"] = Field(default="maximize", description="Optimization direction")

    storage_url: str = Field(default="sqlite:///optuna_study.db", description="Optuna storage URL")

    enable_pruning: bool = Field(default=True, description="Enable trial pruning")
    pruning_warmup_steps: int = Field(default=5, gt=0, description="Steps before pruning can start")

    # Sampler configuration
    sampler_type: Literal["tpe", "random", "cmaes"] = Field(default="tpe", description="Type of sampler to use")

    model_config = {'env_prefix': 'PVTR_AUTO_',
                    'env_file': '.env',
                    'extra': 'ignore',
                    'case_sensitive': False}


class FederatedLearningConfig(BaseSettings):
    """Federated learning settings"""

    server_address: str = Field(default="[::]:8081")
    num_rounds: int = Field(default=100, gt=0)
    n_clients: int = Field(default=2, gt=0)
    fraction_fit: float = Field(default=1.0, gt=0.0, le=1.0)
    fraction_evaluate: float = Field(default=1.0, gt=0.0, le=1.0)

    secure_aggregation_enabled: bool = Field(default=True)
    num_shares: int = Field(default=3, gt=1)
    reconstruction_threshold: int = Field(default=2, gt=0)
    secagg_timeout: int = Field(default=300, gt=0)
    secagg_max_weight: int = Field(default=200000, gt=0)

    epochs_per_round: int = Field(default=1, gt=0)
    partition_data: bool = Field(default=True)

    model_config = {'env_prefix': 'PVTR_FL_',
                    'env_file': '.env',
                    'extra': 'ignore',
                    'case_sensitive': False}


class PrivacyConfig(BaseSettings):
    """Privacy and differential privacy settings"""

    enabled: bool = Field(default=False)
    target_epsilon: float = Field(default=0.5, gt=0.0)
    target_delta: float = Field(default=1e-7, gt=0.0)
    max_grad_norm: float = Field(default=0.5, gt=0.0)
    secure_mode: bool = Field(default=True)

    anonymization_enabled: bool = Field(default=False)
    location_privacy_enabled: bool = Field(default=False)

    model_config = {'env_prefix': 'PVTR_DP_',
                    'env_file': '.env',
                    'extra': 'ignore',
                    'case_sensitive': False}


class MLFlowConfig(BaseSettings):
    """MLFlow experiment tracking configuration"""

    enabled: bool = Field(default=True, description="Enable MLFlow tracking")
    server_address: str = Field(default="http://localhost:5001", description="MLFlow server address")
    experiment_name: str = Field(default="privateer-ad", description="MLFlow experiment name")
    server_run_id: str | None = Field(default=None, description="Server run id")
    client_run_id: str | None = Field(default=None, description="Client run id")

    model_config = {'env_prefix': 'PVTR_MLFLOW_',
                    'env_file': '.env',
                    'extra': 'ignore',
                    'case_sensitive': False}

# =============================================================================
# SIMPLE CONFIGURATION ACCESS
# =============================================================================

# Direct instantiation - no complex registry pattern
def get_paths() -> PathConfig:
    """Get path configuration"""
    return PathConfig()

def get_model_config() -> ModelConfig:
    """Get model configuration"""
    return ModelConfig()

def get_training_config() -> TrainingConfig:
    """Get training configuration"""
    return TrainingConfig()

def get_autotuning_config() -> AutotuningConfig:
    """Get autotuning configuration"""
    return AutotuningConfig()

def get_data_config() -> DataConfig:
    """Get data configuration"""
    return DataConfig()

def get_fl_config() -> FederatedLearningConfig:
    """Get federated learning configuration"""
    return FederatedLearningConfig()

def get_privacy_config() -> PrivacyConfig:
    """Get privacy configuration"""
    return PrivacyConfig()

def get_mlflow_config() -> MLFlowConfig:
    """Get MLFlow configuration"""
    return MLFlowConfig()

def get_metadata() -> MetadataConfig:
    """Get metadata configuration"""
    return MetadataConfig()


# =============================================================================
# CONFIGURATION VALIDATION
# =============================================================================

def validate_config():
    """Validate the entire configuration for consistency"""
    errors = []

    # Basic validation - individual configs handle their own validation
    try:
        fl_cfg = get_fl_config()
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