"""
Configuration files
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Literal, Any
from dataclasses import dataclass
from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings
import yaml

logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION CLASSES
# =============================================================================

class PathConfig(BaseSettings):
    """All file and directory paths using package-based root resolution"""

    root_dir: Path = Field(default_factory=lambda: PathConfig._get_package_root())
    data_url: str = "https://zenodo.org/api/records/13900057/files-archive"

    model_config = {'env_prefix': 'PRIVATEER_',
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


class ModelConfig(BaseModel):
    """Model architecture and hyperparameters"""

    model_type: str = Field(default='TransformerAD')
    input_size: int = Field(default=8)
    seq_len: int = Field(default=12)
    num_layers: int = Field(default=1)
    hidden_dim: int = Field(default=32)
    latent_dim: int = Field(default=16)
    num_heads: int = Field(default=1)
    dropout: float = Field(default=0.2, ge=0.0, le=1.0)


class TrainingConfig(BaseModel):
    """Training process configuration"""

    batch_size: int = Field(default=4096, gt=0)
    learning_rate: float = Field(default=0.001, gt=0.0)
    epochs: int = Field(default=500, gt=0)
    loss_function: str = Field(default='L1Loss')

    # Early stopping
    early_stopping_enabled: bool = Field(default=True)
    early_stopping_patience: int = Field(default=20, gt=0)
    early_stopping_warmup: int = Field(default=10, gt=0)

    # Optimization
    target_metric: str = Field(default='val_loss')
    optimization_direction: Literal['minimize', 'maximize'] = Field(default='minimize')

    model_config = {'env_prefix': 'PRIVATEER_',
                    'env_file': '.env',
                    'extra': 'ignore',
                    'case_sensitive': False}

class DataConfig(BaseModel):
    """Data processing and loading configuration"""

    train_size: float = Field(default=0.8, gt=0.0, lt=1.0)
    val_size: float = Field(default=0.1, gt=0.0, lt=1.0)
    test_size: float = Field(default=0.1, gt=0.0, lt=1.0)

    only_benign_for_training: bool = Field(default=True)
    apply_scaling: bool = Field(default=True)
    random_seed: int = Field(default=42)

    num_workers: int = Field(default=4, ge=0)
    pin_memory: bool = Field(default=True)

    @model_validator(mode='after')
    def validate_splits_sum_to_one(self):
        total = self.train_size + self.val_size + self.test_size
        if abs(total - 1.0) > 0.001:
            raise ValueError(f'Dataset splits must sum to 1.0, got {total}')
        return self


class AutotuningConfig(BaseModel):
    """Hyperparameter optimization configuration"""

    study_name: str = Field(default="privateer-autotune", description="Name for the Optuna study")
    n_trials: int = Field(default=50, gt=0, description="Number of optimization trials")
    timeout: Optional[int] = Field(default=None, description="Timeout in seconds for optimization")

    target_metric: str = Field(default="eval_test_f1-score", description="Metric to optimize")
    direction: Literal["minimize", "maximize"] = Field(default="maximize", description="Optimization direction")

    storage_url: str = Field(default="sqlite:///optuna_study.db", description="Optuna storage URL")

    enable_pruning: bool = Field(default=True, description="Enable trial pruning")
    pruning_warmup_steps: int = Field(default=5, gt=0, description="Steps before pruning can start")

    # Sampler configuration
    sampler_type: Literal["tpe", "random", "cmaes"] = Field(default="tpe", description="Type of sampler to use")

    model_config = {
        'env_prefix': 'AUTOTUNE_',
        'env_file': '.env',
        'extra': 'ignore',
        'case_sensitive': False
    }


class FederatedLearningConfig(BaseModel):
    """Federated learning settings"""

    server_address: str = Field(default="[::]:8081")
    num_rounds: int = Field(default=100, gt=0)
    min_clients: int = Field(default=2, gt=0)
    fraction_fit: float = Field(default=1.0, gt=0.0, le=1.0)
    fraction_evaluate: float = Field(default=1.0, gt=0.0, le=1.0)

    secure_aggregation_enabled: bool = Field(default=True)
    num_shares: int = Field(default=3, gt=1)
    reconstruction_threshold: int = Field(default=2, gt=0)
    secagg_timeout: int = Field(default=300, gt=0)
    secagg_max_weight: int = Field(default=200000, gt=0)

    epochs_per_round: int = Field(default=1, gt=0)
    partition_data: bool = Field(default=True)

    model_config = {'env_prefix': 'FL_',
                    'env_file': '.env',
                    'extra': 'ignore',
                    'case_sensitive': False}


class PrivacyConfig(BaseModel):
    """Privacy and differential privacy settings"""

    dp_enabled: bool = Field(default=False)
    target_epsilon: float = Field(default=0.5, gt=0.0)
    target_delta: float = Field(default=1e-7, gt=0.0)
    max_grad_norm: float = Field(default=0.5, gt=0.0)
    secure_mode: bool = Field(default=True)

    anonymization_enabled: bool = Field(default=False)
    location_privacy_enabled: bool = Field(default=False)

    model_config = {'env_prefix': 'DP_',
                    'env_file': '.env',
                    'extra': 'ignore',
                    'case_sensitive': False}


class MLFlowConfig(BaseSettings):
    """MLFlow experiment tracking configuration"""

    enabled: bool = Field(default=True, description="Enable MLFlow tracking")
    server_address: str = Field(default="http://localhost:5001", description="MLFlow server address")
    experiment_name: str = Field(default="privateer-ad", description="MLFlow experiment name")
    server_run_name: str = Field(default="federated-learning", description="Server run name")

    model_config = {'env_prefix': 'MLFLOW_',
                    'env_file': '.env',
                    'extra': 'ignore',
                    'case_sensitive': False}



# =============================================================================
# METADATA CONFIGURATION
# =============================================================================

@dataclass
class DeviceInfo:
    """Device information"""
    imeisv: str
    ip: str
    type: str
    malicious: bool
    in_attacks: List[int]


@dataclass
class AttackInfo:
    """Attack information"""
    start: str
    stop: str


@dataclass
class FeatureInfo:
    """Feature configuration"""
    dtype: str = 'str'
    drop: bool = False
    is_input: bool = False
    process: List[str] = None

    def __post_init__(self):
        if self.process is None:
            self.process = []


class MetadataConfig:
    """Configuration loaded from metadata.yaml"""

    def __init__(self, metadata_path: Optional[Path] = None):
        data = self._load_metadata(metadata_path)
        self._parse_metadata(data)

    def _load_metadata(self, metadata_path: Optional[Path]) -> dict:
        """Load metadata from file or package resources"""

        if metadata_path is not None:
            with metadata_path.open() as f:
                return yaml.safe_load(f)

        import importlib.resources as resources
        try:
            # Try to find metadata.yaml in the config package
            config_package = resources.files('privateer_ad.config')
            metadata_file = config_package.joinpath('metadata.yaml')

            logger.info(f'Looking for metadata at: {metadata_file}')

            if metadata_file.is_file():
                content = metadata_file.read_text(encoding='utf-8')
                logger.info('Loaded metadata.yaml from package resources')
                return yaml.safe_load(content)
            else:
                # Fallback: try to find it in the project root
                root_package = resources.files('privateer_ad')
                with resources.as_file(root_package) as pkg_path:
                    metadata_file_alt = pkg_path.joinpath('privateer_ad', 'config', 'metadata.yaml')

                    if metadata_file_alt.exists():
                        with metadata_file_alt.open('r', encoding='utf-8') as f:
                            logger.info('Loaded metadata.yaml from project directory')
                            return yaml.safe_load(f)
                    else:
                        raise FileNotFoundError("metadata.yaml not found, using default configuration")
        except Exception as e:
            logger.error(f"Could not load metadata.yaml from package: {e}, using defaults")
            raise e

    def _parse_metadata(self, data: dict):
        """Parse loaded data into structured objects"""
        self.devices = {k: DeviceInfo(**v) for k, v in data['devices'].items()}
        self.attacks = {k: AttackInfo(**v) for k, v in data['attacks'].items()}
        self.features = {k: FeatureInfo(**v) for k, v in data['features'].items()}

    def get_input_features(self) -> List[str]:
        return [feat for feat, info in self.features.items() if info.is_input]

    def get_drop_features(self) -> List[str]:
        return [feat for feat, info in self.features.items() if info.drop]

    def get_features_dtypes(self) -> Dict[str, str]:
        return {feat: info.dtype for feat, info in self.features.items()}


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