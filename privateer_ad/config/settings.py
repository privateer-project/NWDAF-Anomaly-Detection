"""
Centralized configuration system for PRIVATEER AD
All configurations are defined here with clear organization and type safety.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings
import yaml


# =============================================================================
# CORE CONFIGURATION REGISTRY
# =============================================================================

class PrivateerConfig:
    """Central configuration registry for the entire application"""

    def __init__(self):
        self._configs = {}
        self._load_all_configs()

    def _load_all_configs(self):
        """Load all configuration sections"""
        self._configs = {
            'paths': PathConfig(),
            'model': ModelConfig(),
            'training': TrainingConfig(),
            'data': DataConfig(),
            'federated_learning': FederatedLearningConfig(),
            'privacy': PrivacyConfig(),
            'mlflow': MLFlowConfig(),
        }

    def get(self, config_name: str) -> BaseModel:
        """Get a specific configuration section"""
        if config_name not in self._configs:
            raise ValueError(f"Configuration '{config_name}' not found. Available: {list(self._configs.keys())}")
        return self._configs[config_name]

    def reload(self):
        """Reload all configurations (useful for testing)"""
        self._load_all_configs()

    def get_all(self) -> Dict[str, BaseModel]:
        """Get all configurations"""
        return self._configs.copy()

    def summary(self) -> str:
        """Get a summary of all configurations"""
        summary = "PRIVATEER Configuration Summary\n"
        summary += "=" * 50 + "\n"
        for name, config in self._configs.items():
            summary += f"\n{name.upper()}:\n"
            for field_name, field_value in config.dict().items():
                summary += f"  {field_name}: {field_value}\n"
        return summary


# =============================================================================
# ENVIRONMENT SETTINGS
# =============================================================================

class EnvironmentSettings(BaseSettings):
    """Base class for environment-based settings"""

    def __init__(self, _env_prefix=None,
                 _env_file=None,
                 _env_file_encoding=None,
                 _case_sensitive=None,
                 **values):
        if not _env_file:
            _env_file = Path(__file__).parent.joinpath('.env')
        if not _env_file_encoding:
            _env_file_encoding = 'utf-8'
        if not _case_sensitive:
            _case_sensitive = False

        super().__init__(_env_prefix=_env_prefix,
                         _env_file=_env_file,
                         _env_file_encoding=_env_file_encoding,
                         _case_sensitive=_case_sensitive,
                         **values)


# =============================================================================
# PATH CONFIGURATION
# =============================================================================

class PathConfig(EnvironmentSettings):
    """All file and directory paths"""

    # Base paths
    root_dir: Path = Field(default_factory=lambda: Path(__file__).parents[2])
    data_dir: Optional[Path] = None

    # Derived paths
    raw_dir: Optional[Path] = None
    processed_dir: Optional[Path] = None
    experiments_dir: Optional[Path] = None
    scalers_dir: Optional[Path] = None
    models_dir: Optional[Path] = None
    analysis_dir: Optional[Path] = None

    # Dataset paths
    raw_dataset: Optional[Path] = None
    data_url: str = "https://zenodo.org/api/records/13900057/files-archive"

    def __init__(self, _env_prefix='PRIVATEER_', **kwargs):
        super().__init__(_env_prefix=_env_prefix, **kwargs)
        self._setup_derived_paths()

    def _setup_derived_paths(self):
        """Setup derived paths based on root_dir"""
        if self.data_dir is None:
            self.data_dir = self.root_dir.joinpath('data')

        if self.raw_dir is None:
            self.raw_dir = self.data_dir.joinpath('raw')

        if self.processed_dir is None:
            self.processed_dir = self.data_dir.joinpath('processed')

        if self.experiments_dir is None:
            self.experiments_dir = self.root_dir.joinpath('experiments')

        if self.scalers_dir is None:
            self.scalers_dir = self.root_dir.joinpath('scalers')

        if self.models_dir is None:
            self.models_dir = self.root_dir.joinpath('models')

        if self.analysis_dir is None:
            self.analysis_dir = self.root_dir.joinpath('analysis_results')

        if self.raw_dataset is None:
            self.raw_dataset = self.raw_dir.joinpath('amari_ue_data_merged_with_attack_number.csv')


# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

class ModelConfig(BaseModel):
    """Model architecture and hyperparameters"""

    # Model selection
    model_type: str = Field(default='TransformerAD', description="Type of model to use")

    # Architecture parameters
    input_size: int = Field(default=8, description="Number of input features")
    seq_len: int = Field(default=12, description="Sequence length for time series")
    num_layers: int = Field(default=1, description="Number of transformer layers")
    hidden_dim: int = Field(default=32, description="Hidden dimension size")
    latent_dim: int = Field(default=16, description="Latent dimension size")
    num_heads: int = Field(default=1, description="Number of attention heads")
    dropout: float = Field(default=0.2, ge=0.0, le=1.0, description="Dropout rate")

    @field_validator('dropout')
    def validate_dropout(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError('Dropout must be between 0.0 and 1.0')
        return v


# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================

class TrainingConfig(BaseModel):
    """Training process configuration"""

    # Basic training parameters
    batch_size: int = Field(default=4096, gt=0, description="Training batch size")
    learning_rate: float = Field(default=0.001, gt=0.0, description="Learning rate")
    epochs: int = Field(default=500, gt=0, description="Number of training epochs")
    loss_function: str = Field(default='L1Loss', description="Loss function to use")

    # Early stopping
    early_stopping_enabled: bool = Field(default=True, description="Enable early stopping")
    early_stopping_patience: int = Field(default=20, gt=0, description="Early stopping patience")
    early_stopping_warmup: int = Field(default=10, gt=0, description="Early stopping warmup epochs")

    # Optimization target
    target_metric: str = Field(default='val_loss', description="Metric to optimize")
    optimization_direction: str = Field(default='minimize', description="Optimization direction")

    @field_validator('optimization_direction')
    def validate_direction(cls, v):
        if v not in ['minimize', 'maximize']:
            raise ValueError('Direction must be either "minimize" or "maximize"')
        return v


# =============================================================================
# DATA CONFIGURATION
# =============================================================================

class DataConfig(BaseModel):
    """Data processing and loading configuration"""

    # Dataset splitting
    train_size: float = Field(default=0.8, gt=0.0, lt=1.0, description="Training set proportion")
    val_size: float = Field(default=0.1, gt=0.0, lt=1.0, description="Validation set proportion")
    test_size: float = Field(default=0.1, gt=0.0, lt=1.0, description="Test set proportion")

    # Data processing
    only_benign_for_training: bool = Field(default=True, description="Use only benign data for training")
    apply_scaling: bool = Field(default=True, description="Apply feature scaling")
    random_seed: int = Field(default=42, description="Random seed for reproducibility")

    # Loading
    num_workers: int = Field(default=4, ge=0, description="Number of data loading workers")
    pin_memory: bool = Field(default=True, description="Pin memory for faster GPU transfer")

    @field_validator('train_size', 'val_size', 'test_size')
    def validate_proportions(cls, v):
        if not 0.0 < v < 1.0:
            raise ValueError('Dataset proportions must be between 0 and 1')
        return v


# =============================================================================
# FEDERATED LEARNING CONFIGURATION
# =============================================================================

class FederatedLearningConfig(BaseModel):
    """Federated learning specific settings"""

    # Server settings
    server_address: str = Field(default="[::]:8081", description="FL server address")
    num_rounds: int = Field(default=100, gt=0, description="Number of FL rounds")
    min_clients: int = Field(default=2, gt=0, description="Minimum number of clients")
    fraction_fit: float = Field(default=1.0, gt=0.0, le=1.0, description="Fraction of clients for training")
    fraction_evaluate: float = Field(default=1.0, gt=0.0, le=1.0, description="Fraction of clients for evaluation")

    # Secure aggregation
    secure_aggregation_enabled: bool = Field(default=True, description="Enable secure aggregation")
    num_shares: int = Field(default=3, gt=1, description="Number of secret shares")
    reconstruction_threshold: int = Field(default=2, gt=0, description="Reconstruction threshold")
    secagg_timeout: int = Field(default=300, gt=0, description="SecAgg timeout in seconds")
    secagg_max_weight: int = Field(default=200000, gt=0, description="Maximum weight for secure aggregation")

    # Client settings
    epochs_per_round: int = Field(default=1, gt=0, description="Epochs per FL round")
    partition_data: bool = Field(default=True, description="Partition data across clients")

    @field_validator('reconstruction_threshold')
    def validate_reconstruction_threshold(cls, v, info):
        # Note: We can't access num_shares here directly, so this validation
        # will be done in the main validate_config() function
        return v
# =============================================================================
# PRIVACY CONFIGURATION
# =============================================================================

class PrivacyConfig(BaseModel):
    """Privacy and differential privacy settings"""

    # Differential Privacy
    dp_enabled: bool = Field(default=False, description="Enable differential privacy")
    target_epsilon: float = Field(default=0.5, gt=0.0, description="DP epsilon parameter")
    target_delta: float = Field(default=1e-7, gt=0.0, description="DP delta parameter")
    max_grad_norm: float = Field(default=0.5, gt=0.0, description="Gradient clipping threshold")
    secure_mode: bool = Field(default=True, description="Enable secure RNG for DP")

    # Anonymization
    anonymization_enabled: bool = Field(default=False, description="Enable data anonymization")
    location_privacy_enabled: bool = Field(default=False, description="Enable location privacy")


# =============================================================================
# MLFLOW CONFIGURATION
# =============================================================================

class MLFlowConfig(EnvironmentSettings):
    """MLFlow experiment tracking configuration"""

    enabled: bool = Field(default=True, description="Enable MLFlow tracking")
    server_address: str = Field(default="http://localhost:5001", description="MLFlow server address")
    experiment_name: str = Field(default="privateer-ad", description="MLFlow experiment name")
    server_run_name: str = Field(default="federated-learning", description="Server run name")

    def __init__(self, _env_prefix='MLFLOW_', **values):
        super().__init__(_env_prefix=_env_prefix, **values)

# =============================================================================
# METADATA CONFIGURATION (from YAML)
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
        if metadata_path is None:
            metadata_path = Path(__file__).parent.joinpath('metadata.yaml')

        self.metadata_path = metadata_path
        self._load_metadata()

    def _load_metadata(self):
        """Load metadata from YAML file"""
        with self.metadata_path.open() as f:
            data = yaml.safe_load(f)

        self.devices = {
            k: DeviceInfo(**v) for k, v in data['devices'].items()
        }

        self.attacks = {
            k: AttackInfo(**v) for k, v in data['attacks'].items()
        }

        self.features = {
            k: FeatureInfo(**v) for k, v in data['features'].items()
        }

    def get_input_features(self) -> List[str]:
        """Get list of input features"""
        return [
            feat for feat, info in self.features.items()
            if info.is_input
        ]

    def get_drop_features(self) -> List[str]:
        """Get list of features to drop"""
        return [
            feat for feat, info in self.features.items()
            if info.drop
        ]

    def get_features_dtypes(self) -> Dict[str, str]:
        """Get feature data types"""
        return {
            feat: info.dtype for feat, info in self.features.items()
        }


# =============================================================================
# AUTOTUNING CONFIGURATION
# =============================================================================

class AutotuningConfig(BaseModel):
    """Hyperparameter autotuning configuration"""

    # Study settings
    study_name: str = Field(default='privateer_study', description="Optuna study name")
    storage_url: str = Field(default='sqlite:///optuna.db', description="Optuna storage URL")
    n_trials: int = Field(default=10, gt=0, description="Number of optimization trials")
    timeout: int = Field(default=3600 * 8, gt=0, description="Timeout in seconds")

    # Optimization
    target_metric: str = Field(default='f1-score', description="Metric to optimize")
    direction: str = Field(default='maximize', description="Optimization direction")

    # Search space (simplified - can be extended)
    tune_seq_len: bool = Field(default=True, description="Tune sequence length")
    tune_architecture: bool = Field(default=True, description="Tune model architecture")
    tune_training: bool = Field(default=True, description="Tune training parameters")


# =============================================================================
# GLOBAL CONFIGURATION INSTANCE
# =============================================================================

# Global configuration instance - singleton pattern
_config_instance = None


def get_config() -> PrivateerConfig:
    """Get the global configuration instance"""
    global _config_instance
    if _config_instance is None:
        _config_instance = PrivateerConfig()
    return _config_instance


def reload_config():
    """Reload the global configuration"""
    global _config_instance
    _config_instance = None
    return get_config()


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_paths() -> PathConfig:
    """Get path configuration"""
    return get_config().get('paths')


def get_model_config() -> ModelConfig:
    """Get model configuration"""
    return get_config().get('model')


def get_training_config() -> TrainingConfig:
    """Get training configuration"""
    return get_config().get('training')


def get_data_config() -> DataConfig:
    """Get data configuration"""
    return get_config().get('data')


def get_fl_config() -> FederatedLearningConfig:
    """Get federated learning configuration"""
    return get_config().get('federated_learning')


def get_privacy_config() -> PrivacyConfig:
    """Get privacy configuration"""
    return get_config().get('privacy')


def get_mlflow_config() -> MLFlowConfig:
    """Get MLFlow configuration"""
    return get_config().get('mlflow')


def get_metadata() -> MetadataConfig:
    """Get metadata configuration"""
    return MetadataConfig()


# =============================================================================
# CONFIGURATION VALIDATION
# =============================================================================

def validate_config():
    """Validate the entire configuration for consistency"""
    config = get_config()
    errors = []

    # Validate data splits sum to 1.0
    data_cfg = config.get('data')
    total_split = data_cfg.train_size + data_cfg.val_size + data_cfg.test_size
    if abs(total_split - 1.0) > 0.001:
        errors.append(f"Data splits don't sum to 1.0: {total_split}")

    # Validate FL configuration
    fl_cfg = config.get('federated_learning')
    if fl_cfg.reconstruction_threshold > fl_cfg.num_shares:
        errors.append(
            f"Reconstruction threshold ({fl_cfg.reconstruction_threshold}) cannot be greater than num_shares ({fl_cfg.num_shares})")

    # Validate model configuration with training
    model_cfg = config.get('model')
    training_cfg = config.get('training')
    if model_cfg.seq_len <= 0:
        errors.append(f"Sequence length must be positive: {model_cfg.seq_len}")

    if errors:
        raise ValueError(f"Configuration validation errors:\n" + "\n".join(f"- {error}" for error in errors))

    return True


# =============================================================================
# DEVELOPMENT HELPERS
# =============================================================================

def print_config_summary():
    """Print a summary of all configurations"""
    config = get_config()
    print(config.summary())


def save_config_to_file(filepath: Union[str, Path]):
    """Save current configuration to a YAML file"""
    config = get_config()
    config_dict = {}

    for name, cfg in config.get_all().items():
        config_dict[name] = cfg.model_dump()

    with open(filepath, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, indent=2)


def load_config_from_file(filepath: Union[str, Path]):
    """Load configuration from a YAML file (for testing/development)"""
    pass