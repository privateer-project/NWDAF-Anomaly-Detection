"""
Configuration management system for PRIVATEER anomaly detection framework.

This module provides comprehensive configuration management through Pydantic
BaseSettings classes, enabling type-safe parameter validation, environment
variable integration, and consistent configuration across distributed training
scenarios. The implementation supports all major framework components including
data processing, model architecture, training procedures, privacy mechanisms,
and federated learning orchestration.

The configuration system maintains flexibility for research experimentation and
production deployment scenarios. All configuration classes support environment
variable overrides and validation to ensure parameter consistency across
different execution environments.
"""
import importlib.resources as resources

from pathlib import Path
from typing import Optional, Literal, ClassVar
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings

package_path = resources.files('privateer_ad')
# Get the src package location
with resources.as_file(package_path) as pkg_path:
    root_dir: Path = pkg_path.parent


# =============================================================================
# CONFIGURATION CLASSES
# =============================================================================

class PathConfig(BaseSettings):
    """
    File system path configuration with package-based root resolution.

    This configuration class manages all file and directory paths within the
    PRIVATEER framework using package-based root directory resolution to ensure
    consistent path handling across different deployment environments.
    Attributes:
        root_dir (Path): Package root directory automatically resolved from
                        installation location
        data_url (str): Remote data source URL for dataset download operations
    """
    root_dir: Path = Field(default=root_dir, description="Package root directory")
    data_url: str = Field(default="https://zenodo.org/api/records/13900057/files-archive", description="Data URL")

    model_config = {'env_prefix': 'PRIVATEER_PATH_',
                    'env_file': '.env',
                    'extra': 'ignore',
                    'case_sensitive': False}

    @property
    def data_dir(self) -> Path:
        """Primary data directory for all dataset storage and processing."""
        return self.root_dir.joinpath('data')

    @property
    def models_dir(self) -> Path:
        """Trained model storage directory for model persistence."""
        return self.root_dir.joinpath('models')

    @property
    def experiments_dir(self) -> Path:
        """Experiment results directory for training artifacts and logs."""
        return self.root_dir.joinpath('experiments')

    @property
    def scalers_dir(self) -> Path:
        """Feature scaling parameters directory for preprocessing consistency."""
        return self.root_dir.joinpath('scalers')

    @property
    def analysis_dir(self) -> Path:
        """Analysis results directory for evaluation metrics and visualizations."""
        return self.root_dir.joinpath('analysis_results')

    @property
    def requirements_file(self) -> Path:
        """Python dependencies specification file for reproducible environments."""
        return self.root_dir.joinpath('requirements.txt')

    @property
    def raw_dir(self) -> Path:
        """Raw dataset storage directory for unprocessed network traffic data."""
        return self.data_dir.joinpath('raw')

    @property
    def processed_dir(self) -> Path:
        """Processed dataset directory for cleaned and prepared training data."""
        return self.data_dir.joinpath('processed')

    @property
    def zip_file(self) -> Path:
        """Temporary zip file location for dataset download operations."""
        return self.raw_dir.joinpath('nwdaf-data.zip')

    @property
    def raw_dataset(self) -> Path:
        """Primary raw dataset file containing network traffic with attack labels."""
        return self.raw_dir.joinpath('amari_ue_data_merged_with_attack_number.csv')


class DataConfig(BaseSettings):
    """
    Data processing and loading configuration for privacy-preserving analytics.

    This configuration class controls all aspects of data preparation including
    dataset splitting, batch processing, sequence length specification, and
    federated learning partitioning. The implementation supports both centralized
    and distributed training scenarios while maintaining privacy-preserving
    data handling throughout the processing pipeline.

    The configuration enables pathological data partitioning for realistic
    federated learning scenarios where data distribution across participants
    reflects real-world heterogeneity. Performance optimization parameters
    balance throughput with memory usage across different hardware configurations.

    Attributes:
        train_size (float): Training set proportion for dataset splitting
        val_size (float): Validation set proportion for dataset splitting
        partition_id (int): Client partition identifier for federated learning
        partition_by (str): Feature column for pathological partitioning
        num_partitions (int): Total partition count for federated scenarios
        num_classes_per_partition (int): Class limit per partition for non-IID data
        batch_size (int): Training batch size for model optimization
        seq_len (int): Temporal sequence length for time series modeling
        num_workers (int): DataLoader worker processes for parallel loading
        prefetch_factor (int): Batch prefetching count for performance optimization
        persistent_workers (bool): Worker process persistence across epochs
        pin_memory (bool): GPU memory pinning for accelerated transfer
    """
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
        """
        Validate dataset split proportions sum to valid range.
        """
        total = self.train_size + self.val_size
        if 1.0 - total < 0.:
            raise ValueError(f'Train and val sizes must be between 0 and 1. Total: {total}')
        return self

    model_config = {'env_prefix': 'PRIVATEER_DATA_',
                    'env_file': '.env',
                    'extra': 'ignore',
                    'case_sensitive': False}


class ModelConfig(BaseSettings):
    """
    Transformer architecture configuration for anomaly detection models.
    Attributes:
        model_name (str): Model identifier for experiment tracking and persistence
        input_size (int): Input feature dimension determined by dataset characteristics
        num_layers (int): Transformer encoder layer count for representational depth
        embed_dim (int): Model embedding dimension for attention mechanisms
        latent_dim (int): Compressed representation dimension for reconstruction
        num_heads (int): Multi-head attention head count for parallel processing
        dropout (float): Dropout probability for regularization during training
        seq_len (int): Input sequence length for temporal modeling
    """
    model_name: str = Field(default='TransformerAD')
    input_size: int = Field(default=8, ge=1)
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
    """
    Training process configuration for optimization and convergence control.
    Attributes:
        learning_rate (float): Optimizer learning rate for gradient descent
        epochs (int): Maximum training epochs before termination
        loss_fn_name (str): PyTorch loss function name for reconstruction training
        es_enabled (bool): Early stopping activation for convergence control
        es_patience (int): Early stopping patience epochs before termination
        es_warmup (int): Early stopping warmup epochs before monitoring
        target_metric (str): Primary evaluation metric for model selection
        direction (Literal): Optimization direction for target metric
    """
    # Optimization parameters
    learning_rate: float = Field(default=0.0001, gt=0.0)
    epochs: int = Field(default=100, gt=0)
    loss_fn_name: str = Field(default='L1Loss')

    # Early stopping
    es_enabled: bool = Field(default=True)
    es_patience: int = Field(default=20, gt=0)
    es_warmup: int = Field(default=10, gt=0)

    # Evaluation parameters
    target_metric: str = Field(default='val_unbalanced_f1-score')
    direction: Literal['minimize', 'maximize'] = Field(default='maximize')

    model_config = {'env_prefix': 'PRIVATEER_TRAIN_',
                    'env_file': '.env',
                    'extra': 'ignore',
                    'case_sensitive': False}


class AutotuningConfig(BaseSettings):
    """
    Hyperparameter optimization configuration using Optuna framework.

    This configuration class controls automated hyperparameter search through
    Optuna optimization framework, enabling systematic exploration of model
    and training parameter spaces. The implementation supports various sampling
    strategies, pruning mechanisms, and storage backends for scalable
    hyperparameter optimization across distributed computing environments.
    Attributes:
        study_name (str): Optuna study identifier for experiment organization
        n_trials (int): Total optimization trials for parameter exploration
        timeout (Optional[int]): Maximum optimization time in seconds
        target_metric (str): Optimization objective for parameter selection
        direction (Literal): Optimization direction for objective function
        storage_url (str): Database storage URL for trial persistence
        enable_pruning (bool): Trial pruning activation for efficiency
        pruning_warmup_steps (int): Warmup steps before pruning evaluation
        sampler_type (Literal): Sampling strategy for parameter space exploration
    """
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
    """
    Federated learning orchestration configuration with privacy guarantees.
    Attributes:
        server_address (str): Federated learning server binding address
        num_rounds (int): Total federated learning rounds for convergence
        n_clients (int): Active client count for each training round
        fraction_fit (float): Client participation fraction for training
        fraction_evaluate (float): Client participation fraction for evaluation
        secure_aggregation_enabled (bool): Secure aggregation protocol activation
        num_shares (int): Secret sharing parameter count for secure aggregation
        reconstruction_threshold (int): Minimum shares for reconstruction
        secagg_timeout (int): Secure aggregation timeout in seconds
        secagg_max_weight (int): Maximum model weight for secure aggregation
        epochs_per_round (int): Local training epochs per federated round
        partition_data (bool): Data partitioning activation for federated scenarios
    """

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
    """
    Privacy protection configuration for differential privacy and anonymization.
    Attributes:
        dp_enabled (bool): Differential privacy mechanism activation
        target_epsilon (float): Privacy budget parameter for DP guarantees
        target_delta (float): Failure probability parameter for DP guarantees
        max_grad_norm (float): Gradient clipping threshold for sensitivity control
        secure_mode (bool): Enhanced security mode for production deployments
        anonymization_enabled (bool): Data anonymization protocol activation
    """

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
    """
    Experiment tracking configuration for MLFlow integration.
    Attributes:
        enabled (bool): MLFlow tracking activation for experiment management
        tracking_uri (str): MLFlow server address for experiment storage
        experiment_name (str): Experiment identifier for organization
        parent_run_id (Optional[str]): Parent run identifier for nested experiments
        child_run_id (Optional[str]): Child run identifier for federated clients
    """
    enabled: bool = Field(default=True, description="Enable MLFlow tracking")
    tracking_uri: str = Field(default="http://localhost:5001", description="MLFlow server address")
    experiment_name: str = Field(default="privateer-ad", description="MLFlow experiment name")
    parent_run_id: str | None = Field(default=None, description="Parent run id")
    child_run_id: str | None = Field(default=None, description="Client run id")

    model_config = {'env_prefix': 'PRIVATEER_MLFLOW_',
                    'env_file': '.env',
                    'extra': 'ignore',
                    'case_sensitive': False}
