"""
PRIVATEER Configuration Module

This module provides a centralized, type-safe configuration system for the entire project.
All configurations are organized by domain and easily accessible through convenient functions.

Usage Examples:
    # Get specific configurations
    paths = get_paths()
    model_cfg = get_model_config()

    # Access specific values
    batch_size = get_training_config().batch_size
    data_dir = get_paths().data_dir

    # Get the full configuration
    config = get_config()

    # Print configuration summary
    print_config_summary()

    # Validate configuration
    validate_config()
"""

from .settings import (
    PathConfig,
    ModelConfig,
    TrainingConfig,
    AutotuningConfig,
    DataConfig,
    FederatedLearningConfig,
    PrivacyConfig,
    MLFlowConfig,
    MetadataConfig,
    get_paths,
    get_model_config,
    get_training_config,
    get_data_config,
    get_fl_config,
    get_privacy_config,
    get_mlflow_config,
    get_autotuning_config,
    get_metadata
)

__all__ = [
    'PathConfig',
    'ModelConfig',
    'TrainingConfig',
    'DataConfig',
    'FederatedLearningConfig',
    'PrivacyConfig',
    'MLFlowConfig',
    'MetadataConfig',
    'AutotuningConfig',
    'get_paths',
    'get_model_config',
    'get_training_config',
    'get_autotuning_config',
    'get_data_config',
    'get_fl_config',
    'get_privacy_config',
    'get_mlflow_config',
    'get_metadata',
]
