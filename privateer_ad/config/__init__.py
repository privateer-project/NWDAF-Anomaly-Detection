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
    # Configuration classes
    PathConfig,
    ModelConfig,
    TrainingConfig,
    DataConfig,
    FederatedLearningConfig,
    PrivacyConfig,
    MLFlowConfig,
    MetadataConfig,

    # Data classes
    DeviceInfo,
    AttackInfo,
    FeatureInfo,

    # Convenience functions for specific configs
    get_paths,
    get_model_config,
    get_training_config,
    get_data_config,
    get_fl_config,
    get_privacy_config,
    get_mlflow_config,
    get_metadata,
)

__all__ = [
    # Configuration classes
    'PathConfig',
    'ModelConfig',
    'TrainingConfig',
    'DataConfig',
    'FederatedLearningConfig',
    'PrivacyConfig',
    'MLFlowConfig',
    'MetadataConfig',

    # Data classes
    'DeviceInfo',
    'AttackInfo',
    'FeatureInfo',

    # Convenience getters
    'get_paths',
    'get_model_config',
    'get_training_config',
    'get_data_config',
    'get_fl_config',
    'get_privacy_config',
    'get_mlflow_config',
    'get_metadata',
]