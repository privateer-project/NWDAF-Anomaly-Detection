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

# Import all configuration classes and functions
from .settings import (
    # Configuration classes
    PrivateerConfig,
    PathConfig,
    ModelConfig,
    TrainingConfig,
    DataConfig,
    FederatedLearningConfig,
    PrivacyConfig,
    MLFlowConfig,
    MetadataConfig,
    AutotuningConfig,

    # Data classes
    DeviceInfo,
    AttackInfo,
    FeatureInfo,

    # Main configuration instance and getters
    get_config,
    reload_config,

    # Convenience functions for specific configs
    get_paths,
    get_model_config,
    get_training_config,
    get_data_config,
    get_fl_config,
    get_privacy_config,
    get_mlflow_config,
    get_metadata,

    # Utilities
    validate_config,
    print_config_summary,
    save_config_to_file,
    load_config_from_file,
)

__all__ = [
    # Configuration classes
    'PrivateerConfig',
    'PathConfig',
    'ModelConfig',
    'TrainingConfig',
    'DataConfig',
    'FederatedLearningConfig',
    'PrivacyConfig',
    'MLFlowConfig',
    'MetadataConfig',
    'AutotuningConfig',

    # Data classes
    'DeviceInfo',
    'AttackInfo',
    'FeatureInfo',

    # Main functions
    'get_config',
    'reload_config',

    # Convenience getters
    'get_paths',
    'get_model_config',
    'get_training_config',
    'get_data_config',
    'get_fl_config',
    'get_privacy_config',
    'get_mlflow_config',
    'get_metadata',

    # Utilities
    'validate_config',
    'print_config_summary',
    'save_config_to_file',
    'load_config_from_file',
]