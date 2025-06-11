"""
PRIVATEER Configuration Module

This module provides a centralized, type-safe configuration system for the entire project.

Usage Examples:
    # Get specific configurations
    paths = PathConfig()
    model_cfg = ModelConfig()

    # Access specific values
    batch_size = DataConfig().batch_size
    data_dir = PathConfig().data_dir

    # Override with environment variables
    config = ModelConfig() # Automatically loads PRIVATEER_MODEL_* env vars

The configuration system supports environment variable overrides with the PRIVATEER_ prefix,
enabling flexible deployment across different environments while maintaining type safety
and validation through Pydantic models.
"""
from .metadata import MetadataConfig
from .settings import (
    PathConfig,
    ModelConfig,
    TrainingConfig,
    AutotuningConfig,
    DataConfig,
    FederatedLearningConfig,
    PrivacyConfig,
    MLFlowConfig
)

__all__ = [
    'MetadataConfig',
    'PathConfig',
    'ModelConfig',
    'TrainingConfig',
    'DataConfig',
    'FederatedLearningConfig',
    'PrivacyConfig',
    'MLFlowConfig',
    'AutotuningConfig',
]
