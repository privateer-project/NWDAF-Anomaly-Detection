"""
Configuration management for HITL system.

This module provides a Config dataclass and utilities for loading configuration from
environment variables and validating settings.

Classes:
    - Config: Configuration dataclass with all system settings

Functions:
    - get_env_config() -> Config: Load configuration from environment variables
    - paths(config: Config) -> Config: Resolve and create necessary directories
    - validate_config(config: Config) -> None: Validate configuration values
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass
class Config:
    """
    Configuration dataclass for HITL system.

    Attributes:
        sqlite_path: Path to SQLite database file (accepts str or Path)
        artifacts_dir: Directory for storing model artifacts (accepts str or Path)
        mode: Model architecture mode ('dense' or 'conv1d')
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        dev_mode: If True, use human-readable logging; if False, use JSON
    """

    sqlite_path: Path | str = field(default_factory=lambda: Path("./hitl.db"))
    artifacts_dir: Path | str = field(default_factory=lambda: Path("./artifacts"))
    mode: Literal["dense", "conv1d"] = "dense"
    log_level: str = "INFO"
    dev_mode: bool = False

    def __post_init__(self) -> None:
        """Convert string paths to Path objects."""
        if isinstance(self.sqlite_path, str):
            object.__setattr__(self, 'sqlite_path', Path(self.sqlite_path))
        if isinstance(self.artifacts_dir, str):
            object.__setattr__(self, 'artifacts_dir', Path(self.artifacts_dir))
        # Ensure types are correct for type checker
        assert isinstance(self.sqlite_path, Path)
        assert isinstance(self.artifacts_dir, Path)


def get_env_config() -> Config:
    """
    Load configuration from environment variables.

    Environment variables:
        HITL_SQLITE_PATH: Path to database file (default: ./hitl.db)
        HITL_ARTIFACTS_DIR: Path to artifacts directory (default: ./artifacts)
        HITL_MODE: Model mode (default: dense)
        HITL_LOG_LEVEL: Log level (default: INFO)
        HITL_DEV_MODE: Dev mode flag (default: False)

    Returns:
        Config: Configuration object with values from environment

    Example:
        >>> config = get_env_config()
        >>> config.log_level
        'INFO'
    """
    return Config(
        sqlite_path=Path(os.getenv("HITL_SQLITE_PATH", "./hitl.db")),
        artifacts_dir=Path(os.getenv("HITL_ARTIFACTS_DIR", "./artifacts")),
        mode=os.getenv("HITL_MODE", "dense"),  # type: ignore
        log_level=os.getenv("HITL_LOG_LEVEL", "INFO").upper(),
        dev_mode=os.getenv("HITL_DEV_MODE", "").lower() in ("1", "true", "yes"),
    )


def paths(config: Config) -> Config:
    """
    Resolve paths and create necessary directories.

    Args:
        config: Configuration object

    Returns:
        Config: Configuration with resolved absolute paths

    Raises:
        OSError: If directory creation fails

    Example:
        >>> config = Config()
        >>> config = paths(config)
        >>> config.artifacts_dir.exists()
        True
    """
    # Resolve to absolute paths
    assert isinstance(config.sqlite_path, Path)
    assert isinstance(config.artifacts_dir, Path)
    config.sqlite_path = config.sqlite_path.resolve()
    config.artifacts_dir = config.artifacts_dir.resolve()

    # Create directories
    config.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    config.artifacts_dir.mkdir(parents=True, exist_ok=True)

    return config


def validate_config(config: Config) -> None:
    """
    Validate configuration values.

    Args:
        config: Configuration object to validate

    Raises:
        ValueError: If any configuration value is invalid

    Example:
        >>> config = Config(mode="dense")
        >>> validate_config(config)  # No exception
        >>> config.mode = "invalid"
        >>> validate_config(config)  # Raises ValueError
        Traceback (most recent call last):
        ...
        ValueError: Invalid mode: invalid (must be 'dense' or 'conv1d')
    """
    # Validate mode
    valid_modes = ("dense", "conv1d")
    if config.mode not in valid_modes:
        raise ValueError(f"Invalid mode: {config.mode} (must be 'dense' or 'conv1d')")

    # Validate log level
    valid_levels = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
    if config.log_level.upper() not in valid_levels:
        raise ValueError(
            f"Invalid log_level: {config.log_level} "
            f"(must be one of {', '.join(valid_levels)})"
        )

    # Validate paths are not empty
    if not config.sqlite_path:
        raise ValueError("sqlite_path cannot be empty")

    if not config.artifacts_dir:
        raise ValueError("artifacts_dir cannot be empty")
