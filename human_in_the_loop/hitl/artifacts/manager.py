"""
Artifact Manager for HITL System

Handles storage and retrieval of model artifacts on the local filesystem,
including weights, configuration, and thresholds.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from hitl.errors import ArtifactMissing
from hitl.utils.ids import model_version as generate_model_version

__all__ = ["Artifacts"]


def _find_next_sequence(base_dir: Path, date_str: str) -> int:
    """
    Find next sequence number for a date.

    Args:
        base_dir: Artifacts directory
        date_str: Date in YYYY.MM.DD format

    Returns:
        Next available sequence number (starting at 1)

    Example:
        >>> _find_next_sequence(Path("artifacts"), "2025.11.04")
        1  # or higher if directories exist
    """
    # Pattern: AE-{date_str}-*
    pattern = f"AE-{date_str}-*"

    # Find existing directories
    existing = list(base_dir.glob(pattern))

    if not existing:
        return 1

    # Extract sequence numbers
    sequences = []
    for path in existing:
        try:
            # Parse "AE-2025.11.04-3" → 3
            parts = path.name.split("-")
            if len(parts) >= 3:
                seq = int(parts[-1])
                sequences.append(seq)
        except (ValueError, IndexError):
            continue

    if not sequences:
        return 1

    return max(sequences) + 1


class Artifacts:
    """Manager for model artifact storage and retrieval."""

    def __init__(self, base_dir: str = "artifacts") -> None:
        """
        Initialize artifacts manager.

        Args:
            base_dir: Root directory for all artifacts

        Creates base_dir if it doesn't exist.
        Typical structure:
            artifacts/
                AE-2025.11.04-1/
                    model.pt
                    config.json
                    threshold.json
                AE-2025.11.04-2/
                    ...
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def create_version(
        self,
        mode: str,
        input_shape: tuple[int, ...],
        date: datetime | None = None,
    ) -> tuple[str, str]:
        """
        Generate new model version and artifact directory.

        Args:
            mode: "dense" or "conv1d"
            input_shape: Model input dimensions
            date: Optional datetime for version (defaults to now)

        Returns:
            (model_version, artifact_path) tuple
            - model_version: e.g., "AE-2025.11.04-1"
            - artifact_path: relative path from base_dir

        Example:
            >>> version, path = artifacts.create_version("dense", (128,))
            >>> version
            'AE-2025.11.04-1'
        """
        # Generate version string
        if date is None:
            date = datetime.now()

        # Get date string
        date_str = date.strftime("%Y.%m.%d")

        # Find next sequence number
        sequence = _find_next_sequence(self.base_dir, date_str)

        # Generate version using utility
        version = generate_model_version(date, sequence)

        # Create directory
        version_dir = self.base_dir / version
        version_dir.mkdir(parents=True, exist_ok=True)

        return version, version

    def save_model(self, model_version: str, state_dict: dict[str, Any]) -> None:
        """
        Save PyTorch model state dict.

        Args:
            model_version: Model version identifier
            state_dict: PyTorch model.state_dict()

        Saves to: {base_dir}/{model_version}/model.pt

        Raises:
            ArtifactMissing: if version directory doesn't exist
        """
        version_dir = self.base_dir / model_version
        if not version_dir.exists():
            raise ArtifactMissing(f"Version directory not found: {version_dir}")

        model_path = version_dir / "model.pt"
        torch.save(state_dict, model_path)

    def save_config(self, model_version: str, config: dict[str, Any]) -> None:
        """
        Save model configuration as JSON.

        Args:
            model_version: Model version identifier
            config: Dict with model hyperparameters

        Saves to: {base_dir}/{model_version}/config.json

        Config should include:
            - mode: "dense" or "conv1d"
            - input_shape: tuple
            - architecture params
            - training params
        """
        version_dir = self.base_dir / model_version
        if not version_dir.exists():
            raise ArtifactMissing(f"Version directory not found: {version_dir}")

        config_path = version_dir / "config.json"
        with config_path.open("w") as f:
            json.dump(config, f, indent=2)

    def save_threshold(self, model_version: str, threshold: dict[str, Any]) -> None:
        """
        Save anomaly detection threshold as JSON.

        Args:
            model_version: Model version identifier
            threshold: Dict with threshold value and metadata

        Saves to: {base_dir}/{model_version}/threshold.json

        Threshold dict format:
            {
                "value": 0.123,
                "percentile": 99.5,
                "train_errors": [...],  # optional
                "computed_at": "2025-11-04T10:00:00Z"
            }
        """
        version_dir = self.base_dir / model_version
        if not version_dir.exists():
            raise ArtifactMissing(f"Version directory not found: {version_dir}")

        threshold_path = version_dir / "threshold.json"
        with threshold_path.open("w") as f:
            json.dump(threshold, f, indent=2)

    def load_all(self, model_version: str) -> dict[str, Any]:
        """
        Load all artifacts for a model version.

        Args:
            model_version: Model version identifier

        Returns:
            Dict with all artifacts:
            {
                "model": state_dict,
                "config": config_dict,
                "threshold": threshold_dict
            }

        Raises:
            ArtifactMissing: if directory or any file missing
            ValueError: if JSON files malformed
        """
        return {
            "model": self.load_model(model_version),
            "config": self.load_config(model_version),
            "threshold": self.load_threshold(model_version),
        }

    def load_model(self, model_version: str) -> dict[str, Any]:
        """
        Load just the model state dict.

        Args:
            model_version: Model version identifier

        Returns:
            PyTorch state dict

        Raises:
            ArtifactMissing: if file doesn't exist
        """
        model_path = self.base_dir / model_version / "model.pt"
        if not model_path.exists():
            raise ArtifactMissing(f"Model file not found: {model_path}")

        return torch.load(model_path, weights_only=True)

    def load_config(self, model_version: str) -> dict[str, Any]:
        """
        Load just the config.

        Args:
            model_version: Model version identifier

        Returns:
            Config dict

        Raises:
            ArtifactMissing: if file doesn't exist
        """
        config_path = self.base_dir / model_version / "config.json"
        if not config_path.exists():
            raise ArtifactMissing(f"Config file not found: {config_path}")

        with config_path.open() as f:
            return json.load(f)

    def load_threshold(self, model_version: str) -> dict[str, Any]:
        """
        Load just the threshold.

        Args:
            model_version: Model version identifier

        Returns:
            Threshold dict

        Raises:
            ArtifactMissing: if file doesn't exist
        """
        threshold_path = self.base_dir / model_version / "threshold.json"
        if not threshold_path.exists():
            raise ArtifactMissing(f"Threshold file not found: {threshold_path}")

        with threshold_path.open() as f:
            return json.load(f)

    def exists(self, model_version: str) -> bool:
        """
        Check if artifacts exist for model version.

        Args:
            model_version: Model version identifier

        Returns:
            True if directory exists with all required files

        Required files:
            - model.pt
            - config.json
            - threshold.json
        """
        version_dir = self.base_dir / model_version
        if not version_dir.exists():
            return False

        required_files = ["model.pt", "config.json", "threshold.json"]
        return all((version_dir / fname).exists() for fname in required_files)

    def list_versions(self) -> list[str]:
        """
        List all artifact versions in base_dir.

        Returns:
            Sorted list of version strings (newest first)

        Scans base_dir for directories matching pattern "AE-*".
        """
        # Find all directories starting with "AE-"
        versions = []
        for path in self.base_dir.glob("AE-*"):
            if path.is_dir():
                versions.append(path.name)

        # Sort by version name (which includes date) in reverse order
        return sorted(versions, reverse=True)

    def delete(self, model_version: str) -> None:
        """
        Delete artifacts for a model version.

        Args:
            model_version: Model version identifier

        Removes entire directory: {base_dir}/{model_version}/

        Warning: This is permanent!
        """
        version_dir = self.base_dir / model_version
        if version_dir.exists():
            shutil.rmtree(version_dir)

    def get_path(self, model_version: str) -> Path:
        """
        Get absolute path to model version directory.

        Args:
            model_version: Model version identifier

        Returns:
            Path object for artifact directory
        """
        return (self.base_dir / model_version).absolute()
