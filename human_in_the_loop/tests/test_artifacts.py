"""Tests for Artifacts Manager."""

from __future__ import annotations

import tempfile
from datetime import datetime
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from hitl.artifacts.manager import Artifacts, _find_next_sequence
from hitl.errors import ArtifactMissing


class SimpleModel(nn.Module):
    """Simple model for testing."""
    
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)
    
    def forward(self, x):
        return self.linear(x)


@pytest.fixture
def temp_dir():
    """Create temporary directory for artifacts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def artifacts(temp_dir):
    """Create artifacts manager with temp directory."""
    return Artifacts(str(temp_dir))


@pytest.fixture
def sample_state_dict():
    """Create sample PyTorch state dict."""
    model = SimpleModel()
    return model.state_dict()


@pytest.fixture
def sample_config():
    """Create sample config dict."""
    return {
        "mode": "dense",
        "input_shape": [128],
        "latent_dim": 32,
        "epochs": 100,
        "batch_size": 32,
        "lr": 0.001,
    }


@pytest.fixture
def sample_scaler():
    """Create sample scaler dict."""
    return {
        "mean": [0.1, 0.2, 0.3],
        "std": [1.0, 1.1, 1.2],
    }


@pytest.fixture
def sample_threshold():
    """Create sample threshold dict."""
    return {
        "value": 0.123,
        "percentile": 99.5,
        "computed_at": "2025-11-04T10:00:00Z",
    }


class TestVersionCreation:
    """Test version creation and sequencing."""

    def test_create_version_generates_version(self, artifacts):
        """Test model version generation."""
        version, path = artifacts.create_version("dense", (128,))
        
        # Version should start with "AE-"
        assert version.startswith("AE-")
        # Should contain date
        assert len(version.split("-")) == 3
        # Path should match version
        assert path == version

    def test_create_version_with_date(self, artifacts):
        """Test version creation with specific date."""
        date = datetime(2025, 11, 4, 10, 0, 0)
        version, _ = artifacts.create_version("dense", (128,), date=date)
        
        # Should contain the specified date
        assert "2025.11.04" in version

    def test_create_version_increments_sequence(self, artifacts):
        """Test sequence number increments for same date."""
        date = datetime(2025, 11, 4)
        
        # Create first version
        version1, _ = artifacts.create_version("dense", (128,), date=date)
        
        # Create second version on same date
        version2, _ = artifacts.create_version("dense", (128,), date=date)
        
        # Sequence should increment
        assert version1.endswith("-1")
        assert version2.endswith("-2")

    def test_create_version_creates_directory(self, artifacts, temp_dir):
        """Test artifact directory is created."""
        version, _ = artifacts.create_version("dense", (128,))
        
        # Directory should exist
        version_dir = temp_dir / version
        assert version_dir.exists()
        assert version_dir.is_dir()


class TestArtifactSaving:
    """Test saving artifacts."""

    def test_save_model_writes_file(self, artifacts, sample_state_dict, temp_dir):
        """Test model state dict is saved."""
        version, _ = artifacts.create_version("dense", (128,))
        
        artifacts.save_model(version, sample_state_dict)
        
        # File should exist
        model_path = temp_dir / version / "model.pt"
        assert model_path.exists()

    def test_save_config_writes_json(self, artifacts, sample_config, temp_dir):
        """Test config is saved as JSON."""
        version, _ = artifacts.create_version("dense", (128,))
        
        artifacts.save_config(version, sample_config)
        
        # File should exist
        config_path = temp_dir / version / "config.json"
        assert config_path.exists()
        
        # Should be valid JSON
        import json
        with config_path.open() as f:
            loaded = json.load(f)
        assert loaded == sample_config

    def test_save_scaler_writes_json(self, artifacts, sample_scaler, temp_dir):
        """Test scaler params are saved as JSON."""
        version, _ = artifacts.create_version("dense", (128,))
        
        artifacts.save_scaler(version, sample_scaler)
        
        # File should exist
        scaler_path = temp_dir / version / "scaler.json"
        assert scaler_path.exists()

    def test_save_threshold_writes_json(self, artifacts, sample_threshold, temp_dir):
        """Test threshold is saved as JSON."""
        version, _ = artifacts.create_version("dense", (128,))
        
        artifacts.save_threshold(version, sample_threshold)
        
        # File should exist
        threshold_path = temp_dir / version / "threshold.json"
        assert threshold_path.exists()

    def test_save_raises_if_version_missing(self, artifacts, sample_config):
        """Test saving raises error if version doesn't exist."""
        with pytest.raises(ArtifactMissing):
            artifacts.save_config("AE-2025.11.04-99", sample_config)


class TestArtifactLoading:
    """Test loading artifacts."""

    @pytest.fixture
    def complete_artifacts(self, artifacts, sample_state_dict, sample_config, 
                           sample_scaler, sample_threshold):
        """Create complete set of artifacts."""
        version, _ = artifacts.create_version("dense", (128,))
        artifacts.save_model(version, sample_state_dict)
        artifacts.save_config(version, sample_config)
        artifacts.save_scaler(version, sample_scaler)
        artifacts.save_threshold(version, sample_threshold)
        return version

    def test_load_all_returns_dict(self, artifacts, complete_artifacts):
        """Test loading all artifacts returns complete dict."""
        all_artifacts = artifacts.load_all(complete_artifacts)
        
        assert "model" in all_artifacts
        assert "config" in all_artifacts
        assert "scaler" in all_artifacts
        assert "threshold" in all_artifacts

    def test_load_model_returns_state_dict(self, artifacts, complete_artifacts):
        """Test loading just model state dict."""
        state_dict = artifacts.load_model(complete_artifacts)
        
        assert isinstance(state_dict, dict)
        # Should have some parameters
        assert len(state_dict) > 0

    def test_load_config_returns_dict(self, artifacts, complete_artifacts, sample_config):
        """Test loading just config."""
        config = artifacts.load_config(complete_artifacts)
        
        assert config == sample_config

    def test_load_scaler_returns_dict(self, artifacts, complete_artifacts, sample_scaler):
        """Test loading just scaler."""
        scaler = artifacts.load_scaler(complete_artifacts)
        
        assert scaler == sample_scaler

    def test_load_threshold_returns_dict(self, artifacts, complete_artifacts, sample_threshold):
        """Test loading just threshold."""
        threshold = artifacts.load_threshold(complete_artifacts)
        
        assert threshold == sample_threshold

    def test_load_all_raises_if_missing(self, artifacts):
        """Test ArtifactMissing raised for invalid version."""
        with pytest.raises(ArtifactMissing):
            artifacts.load_all("AE-2025.11.04-99")

    def test_load_model_raises_if_missing(self, artifacts):
        """Test loading model raises if file doesn't exist."""
        version, _ = artifacts.create_version("dense", (128,))
        # Don't save model
        
        with pytest.raises(ArtifactMissing):
            artifacts.load_model(version)


class TestExistenceChecking:
    """Test artifact existence checking."""

    def test_exists_returns_true_if_complete(self, artifacts, sample_state_dict,
                                              sample_config, sample_scaler, 
                                              sample_threshold):
        """Test existence check returns True for complete artifacts."""
        version, _ = artifacts.create_version("dense", (128,))
        artifacts.save_model(version, sample_state_dict)
        artifacts.save_config(version, sample_config)
        artifacts.save_scaler(version, sample_scaler)
        artifacts.save_threshold(version, sample_threshold)
        
        assert artifacts.exists(version) is True

    def test_exists_returns_false_if_missing(self, artifacts):
        """Test existence check returns False for missing version."""
        assert artifacts.exists("AE-2025.11.04-99") is False

    def test_exists_returns_false_if_incomplete(self, artifacts, sample_config):
        """Test existence check returns False if files missing."""
        version, _ = artifacts.create_version("dense", (128,))
        # Only save config, not all files
        artifacts.save_config(version, sample_config)
        
        assert artifacts.exists(version) is False


class TestVersionListing:
    """Test version listing."""

    def test_list_versions_returns_sorted(self, artifacts):
        """Test version listing is sorted."""
        # Create multiple versions
        date1 = datetime(2025, 11, 3)
        date2 = datetime(2025, 11, 4)
        
        v1, _ = artifacts.create_version("dense", (128,), date=date1)
        v2, _ = artifacts.create_version("dense", (128,), date=date2)
        v3, _ = artifacts.create_version("dense", (128,), date=date2)
        
        versions = artifacts.list_versions()
        
        # Should have all versions
        assert len(versions) == 3
        assert v1 in versions
        assert v2 in versions
        assert v3 in versions
        
        # Should be sorted (newest first)
        assert versions[0] > versions[-1]

    def test_list_versions_empty_when_none(self, artifacts):
        """Test listing returns empty list when no versions."""
        versions = artifacts.list_versions()
        assert versions == []


class TestArtifactDeletion:
    """Test artifact deletion."""

    def test_delete_removes_directory(self, artifacts, temp_dir):
        """Test artifact deletion."""
        version, _ = artifacts.create_version("dense", (128,))
        
        # Verify directory exists
        version_dir = temp_dir / version
        assert version_dir.exists()
        
        # Delete
        artifacts.delete(version)
        
        # Should no longer exist
        assert not version_dir.exists()

    def test_delete_nonexistent_doesnt_raise(self, artifacts):
        """Test deleting nonexistent version doesn't raise."""
        # Should not raise
        artifacts.delete("AE-2025.11.04-99")


class TestPathResolution:
    """Test path resolution."""

    def test_get_path_returns_absolute(self, artifacts, temp_dir):
        """Test path resolution."""
        version, _ = artifacts.create_version("dense", (128,))
        
        path = artifacts.get_path(version)
        
        # Should be absolute
        assert path.is_absolute()
        # Should include version
        assert str(path).endswith(version)
        # Should be under temp_dir
        assert str(path).startswith(str(temp_dir))


class TestHelperFunctions:
    """Test helper functions."""

    def test_find_next_sequence_starts_at_1(self, temp_dir):
        """Test sequence starts at 1 when none exist."""
        seq = _find_next_sequence(temp_dir, "2025.11.04")
        assert seq == 1

    def test_find_next_sequence_increments(self, temp_dir):
        """Test sequence increments based on existing."""
        # Create some directories
        (temp_dir / "AE-2025.11.04-1").mkdir()
        (temp_dir / "AE-2025.11.04-2").mkdir()
        
        seq = _find_next_sequence(temp_dir, "2025.11.04")
        assert seq == 3

    def test_find_next_sequence_different_dates(self, temp_dir):
        """Test different dates have independent sequences."""
        (temp_dir / "AE-2025.11.03-1").mkdir()
        (temp_dir / "AE-2025.11.03-2").mkdir()
        
        # New date should start at 1
        seq = _find_next_sequence(temp_dir, "2025.11.04")
        assert seq == 1

