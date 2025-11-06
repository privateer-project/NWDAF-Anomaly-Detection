"""Tests for Training Pipeline"""

import pytest
import numpy as np
import torch
from pathlib import Path

from hitl.training.trainer import Trainer, mse_per_sample, get_device
from hitl.store.repository import Repository
from hitl.store.sqlite import SQLite
from hitl.artifacts.manager import Artifacts
from hitl.schemas.registry import SchemaRegistry
from hitl.settings import Config
from hitl.utils.logging import get_logger
from hitl.io.serialization import encode_npy


# Fixtures

@pytest.fixture
def config(tmp_path):
    """Test configuration."""
    return Config(
        sqlite_path=str(tmp_path / "test.db"),
        artifacts_dir=str(tmp_path / "artifacts"),
        mode="dense",
        log_level="INFO"
    )


@pytest.fixture
def repository(config):
    """Repository instance."""
    db = SQLite(config.sqlite_path)
    return Repository(db)


@pytest.fixture
def artifacts(config):
    """Artifacts manager instance."""
    return Artifacts(config.artifacts_dir)


@pytest.fixture
def logger():
    """Logger instance."""
    return get_logger("test_trainer")


@pytest.fixture
def trainer(repository, artifacts, config, logger):
    """Trainer instance."""
    return Trainer(repository, artifacts, config, logger)


@pytest.fixture
def registry(repository):
    """Schema registry instance."""
    return SchemaRegistry(repository)


@pytest.fixture
def sample_vector_1d():
    """Sample 1D vector."""
    return np.random.randn(84).astype("float32")


@pytest.fixture
def sample_vector_2d():
    """Sample 2D time-series."""
    return np.random.randn(240, 84).astype("float32")


# Tests - Helper Functions

def test_mse_per_sample_dense():
    """Test MSE computation for dense mode."""
    x = torch.randn(10, 50)  # (B, D)
    x_hat = torch.randn(10, 50)
    
    mse = mse_per_sample(x, x_hat)
    
    assert mse.shape == (10,)
    assert mse.dtype == torch.float32
    assert torch.all(mse >= 0)


def test_mse_per_sample_conv1d():
    """Test MSE computation for conv1d mode."""
    x = torch.randn(10, 84, 240)  # (B, F, T)
    x_hat = torch.randn(10, 84, 240)
    
    mse = mse_per_sample(x, x_hat)
    
    assert mse.shape == (10,)
    assert mse.dtype == torch.float32
    assert torch.all(mse >= 0)


def test_get_device():
    """Test device selection."""
    device = get_device()
    assert isinstance(device, torch.device)
    assert device.type in ["cpu", "cuda"]


# Tests - Dataset Loading

def test_load_dataset_returns_array_and_ids(trainer, repository, registry, sample_vector_1d):
    """Test loading vectors from database."""
    # Setup: Insert some vectors
    schema_info = registry.ensure(sample_vector_1d.shape, "float32")
    schema_id = schema_info["schema_id"]
    
    for i in range(5):
        anomaly_id = f"A{i}"
        repository.upsert_anomaly(
            anomaly_id, "2025-11-06T10:00:00Z", "test", schema_id,
            "2025-11-06T10:00:00Z", "2025-11-06T10:00:00Z"
        )
        blob = encode_npy(sample_vector_1d + i * 0.1)  # Slightly different
        repository.put_vector(anomaly_id, schema_id, blob, "2025-11-06T10:00:00Z")
    
    # Test
    X, ids = trainer.load_dataset(schema_id)
    
    assert X.shape == (5, 84)
    assert len(ids) == 5
    assert all(isinstance(id, str) for id in ids)


def test_load_dataset_raises_if_empty(trainer):
    """Test error when no vectors for schema."""
    with pytest.raises(ValueError, match="No vectors found"):
        trainer.load_dataset("nonexistent-schema")


# Tests - Data Preprocessing

def test_split_data(trainer):
    """Test train/val splitting."""
    X = np.random.randn(100, 84).astype("float32")
    
    X_train, X_val = trainer._split_data(X, val_split=0.1, shuffle=False)
    
    assert X_train.shape == (90, 84)
    assert X_val.shape == (10, 84)


def test_compute_scaler_dense(trainer):
    """Test scaler computation for dense mode."""
    X_train = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype="float32")
    
    scaler = trainer._compute_scaler(X_train, "dense")
    
    assert scaler["kind"] == "per_feature"
    assert len(scaler["mean"]) == 3
    assert len(scaler["std"]) == 3
    assert scaler["eps"] == 1e-8
    
    # Check values
    expected_mean = X_train.mean(axis=0)
    assert np.allclose(scaler["mean"], expected_mean)


def test_compute_scaler_conv1d(trainer):
    """Test scaler computation for conv1d mode."""
    X_train = np.random.randn(10, 84, 240).astype("float32")
    
    scaler = trainer._compute_scaler(X_train, "conv1d")
    
    assert scaler["kind"] == "per_feature"
    assert len(scaler["mean"]) == 84  # One per feature/channel
    assert len(scaler["std"]) == 84
    
    # Check computation
    expected_mean = X_train.mean(axis=(0, 2))
    assert np.allclose(scaler["mean"], expected_mean)


def test_normalize_dense(trainer):
    """Test normalization for dense mode."""
    X = np.array([[1, 2, 3], [4, 5, 6]], dtype="float32")
    scaler = {
        "mean": [2.5, 3.5, 4.5],
        "std": [1.5, 1.5, 1.5],
        "eps": 1e-8
    }
    
    X_norm = trainer._normalize(X, scaler, "dense")
    
    assert X_norm.shape == X.shape
    # Check approximate standardization
    expected = (X - np.array(scaler["mean"])) / np.array(scaler["std"])
    assert np.allclose(X_norm, expected)


# Tests - Training

def test_train_epoch(trainer):
    """Test training epoch runs."""
    # Create tiny model and data
    model = torch.nn.Sequential(
        torch.nn.Linear(10, 10)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    X_train = np.random.randn(20, 10).astype("float32")
    device = torch.device("cpu")
    
    loss = trainer._train_epoch(model, optimizer, X_train, batch_size=5, device=device)
    
    assert isinstance(loss, float)
    assert loss >= 0


def test_validate(trainer):
    """Test validation computes loss."""
    model = torch.nn.Sequential(
        torch.nn.Linear(10, 10)
    )
    X_val = np.random.randn(10, 10).astype("float32")
    device = torch.device("cpu")
    
    loss = trainer._validate(model, X_val, batch_size=5, device=device)
    
    assert isinstance(loss, float)
    assert loss >= 0


def test_compute_threshold(trainer):
    """Test threshold computation from train errors."""
    model = torch.nn.Sequential(
        torch.nn.Linear(10, 10)
    )
    X_train = np.random.randn(100, 10).astype("float32")
    device = torch.device("cpu")
    percentile = 95.0
    
    threshold = trainer._compute_threshold(model, X_train, percentile, device)
    
    assert "value" in threshold
    assert "percentile" in threshold
    assert "strategy" in threshold
    assert "computed_at" in threshold
    assert threshold["percentile"] == percentile
    assert threshold["strategy"] == "percentile"
    assert threshold["value"] >= 0


# Tests - Integration

def test_fit_trains_model(trainer):
    """Test model training completes."""
    # Small dataset
    X = np.random.randn(50, 20).astype("float32")
    params = {
        "mode": "dense",
        "epochs": 5,
        "batch_size": 10,
        "lr": 0.01,
        "val_split": 0.2,
        "patience": 10,
        "percentile": 95.0
    }
    
    state_dict, scaler, threshold, metrics = trainer.fit(X, params)
    
    # Check returns
    assert isinstance(state_dict, dict)
    assert isinstance(scaler, dict)
    assert isinstance(threshold, dict)
    assert isinstance(metrics, dict)
    
    # Check metrics
    assert "train_losses" in metrics
    assert "val_losses" in metrics
    assert len(metrics["train_losses"]) <= 5


def test_fit_returns_artifacts(trainer):
    """Test fit returns all required artifacts."""
    X = np.random.randn(30, 15).astype("float32")
    params = {
        "mode": "dense",
        "epochs": 3,
        "batch_size": 10,
        "lr": 0.01,
        "val_split": 0.2,
        "patience": 5,
        "percentile": 99.0
    }
    
    state_dict, scaler, threshold, metrics = trainer.fit(X, params)
    
    # Check state_dict
    assert len(state_dict) > 0
    assert all(isinstance(v, torch.Tensor) for v in state_dict.values())
    
    # Check scaler
    assert "mean" in scaler
    assert "std" in scaler
    assert len(scaler["mean"]) == 15
    
    # Check threshold
    assert "value" in threshold
    assert threshold["value"] > 0
    
    # Check metrics
    assert "epochs_trained" in metrics
    assert metrics["epochs_trained"] <= 3


def test_train_and_publish_creates_version(trainer, repository, registry, artifacts):
    """Test complete train and publish workflow."""
    # Setup: Create dataset in DB
    X_samples = [np.random.randn(50).astype("float32") for _ in range(20)]
    schema_info = registry.ensure(X_samples[0].shape, "float32")
    schema_id = schema_info["schema_id"]
    
    for i, X in enumerate(X_samples):
        anomaly_id = f"A{i}"
        repository.upsert_anomaly(
            anomaly_id, "2025-11-06T10:00:00Z", "test", schema_id,
            "2025-11-06T10:00:00Z", "2025-11-06T10:00:00Z"
        )
        blob = encode_npy(X)
        repository.put_vector(anomaly_id, schema_id, blob, "2025-11-06T10:00:00Z")
    
    # Train and publish
    params = {
        "mode": "dense",
        "epochs": 5,
        "batch_size": 5,
        "lr": 0.01,
        "val_split": 0.2,
        "patience": 10,
        "percentile": 95.0
    }
    
    model_version = trainer.train_and_publish(schema_id, params)
    
    # Check model version format
    assert model_version.startswith("AE-")
    
    # Check model registered in DB
    model_row = repository.get_model(model_version)
    assert model_row is not None
    assert model_row["kind"] == "dense"
    
    # Check artifacts exist (need to prepend artifacts dir since path is relative)
    artifact_path = artifacts.base_dir / model_row["artifact_path"]
    assert (artifact_path / "model.pt").exists()
    assert (artifact_path / "config.json").exists()
    assert (artifact_path / "scaler.json").exists()
    assert (artifact_path / "threshold.json").exists()


def test_train_epoch_decreases_loss(trainer):
    """Test training reduces loss (sanity check)."""
    # Simple overfit test
    X = np.ones((10, 5), dtype="float32") * 2.0  # Constant data
    
    # Build model
    model = torch.nn.Sequential(
        torch.nn.Linear(5, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 5)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    device = torch.device("cpu")
    
    # Train for a few epochs
    losses = []
    for _ in range(10):
        loss = trainer._train_epoch(model, optimizer, X, batch_size=5, device=device)
        losses.append(loss)
    
    # Loss should generally decrease
    assert losses[-1] < losses[0]
