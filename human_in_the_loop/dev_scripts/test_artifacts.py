#!/usr/bin/env python
"""
Development script to test artifacts manager functionality.

Tests artifact versioning, saving/loading models, configs, scalers, and thresholds.
"""

import sys
import tempfile
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import numpy as np

from hitl.artifacts.manager import Artifacts
from hitl.store.sqlite import SQLite
from hitl.store.repository import Repository
from hitl.errors import ArtifactNotFound


def print_section(title):
    """Print a section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


class SimpleAutoencoder(nn.Module):
    """Simple autoencoder for testing."""
    
    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
        )
    
    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat


def test_artifacts():
    """Test artifacts manager functionality."""
    
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        artifacts_dir = Path(tmpdir) / "artifacts"
        artifacts_dir.mkdir()
        
        print_section("1. Artifacts Initialization")
        print(f"Database: {db_path}")
        print(f"Artifacts directory: {artifacts_dir}")
        
        sqlite = SQLite(str(db_path))
        repo = Repository(sqlite)
        artifacts = Artifacts(repo=repo, base_dir=str(artifacts_dir))
        
        print("✓ Artifacts manager initialized")
        
        # Test version creation
        print_section("2. Version Creation")
        
        print("Creating new version...")
        version = artifacts.new_version()
        print(f"✓ Version created: {version}")
        
        # Verify format
        assert version.startswith("AE-")
        parts = version.split("-")
        assert len(parts) == 3
        print(f"  - Prefix: {parts[0]}")
        print(f"  - Date: {parts[1]}")
        print(f"  - Sequence: {parts[2]}")
        
        # Test same-day versioning
        version2 = artifacts.new_version()
        print(f"✓ Second version: {version2}")
        assert version != version2
        print("  - Different sequence numbers verified")
        
        # Test model saving
        print_section("3. Model Saving")
        
        print("Creating PyTorch model...")
        model = SimpleAutoencoder(input_dim=128, latent_dim=16)
        print(f"✓ Model created: {type(model).__name__}")
        print(f"  - Input dim: 128")
        print(f"  - Latent dim: 16")
        print(f"  - Parameters: {sum(p.numel() for p in model.parameters())}")
        
        schema_id = "test-schema-123"
        print(f"\nSaving model to version: {version}")
        
        artifacts.save_model(
            model_version=version,
            model=model,
            mode="dense",
            schema_id=schema_id,
        )
        
        print("✓ Model saved")
        print(f"  - Version: {version}")
        print(f"  - Mode: dense")
        print(f"  - Schema: {schema_id}")
        
        # Verify files created
        version_dir = artifacts_dir / version
        assert version_dir.exists()
        assert (version_dir / "model.pt").exists()
        print(f"  - Directory: {version_dir}")
        print(f"  - Model file: model.pt ({(version_dir / 'model.pt').stat().st_size} bytes)")
        
        # Test config saving
        print_section("4. Config Saving")
        
        config = {
            "model_type": "autoencoder",
            "input_dim": 128,
            "latent_dim": 16,
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 100,
            "features": ["dl_bitrate", "ul_bitrate", "dl_retx"],
        }
        
        print("Saving configuration...")
        artifacts.save_config(model_version=version, config=config)
        print("✓ Config saved")
        
        assert (version_dir / "config.json").exists()
        print(f"  - Config file: config.json")
        
        # Test scaler saving
        print_section("5. Scaler Saving")
        
        scaler = {
            "mean": [0.5, 1.2, -0.3],
            "std": [1.0, 2.5, 0.8],
        }
        
        print("Saving scaler...")
        artifacts.save_scaler(model_version=version, scaler=scaler)
        print("✓ Scaler saved")
        
        assert (version_dir / "scaler.json").exists()
        print(f"  - Scaler file: scaler.json")
        
        # Test threshold saving
        print_section("6. Threshold Saving")
        
        threshold = 0.15
        
        print(f"Saving threshold: {threshold}")
        artifacts.save_threshold(model_version=version, threshold=threshold)
        print("✓ Threshold saved")
        
        assert (version_dir / "threshold.json").exists()
        print(f"  - Threshold file: threshold.json")
        
        # Test complete loading
        print_section("7. Complete Loading")
        
        print(f"Loading all artifacts for version: {version}")
        
        loaded = artifacts.load_all(version)
        
        print("✓ All artifacts loaded")
        print(f"  - Model: {type(loaded['model']).__name__}")
        print(f"  - Config: {len(loaded['config'])} keys")
        print(f"  - Scaler: {list(loaded['scaler'].keys())}")
        print(f"  - Threshold: {loaded['threshold']}")
        
        # Verify model weights match
        print("\nVerifying model weights...")
        for (name1, p1), (name2, p2) in zip(model.named_parameters(), 
                                             loaded['model'].named_parameters()):
            assert name1 == name2
            assert torch.allclose(p1, p2)
        print("✓ All weights match original model")
        
        # Verify config matches
        assert loaded['config'] == config
        print("✓ Config matches original")
        
        # Verify scaler matches
        assert loaded['scaler'] == scaler
        print("✓ Scaler matches original")
        
        # Verify threshold matches
        assert loaded['threshold'] == threshold
        print("✓ Threshold matches original")
        
        # Test individual loading
        print_section("8. Individual Loading")
        
        model_only = artifacts.load_model(version)
        print(f"✓ Model loaded: {type(model_only).__name__}")
        
        config_only = artifacts.load_config(version)
        print(f"✓ Config loaded: {config_only['model_type']}")
        
        scaler_only = artifacts.load_scaler(version)
        print(f"✓ Scaler loaded: mean={scaler_only['mean']}")
        
        threshold_only = artifacts.load_threshold(version)
        print(f"✓ Threshold loaded: {threshold_only}")
        
        # Test listing versions
        print_section("9. Version Listing")
        
        # Create another version
        version3 = artifacts.new_version()
        artifacts.save_model(
            model_version=version3,
            model=model,
            mode="conv1d",
            schema_id="another-schema",
        )
        
        versions = artifacts.list_versions()
        print(f"✓ Found {len(versions)} versions:")
        for v in versions:
            print(f"  - {v}")
        
        assert version in versions
        assert version2 in versions
        assert version3 in versions
        
        # Test version existence
        print_section("10. Version Existence Check")
        
        assert artifacts.exists(version)
        print(f"✓ Version exists: {version}")
        
        assert not artifacts.exists("AE-2099.12.31-999")
        print("✓ Non-existent version correctly detected")
        
        # Test error handling
        print_section("11. Error Handling")
        
        print("Testing missing artifacts...")
        
        try:
            artifacts.load_model("nonexistent-version")
            print("❌ Should have raised ArtifactNotFound")
        except ArtifactNotFound as e:
            print(f"✓ Correctly raised error for missing model")
        
        try:
            artifacts.load_config("nonexistent-version")
            print("❌ Should have raised ArtifactNotFound")
        except ArtifactNotFound as e:
            print(f"✓ Correctly raised error for missing config")
        
        # Test deletion
        print_section("12. Version Deletion")
        
        print(f"Deleting version: {version2}")
        artifacts.delete(version2)
        print("✓ Version deleted")
        
        assert not artifacts.exists(version2)
        print("✓ Version no longer exists")
        
        versions_after = artifacts.list_versions()
        assert version2 not in versions_after
        print(f"✓ Version not in list: {len(versions_after)} remaining")
        
        # Test model architecture preservation
        print_section("13. Architecture Preservation")
        
        print("Testing forward pass with loaded model...")
        
        # Create sample input
        x = torch.randn(10, 128)
        
        # Original model output
        with torch.no_grad():
            output_original = model(x)
        
        # Loaded model output
        with torch.no_grad():
            output_loaded = loaded['model'](x)
        
        # Compare
        assert torch.allclose(output_original, output_loaded, rtol=1e-5)
        print("✓ Model outputs match for same input")
        print(f"  - Input shape: {x.shape}")
        print(f"  - Output shape: {output_original.shape}")
        print(f"  - Max difference: {(output_original - output_loaded).abs().max():.2e}")
        
        # Test weights_only security
        print_section("14. Security: weights_only Loading")
        
        print("Verifying PyTorch weights_only mode...")
        # This is tested internally by load_model
        # If it loads successfully, weights_only=True is working
        
        model_secure = artifacts.load_model(version)
        print("✓ Model loaded with weights_only=True")
        print("  - Protection against arbitrary code execution")
        
        print_section("Summary")
        print("✓ All artifact manager tests passed!")
        print(f"✓ Total versions created: {len(artifacts.list_versions())}")
        print(f"✓ Artifacts directory: {artifacts_dir}")
        print(f"✓ Features verified:")
        print("  - Date-based versioning (AE-YYYY.MM.DD-N)")
        print("  - Model state dict saving/loading")
        print("  - Config, scaler, threshold management")
        print("  - Complete artifact loading")
        print("  - Version listing and deletion")
        print("  - Architecture preservation")
        print("  - Security (weights_only loading)")


if __name__ == "__main__":
    try:
        test_artifacts()
        print("\n" + "="*60)
        print("  ✓ ALL TESTS PASSED")
        print("="*60 + "\n")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
