#!/usr/bin/env python
"""
Development script for end-to-end workflow testing.

Simulates a complete anomaly detection lifecycle:
1. Anomaly detection and ingestion
2. Vector storage with schema registration
3. User feedback collection
4. Model training and versioning
5. Model deployment (live model setting)
6. Inference with new anomalies
"""

import sys
import tempfile
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn as nn

from hitl.store.sqlite import SQLite
from hitl.store.repository import Repository
from hitl.schemas.registry import SchemaRegistry
from hitl.io.serialization import encode_npy, decode_npy
from hitl.artifacts.manager import Artifacts
from hitl.utils.time import now_iso, utcnow
from hitl.utils.ids import uuid_str, feedback_id as make_feedback_id


def print_section(title):
    """Print a section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


class SimpleAutoencoder(nn.Module):
    """Simple autoencoder model."""

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


def simulate_network_data(num_samples: int = 10) -> np.ndarray:
    """Simulate network telemetry data."""
    # Generate realistic-looking network metrics
    base = np.random.randn(num_samples, 128).astype(np.float32)

    # Add some structure
    for i in range(num_samples):
        base[i] = base[i] * (i % 3 + 1)  # Varying scales

    return base


def calculate_reconstruction_error(model: nn.Module, data: torch.Tensor) -> float:
    """Calculate reconstruction error."""
    with torch.no_grad():
        reconstructed = model(data)
        error = torch.mean((data - reconstructed) ** 2).item()
    return error


def test_full_workflow():
    """Test complete anomaly detection workflow."""

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "hitl.db"
        artifacts_dir = Path(tmpdir) / "artifacts"
        artifacts_dir.mkdir()

        # Initialize components
        print_section("1. System Initialization")

        print("Initializing HITL system...")
        sqlite = SQLite(str(db_path))
        repo = Repository(sqlite)
        registry = SchemaRegistry(repo)
        artifacts = Artifacts(base_dir=str(artifacts_dir))

        print("✓ Database initialized")
        print("✓ Schema registry ready")
        print("✓ Artifacts manager ready")
        print(f"  - DB: {db_path}")
        print(f"  - Artifacts: {artifacts_dir}")

        # Phase 1: Anomaly Detection
        print_section("2. Anomaly Detection Phase")

        print("Simulating network monitoring...")
        network_data = simulate_network_data(num_samples=20)
        print(f"✓ Generated {len(network_data)} network samples")
        print(f"  - Shape per sample: {network_data[0].shape}")
        print(f"  - Dtype: {network_data.dtype}")

        # Register schema
        shape = network_data[0].shape
        schema = registry.ensure(shape=shape, dtype="float32")
        print(f"✓ Schema registered: {schema['schema_id'][:16]}...")

        # Ingest anomalies
        print("\nIngesting detected anomalies...")
        anomaly_ids = []
        for i in range(10):  # First 10 samples as anomalies
            aid = f"NET-{i:03d}-{uuid_str()[:8]}"
            anomaly_ids.append(aid)

            now = now_iso()

            # Store anomaly metadata
            repo.upsert_anomaly(
                anomaly_id=aid,
                occurred_at=now,
                source="network-monitor",
                schema_id=schema["schema_id"],
                created_at=now,
                updated_at=now,
            )

            # Store vector
            blob = encode_npy(network_data[i])
            repo.put_vector(
                anomaly_id=aid,
                schema_id=schema["schema_id"],
                blob=blob,
                created_at=now,
            )

        print(f"✓ Ingested {len(anomaly_ids)} anomalies")
        print(f"  - Sample IDs: {anomaly_ids[:3]}...")

        # Phase 2: Human Feedback
        print_section("3. Feedback Collection Phase")

        print("Simulating expert review...")

        # Simulate different labels
        labels = ["true"] * 7 + ["false"] * 3  # 70% true positives

        for aid, label in zip(anomaly_ids, labels):
            now = utcnow()
            fid = make_feedback_id(aid, "expert_001", now)

            repo.insert_feedback(
                feedback_id=fid,
                anomaly_id=aid,
                user_id="expert_001",
                label=label,
                confidence=0.9 if label == "true" else 0.8,
                note=f"Manual review: {label} positive",
                created_at=now_iso(),
            )

        print(f"✓ Collected feedback for {len(anomaly_ids)} anomalies")

        # Check feedback stats
        feedback_list = repo.list_feedback()
        true_count = sum(1 for f in feedback_list if f["label"] == "true")
        false_count = sum(1 for f in feedback_list if f["label"] == "false")

        print(f"  - True positives: {true_count}")
        print(f"  - False positives: {false_count}")
        print(f"  - Accuracy: {true_count/len(feedback_list)*100:.1f}%")

        # Phase 3: Model Training
        print_section("4. Model Training Phase")

        print("Preparing training data...")

        # Collect labeled data
        true_anomalies = []
        for aid, label in zip(anomaly_ids, labels):
            if label == "true":
                vector_result = repo.get_vector(aid)
                if vector_result:
                    _, blob = vector_result
                    arr = decode_npy(blob)
                    true_anomalies.append(arr)

        training_data = np.stack(true_anomalies)
        print(f"✓ Training set: {training_data.shape}")

        # Create model
        print("\nInitializing model...")
        model = SimpleAutoencoder(input_dim=128, latent_dim=16)
        print(f"✓ Model created: {type(model).__name__}")
        print("  - Input: 128")
        print("  - Latent: 16")
        print(f"  - Parameters: {sum(p.numel() for p in model.parameters())}")

        # Train model (simplified - just a few steps)
        print("\nTraining model...")
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        X = torch.from_numpy(training_data)
        losses = []

        for epoch in range(50):
            optimizer.zero_grad()
            X_hat = model(X)
            loss = torch.mean((X - X_hat) ** 2)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1:2d}: loss = {loss.item():.6f}")

        print("✓ Training complete")
        print(f"  - Initial loss: {losses[0]:.6f}")
        print(f"  - Final loss: {losses[-1]:.6f}")
        print(f"  - Improvement: {(1 - losses[-1]/losses[0])*100:.1f}%")

        # Calculate threshold (95th percentile of training errors)
        print("\nCalculating anomaly threshold...")
        with torch.no_grad():
            reconstructed = model(X)
            errors = torch.mean((X - reconstructed) ** 2, dim=1)
            threshold = float(np.percentile(errors.numpy(), 95))

        print(f"✓ Threshold: {threshold:.6f}")

        # Phase 4: Model Versioning
        print_section("5. Model Versioning Phase")

        print("Creating new model version...")
        version, _ = artifacts.create_version("dense", (128,))
        print(f"✓ Version: {version}")

        # Save all artifacts
        print("\nSaving artifacts...")

        artifacts.save_model(
            model_version=version,
            state_dict=model.state_dict(),
        )
        print("✓ Model saved")

        config = {
            "model_type": "autoencoder",
            "input_dim": 128,
            "latent_dim": 16,
            "training_samples": len(training_data),
            "epochs": 50,
            "learning_rate": 0.001,
        }
        artifacts.save_config(version, config)
        print("✓ Config saved")

        threshold_dict = {
            "value": threshold,
            "percentile": 95.0,
            "method": "mse"
        }
        artifacts.save_threshold(version, threshold_dict)
        print("✓ Threshold saved")

        # Phase 5: Model Deployment
        print_section("6. Model Deployment Phase")

        print("Deploying model to production...")
        repo.set_live_model(version)
        print(f"✓ Live model set: {version}")

        # Verify deployment
        live = repo.get_live_model()
        assert live == version
        print("✓ Deployment verified")

        # Enable inference
        repo.set_setting("inference_enabled", "true")
        print("✓ Inference enabled")

        # Phase 6: Inference
        print_section("7. Inference Phase")

        print("Loading production model...")
        loaded_artifacts = artifacts.load_all(version)
        prod_model = loaded_artifacts["model"]
        prod_threshold = loaded_artifacts["threshold"]

        print(f"✓ Model loaded: {version}")
        print(f"✓ Threshold: {prod_threshold:.6f}")

        # Test on new data
        print("\nProcessing new network samples...")
        new_data = simulate_network_data(num_samples=10)

        results = []
        for i, sample in enumerate(new_data):
            # Convert to tensor
            x = torch.from_numpy(sample).unsqueeze(0)

            # Calculate reconstruction error
            error = calculate_reconstruction_error(prod_model, x)

            # Classify
            is_anomaly = error > prod_threshold

            results.append(
                {
                    "sample_id": i,
                    "error": error,
                    "threshold": prod_threshold,
                    "is_anomaly": is_anomaly,
                }
            )

        print(f"✓ Processed {len(results)} samples")

        # Show results
        print("\nInference results:")
        print(f"{'Sample':<8} {'Error':<12} {'Threshold':<12} {'Anomaly':<10}")
        print("-" * 45)

        anomaly_count = 0
        for r in results[:5]:  # Show first 5
            status = "YES" if r["is_anomaly"] else "no"
            if r["is_anomaly"]:
                anomaly_count += 1
            print(
                f"{r['sample_id']:<8} {r['error']:<12.6f} {r['threshold']:<12.6f} {status:<10}"
            )

        total_anomalies = sum(1 for r in results if r["is_anomaly"])
        print(f"\n✓ Detected {total_anomalies}/{len(results)} anomalies")

        # Phase 7: System Status
        print_section("8. System Status")

        # Count everything
        all_anomalies = repo.list_anomalies()
        all_feedback = repo.list_feedback()
        all_schemas = repo.list_schemas()
        all_models = repo.list_models()
        all_versions = artifacts.list_versions()

        print("Current system state:")
        print(f"  - Anomalies: {len(all_anomalies)}")
        print(f"  - Feedback entries: {len(all_feedback)}")
        print(f"  - Schemas: {len(all_schemas)}")
        print(f"  - Models: {len(all_models)}")
        print(f"  - Artifact versions: {len(all_versions)}")
        print(f"  - Live model: {repo.get_live_model()}")
        print(f"  - Inference enabled: {repo.get_setting('inference_enabled')}")

        # Summary statistics
        print_section("9. Summary Statistics")

        print("Workflow completion:")
        print(f"✓ Phase 1: Detection - {len(anomaly_ids)} anomalies ingested")
        print(f"✓ Phase 2: Feedback - {len(feedback_list)} reviews collected")
        print(f"✓ Phase 3: Training - Model trained on {len(training_data)} samples")
        print(f"✓ Phase 4: Versioning - Version {version} created")
        print("✓ Phase 5: Deployment - Model deployed to production")
        print(f"✓ Phase 6: Inference - {len(results)} new samples processed")
        print("✓ Phase 7: Status - System operational")

        print("\nModel performance:")
        print(f"  - Training loss reduction: {(1 - losses[-1]/losses[0])*100:.1f}%")
        print(f"  - Threshold: {prod_threshold:.6f}")
        print(f"  - Detection rate: {total_anomalies/len(results)*100:.1f}%")

        print("\nData pipeline:")
        print(f"  - Schema: {schema['shape']}")
        print(f"  - Training samples: {len(training_data)}")
        print(f"  - True positives: {true_count}")
        print(f"  - False positives: {false_count}")
        print(f"  - Labeling accuracy: {true_count/len(feedback_list)*100:.1f}%")

        print_section("Workflow Complete")
        print("✓ End-to-end workflow executed successfully!")
        print("✓ All components integrated and working:")
        print("  1. ✓ Anomaly ingestion")
        print("  2. ✓ Vector storage")
        print("  3. ✓ Schema management")
        print("  4. ✓ Feedback collection")
        print("  5. ✓ Model training")
        print("  6. ✓ Artifact versioning")
        print("  7. ✓ Model deployment")
        print("  8. ✓ Production inference")


if __name__ == "__main__":
    try:
        test_full_workflow()
        print("\n" + "=" * 60)
        print("  ✓ COMPLETE WORKFLOW TESTED SUCCESSFULLY")
        print("=" * 60 + "\n")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
