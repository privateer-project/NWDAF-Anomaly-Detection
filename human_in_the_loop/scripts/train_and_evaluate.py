#!/usr/bin/env python3
"""
End-to-End Training and Evaluation Script

This script performs a complete workflow:
1. Load data from ../data/ directory (validation or test set)
2. Insert anomalies into the database
3. Train an autoencoder model
4. Evaluate the model on the data
5. Generate a report with metrics

Usage:
    python scripts/train_and_evaluate.py --mode dense --use-validation
    python scripts/train_and_evaluate.py --mode conv1d --use-test
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hitl.artifacts.manager import Artifacts
from hitl.settings import Config
from hitl.schemas.registry import SchemaRegistry
from hitl.store.repository import Repository
from hitl.store.sqlite import SQLite
from hitl.training.trainer import Trainer
from hitl.utils.logging import configure_logging, get_logger

configure_logging("INFO")
logger = get_logger("train_eval")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train and evaluate autoencoder on anomaly data"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["dense", "conv1d"],
        default="conv1d",
        help="Model mode: 'dense' for 1D vectors, 'conv1d' for 2D time-series",
    )
    parser.add_argument(
        "--use-validation",
        action="store_true",
        help="Use validation data (default is test data)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="../data",
        help="Directory containing .npz files",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="data/hitl.db",
        help="SQLite database path",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=str,
        default="artifacts",
        help="Model artifacts directory",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Training batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate",
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=95.0,
        help="Threshold percentile (e.g., 95.0)",
    )
    return parser.parse_args()


def load_data(data_dir: str, use_validation: bool = True):
    """
    Load anomaly data from .npz file.

    Args:
        data_dir: Directory containing data files
        use_validation: Use validation set (else test set)

    Returns:
        Tuple of (X, metadata_df)
        X: numpy array of shape (N, T, F) for conv1d or (N, F) for dense
        metadata_df: DataFrame with timestamps and labels
    """
    dataset = "validation" if use_validation else "test"
    npz_path = Path(data_dir) / f"{dataset}_anomalies.npz"
    csv_path = Path(data_dir) / f"{dataset}_anomalies_meta.csv"

    logger.info(f"Loading {dataset} data from {npz_path}")

    if not npz_path.exists():
        raise FileNotFoundError(f"Data file not found: {npz_path}")

    # Load arrays
    data = np.load(npz_path)
    X = data["X"]
    logger.info(f"Loaded X with shape: {X.shape}, dtype: {X.dtype}")

    # Load metadata
    if csv_path.exists():
        metadata = pd.read_csv(csv_path)
        logger.info(f"Loaded metadata: {len(metadata)} rows")
    else:
        # Create minimal metadata
        metadata = pd.DataFrame(
            {
                "timestamp_ns": data.get("ts_ns", np.arange(len(X))),
                "is_anomaly": data.get("is_anom", np.ones(len(X), dtype=bool)),
                "true_label": data.get("y", np.zeros(len(X))),
            }
        )
        logger.warning("No metadata CSV, created minimal metadata")

    return X, metadata


def insert_into_db(repository, registry, X, metadata, source_name: str):
    """
    Insert anomalies into the database.

    Args:
        repository: Repository instance
        registry: SchemaRegistry instance
        X: Data array
        metadata: Metadata DataFrame
        source_name: Source identifier (e.g., 'validation' or 'test')

    Returns:
        schema_id of inserted data
    """
    logger.info(f"Inserting {len(X)} anomalies into database...")

    # Register schema
    if len(X.shape) == 3:
        # Conv1d: (N, T, F)
        sample_shape = X[0].shape  # (T, F)
    else:
        # Dense: (N, F)
        sample_shape = X[0].shape  # (F,)

    schema_info = registry.ensure(sample_shape, str(X.dtype))
    schema_id = schema_info["schema_id"]
    logger.info(f"Schema ID: {schema_id}, shape: {sample_shape}")

    # For large datasets, use smaller sample
    if len(X) > 5000:
        logger.warning(f"Dataset has {len(X)} samples - using first 5000 for training demo")
        X = X[:5000]
        metadata = metadata.iloc[:5000]

    # Insert anomalies (batch mode would be faster, but keeping simple for now)
    from hitl.io.serialization import encode_npy

    for i in range(len(X)):
        anomaly_id = f"{source_name}-{i:06d}"

        # Get timestamp
        if "timestamp" in metadata.columns:
            occurred_at = str(metadata.iloc[i]["timestamp"])
        else:
            occurred_at = f"2025-11-06T00:00:00.{i:09d}Z"

        # Insert anomaly record
        repository.upsert_anomaly(
            anomaly_id=anomaly_id,
            occurred_at=occurred_at,
            source=source_name,
            schema_id=schema_id,
            created_at=occurred_at,
            updated_at=occurred_at,
        )

        # Insert vector
        blob = encode_npy(X[i])
        repository.put_vector(anomaly_id, schema_id, blob, occurred_at)

        if (i + 1) % 1000 == 0:
            logger.info(f"  Inserted {i + 1}/{len(X)} anomalies")

    logger.info(f"✓ Inserted all {len(X)} anomalies")
    return schema_id


def train_model(trainer, schema_id, params):
    """
    Train model and publish artifacts.

    Args:
        trainer: Trainer instance
        schema_id: Schema ID to train on
        params: Training parameters dict

    Returns:
        model_version string
    """
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING MODEL")
    logger.info("=" * 60)
    logger.info(f"Parameters: {params}")

    model_version = trainer.train_and_publish(schema_id, params)

    logger.info(f"\n✓ Model trained and published: {model_version}")
    return model_version


def evaluate_model(artifacts, repository, model_version, schema_id):
    """
    Evaluate trained model on the data it was trained on.

    Args:
        artifacts: Artifacts manager
        repository: Repository instance
        model_version: Model version to evaluate
        schema_id: Schema ID of the data

    Returns:
        Dict with evaluation metrics
    """
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATING MODEL")
    logger.info("=" * 60)

    # Load model artifacts
    logger.info(f"Loading model: {model_version}")
    model_artifacts = artifacts.load_all(model_version)
    config = model_artifacts["config"]
    threshold = model_artifacts["threshold"]

    logger.info(f"Config: mode={config['mode']}, input_shape={config['input_shape']}")
    logger.info(
        f"Threshold: {threshold['value']:.6f} (p{threshold['percentile']})"
    )

    # Build model
    from hitl.models.ae import build_model

    mode = config["mode"]
    input_shape = tuple(config["input_shape"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(mode, input_shape).to(device)
    model.load_state_dict(model_artifacts["model"])
    model.eval()

    logger.info(f"Model loaded on device: {device}")

    # Load data from DB
    from hitl.io.serialization import decode_npy

    vectors_raw = repository.get_vectors_by_schema(schema_id)
    logger.info(f"Loaded {len(vectors_raw)} vectors from database")

    X_list = []
    for v in vectors_raw:
        arr = decode_npy(v["vector_blob"])
        X_list.append(arr)

    X = np.stack(X_list, axis=0)
    logger.info(f"Stacked data shape: {X.shape}")

    # Transpose if Conv1d
    if mode == "conv1d" and len(X.shape) == 3:
        X = np.transpose(X, (0, 2, 1))  # (N, T, F) -> (N, F, T)
        logger.info(f"Transposed for Conv1d: {X.shape}")

    # Compute reconstruction errors
    logger.info("Computing reconstruction errors...")
    X_tensor = torch.from_numpy(X).float().to(device)

    with torch.no_grad():
        X_recon = model(X_tensor)
        mse = ((X_tensor - X_recon) ** 2).mean(dim=tuple(range(1, X_tensor.ndim)))
        errors = mse.cpu().numpy()

    # Analyze results
    threshold_value = threshold["value"]
    predictions = errors > threshold_value

    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 60)
    logger.info("\nReconstruction Errors:")
    logger.info(f"  Mean:   {errors.mean():.6f}")
    logger.info(f"  Std:    {errors.std():.6f}")
    logger.info(f"  Min:    {errors.min():.6f}")
    logger.info(f"  Max:    {errors.max():.6f}")
    logger.info(f"  Median: {np.median(errors):.6f}")

    logger.info(f"\nThreshold: {threshold_value:.6f}")
    logger.info(
        f"Predictions: {predictions.sum()}/{len(predictions)} classified as anomalous ({100 * predictions.mean():.2f}%)"
    )

    # Compute percentiles
    percentiles = [50, 75, 90, 95, 99, 99.5]
    logger.info("\nError Percentiles:")
    for p in percentiles:
        val = np.percentile(errors, p)
        logger.info(f"  p{p:>5.1f}: {val:.6f}")

    return {
        "errors": errors,
        "threshold": threshold_value,
        "predictions": predictions,
        "mean_error": float(errors.mean()),
        "std_error": float(errors.std()),
        "min_error": float(errors.min()),
        "max_error": float(errors.max()),
        "median_error": float(np.median(errors)),
        "n_predicted_anomalous": int(predictions.sum()),
        "n_total": len(predictions),
        "anomaly_rate": float(predictions.mean()),
    }


def main():
    """Main execution function."""
    args = parse_args()

    logger.info("=" * 60)
    logger.info("TRAIN AND EVALUATE WORKFLOW")
    logger.info("=" * 60)
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Dataset: {'validation' if args.use_validation else 'test'}")
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Database: {args.db_path}")
    logger.info(f"Artifacts: {args.artifacts_dir}")

    # Initialize components
    logger.info("\nInitializing components...")
    config = Config(
        sqlite_path=args.db_path,
        artifacts_dir=args.artifacts_dir,
        mode=args.mode,
        log_level="INFO",
    )

    db = SQLite(config.sqlite_path)
    repository = Repository(db)
    registry = SchemaRegistry(repository)
    artifacts = Artifacts(str(config.artifacts_dir))
    trainer = Trainer(repository, artifacts, config, logger)

    logger.info("✓ Components initialized")

    # Load data
    logger.info("\n" + "=" * 60)
    logger.info("LOADING DATA")
    logger.info("=" * 60)
    X, metadata = load_data(args.data_dir, args.use_validation)

    # For dense mode, flatten to 1D
    if args.mode == "dense" and len(X.shape) == 3:
        logger.info("Flattening 3D data to 1D for dense mode...")
        N, T, F = X.shape
        X = X.reshape(N, T * F)
        logger.info(f"New shape: {X.shape}")

    # Insert into database
    logger.info("\n" + "=" * 60)
    logger.info("INSERTING DATA INTO DATABASE")
    logger.info("=" * 60)
    source_name = "validation" if args.use_validation else "test"
    schema_id = insert_into_db(repository, registry, X, metadata, source_name)

    # Train model
    train_params = {
        "mode": args.mode,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "val_split": 0.1,
        "patience": 10,
        "percentile": args.percentile,
    }

    model_version = train_model(trainer, schema_id, train_params)

    # Evaluate model
    results = evaluate_model(artifacts, repository, model_version, schema_id)

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("WORKFLOW COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Model Version: {model_version}")
    logger.info(f"Data Samples: {results['n_total']}")
    logger.info(f"Mean Error: {results['mean_error']:.6f}")
    logger.info(f"Threshold: {results['threshold']:.6f}")
    logger.info(
        f"Anomaly Rate: {results['anomaly_rate']:.2%} ({results['n_predicted_anomalous']}/{results['n_total']})"
    )
    logger.info("\nArtifacts saved to: " + str(Path(args.artifacts_dir) / model_version))
    logger.info("Database: " + args.db_path)


if __name__ == "__main__":
    main()
