#!/usr/bin/env python3
"""
Train autoencoder on ingested False Positive data.

This script uses the standard HITL training workflow to train an autoencoder
on False Positive samples that have been ingested into the database.

Usage:
    python scripts/train_model.py --schema-id fp_77x8_timeseries --db hitl.db
"""

import argparse
import sys
from pathlib import Path
from typing import cast, Literal

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hitl.store.sqlite import SQLite
from hitl.store.repository import Repository
from hitl.artifacts.manager import Artifacts
from hitl.training.trainer import Trainer
from hitl.settings import Config
from hitl.utils.logging import get_logger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train autoencoder on False Positive data"
    )
    parser.add_argument(
        "--schema-id",
        type=str,
        required=True,
        help="Schema ID to train on (from ingestion)"
    )
    parser.add_argument(
        "--db",
        type=str,
        default="hitl.db",
        help="Path to SQLite database file"
    )
    parser.add_argument(
        "--artifacts-dir",
        type=str,
        default="artifacts",
        help="Directory for model artifacts"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="conv1d",
        choices=["dense", "conv1d"],
        help="Model architecture mode"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Training batch size"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate"
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.1,
        help="Validation split ratio"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early stopping patience"
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=99.5,
        help="Threshold percentile"
    )
    return parser.parse_args()


def train_model(
    schema_id: str,
    db_path: str,
    artifacts_dir: str,
    mode: str,
    epochs: int,
    batch_size: int,
    lr: float,
    val_split: float,
    patience: int,
    percentile: float
):
    """
    Train autoencoder using standard HITL workflow.
    
    Args:
        schema_id: Schema ID for training data
        db_path: Database path
        artifacts_dir: Artifacts directory
        mode: Model mode (dense/conv1d)
        epochs: Training epochs
        batch_size: Batch size
        lr: Learning rate
        val_split: Validation split
        patience: Early stopping patience
        percentile: Threshold percentile
    """
    logger = get_logger("train_model")
    
    # Initialize components
    logger.info("Initializing HITL components...")
    config = Config(
        sqlite_path=db_path,
        artifacts_dir=artifacts_dir,
        mode=cast(Literal["dense", "conv1d"], mode)
    )
    
    db = SQLite(config.sqlite_path)
    repo = Repository(db)
    artifacts = Artifacts(str(config.artifacts_dir))
    trainer = Trainer(repo, artifacts, config, logger)
    
    # Check schema exists
    schema_row = repo.get_schema(schema_id)
    if not schema_row:
        logger.error(f"Schema not found: {schema_id}")
        logger.info("Available schemas:")
        for s in repo.list_schemas():
            logger.info(f"  - {s['schema_id']} (shape={s['shape']}, dtype={s['dtype']})")
        sys.exit(1)
    
    # Type narrowing: schema_row is not None after the exit above
    assert schema_row is not None
    logger.info(f"Training on schema: {schema_id}")
    logger.info(f"  Shape: {schema_row['shape']}")
    logger.info(f"  Dtype: {schema_row['dtype']}")
    logger.info(f"  Mode: {mode}")
    
    # Training parameters
    params = {
        "mode": mode,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "val_split": val_split,
        "patience": patience,
        "percentile": percentile
    }
    
    logger.info("Training parameters:")
    for k, v in params.items():
        logger.info(f"  {k}: {v}")
    
    # Train using standard workflow
    logger.info("\n" + "="*60)
    logger.info("Starting training...")
    logger.info("="*60 + "\n")
    
    model_version = trainer.train_and_publish(schema_id, params)
    
    logger.info("\n" + "="*60)
    logger.info("✓ Training complete!")
    logger.info("="*60)
    logger.info(f"Model version: {model_version}")
    logger.info(f"Artifacts saved to: {artifacts_dir}/{model_version}/")
    
    # Show artifact files
    model_dir = Path(artifacts_dir) / model_version
    if model_dir.exists():
        logger.info("\nGenerated artifacts:")
        for f in sorted(model_dir.iterdir()):
            size_kb = f.stat().st_size / 1024
            logger.info(f"  {f.name} ({size_kb:.1f} KB)")
    
    # Next steps
    logger.info("\nNext steps:")
    logger.info(f"  1. Validate model: python scripts/validate_model.py --model-version {model_version}")
    logger.info(f"  2. Test inference: python -m hitl.cli predict --model-version {model_version}")
    logger.info(f"  3. Deploy: Copy {model_dir} to production")


def main():
    """Main entry point."""
    args = parse_args()
    
    try:
        train_model(
            schema_id=args.schema_id,
            db_path=args.db,
            artifacts_dir=args.artifacts_dir,
            mode=args.mode,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            val_split=args.val_split,
            patience=args.patience,
            percentile=args.percentile
        )
    except Exception as e:
        logger = get_logger("train_model")
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
