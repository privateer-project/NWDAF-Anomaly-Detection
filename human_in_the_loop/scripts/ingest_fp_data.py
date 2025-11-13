#!/usr/bin/env python3
"""
Bulk ingestion script for False Positive data from NPZ files.

This script loads False Positive samples from test_anomalies.npz and
ingests them into the HITL database following the existing schema.
Once ingested, the standard training workflow can proceed unchanged.

Usage:
    python scripts/ingest_fp_data.py --data-dir ../data --db hitl.db
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from hitl.store.sqlite import SQLite
from hitl.store.repository import Repository
from hitl.io.serialization import encode_npy
from hitl.utils.logging import get_logger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Ingest False Positive data into HITL database"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="../data",
        help="Directory containing test_anomalies.npz and test_anomalies_meta.csv"
    )
    parser.add_argument(
        "--db",
        type=str,
        default="db/hitl.db",
        help="Path to SQLite database file"
    )
    parser.add_argument(
        "--schema-id",
        type=str,
        default="fp_77x8_timeseries",
        help="Schema ID to use for ingested data"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="upstream_detector",
        help="Source identifier for these anomalies"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of samples to ingest (for testing)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Number of rows between progress logs/commits"
    )
    parser.add_argument(
        "--apply-ddl",
        action="store_true",
        help="Apply DDL from hitl/ddl.sql if database is new"
    )
    return parser.parse_args()


def compute_schema_id(shape: tuple, dtype: str) -> str:
    """
    Generate schema ID from shape and dtype.
    
    Args:
        shape: Tensor shape (e.g., (77, 8))
        dtype: NumPy dtype
    
    Returns:
        Schema ID string
    """
    shape_str = "x".join(map(str, shape))
    return f"fp_{shape_str}_{dtype}"


def ingest_false_positives(
    data_dir: str,
    db_path: str,
    schema_id: str,
    source: str,
    limit: int | None = None,
    batch_size: int = 1000,
    apply_ddl: bool = False,
):
    """
    Load False Positives from NPZ and ingest into database.
    
    Args:
        data_dir: Directory containing data files
        db_path: Path to SQLite database
        schema_id: Schema ID for this data
        source: Source identifier
        limit: Optional limit on number of samples
    """
    logger = get_logger("ingest_fp")
    
    # Load data
    data_path = Path(data_dir) / "test_anomalies.npz"
    meta_path = Path(data_dir) / "test_anomalies_meta.csv"
    
    logger.info(f"Loading data from {data_path}")
    data = np.load(data_path)
    meta = pd.read_csv(meta_path)
    
    # Extract False Positives
    # FP = detector said anomaly (is_anomaly=True) but was wrong (true_label=0)
    fp_mask = (meta['is_anomaly']) & (meta['true_label'] == 0)
    
    X_fp = data['X'][fp_mask]  # Shape: (N, 77, 8)
    ts_fp = data['ts_ns'][fp_mask]  # Timestamps in nanoseconds
    true_labels = meta['true_label'][fp_mask].values
    
    if limit:
        X_fp = X_fp[:limit]
        ts_fp = ts_fp[:limit]
        true_labels = true_labels[:limit]
    
    logger.info(f"Found {len(X_fp)} False Positive samples")
    logger.info(f"Sample shape: {X_fp[0].shape}")
    logger.info(f"Data type: {X_fp.dtype}")
    
    # Initialize database
    db = SQLite(db_path)
    repo = Repository(db)

    # Optionally ensure DDL is applied (SQLite already applies DDL on new DB,
    # but this flag allows forcing initialization if requested)
    if apply_ddl:
        try:
            # call protected method on SQLite; it's idempotent
            db._apply_ddl()
            logger.info("Applied DDL to database (apply_ddl=True)")
        except Exception:
            logger.debug("apply_ddl requested but DDL application failed or already applied", exc_info=True)
    
    # Register schema
    sample_shape = X_fp[0].shape  # (77, 8)
    shape_str = ",".join(map(str, sample_shape))
    ndim = len(sample_shape)
    numel = int(np.prod(sample_shape))
    dtype_str = str(X_fp.dtype)
    now = datetime.now(timezone.utc).isoformat()
    
    logger.info(f"Registering schema: {schema_id}")
    logger.info(f"  Shape: {shape_str} (ndim={ndim}, numel={numel})")
    logger.info(f"  Dtype: {dtype_str}")
    
    repo.insert_schema(
        schema_id=schema_id,
        shape=shape_str,
        ndim=ndim,
        numel=numel,
        dtype=dtype_str,
        created_at=now
    )
    
    # Ingest each False Positive
    logger.info(f"Ingesting {len(X_fp)} samples...")
    
    for idx, (tensor, ts_ns) in enumerate(zip(X_fp, ts_fp)):
        # Generate anomaly ID
        anomaly_id = f"fp_{ts_ns}_{idx}"
        
        # Convert timestamp (nanoseconds) to ISO format
        occurred_at = datetime.fromtimestamp(ts_ns / 1e9, tz=timezone.utc).isoformat()
        
        # Insert anomaly record
        repo.upsert_anomaly(
            anomaly_id=anomaly_id,
            occurred_at=occurred_at,
            source=source,
            schema_id=schema_id,
            created_at=now,
            updated_at=now
        )
        
        # Serialize tensor to .npy format
        blob = encode_npy(tensor)
        
        # Store vector
        repo.put_vector(
            anomaly_id=anomaly_id,
            schema_id=schema_id,
            blob=blob,
            created_at=now
        )
        # Insert feedback derived from true label (map 1 -> 'TP', 0 -> 'FP')
        try:
            tl = int(true_labels[idx])
            label_str = "TP" if tl == 1 else "FP"
        except Exception:
            label_str = "FP"

        feedback_id = f"fb_{anomaly_id}"
        try:
            repo.insert_feedback(
                feedback_id=feedback_id,
                anomaly_id=anomaly_id,
                user_id="ingest_script",
                label=label_str,
                confidence=1.0,
                note=f"ingested_from_{source}",
                created_at=now,
            )
        except Exception:
            logger.debug("Failed to insert feedback for %s", anomaly_id, exc_info=True)
        
        # Progress logging
        if (idx + 1) % batch_size == 0:
            logger.info(f"  Ingested {idx + 1}/{len(X_fp)} samples...")
    
    logger.info(f"✓ Successfully ingested {len(X_fp)} False Positive samples")
    logger.info(f"  Schema ID: {schema_id}")
    logger.info(f"  Database: {db_path}")
    logger.info("\nNext steps:")
    logger.info("  1. Verify data: python -m hitl.cli list-anomalies --limit 5")
    logger.info(f"  2. Train model: python scripts/train_model.py --schema-id {schema_id}")


def main():
    """Main entry point."""
    args = parse_args()
    
    try:
        ingest_false_positives(
            data_dir=args.data_dir,
            db_path=args.db,
            schema_id=args.schema_id,
            source=args.source,
            limit=args.limit,
            batch_size=args.batch_size,
            apply_ddl=args.apply_ddl,
        )
    except Exception as e:
        logger = get_logger("ingest_fp")
        logger.error(f"Ingestion failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
