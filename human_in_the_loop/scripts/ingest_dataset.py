#!/usr/bin/env python3
"""Ingest an NPZ dataset (train/validation/test) into the HITL DB.

Writes anomalies and raw_vectors and registers schema. Prints schema_id on success.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from hitl.store.sqlite import SQLite
from hitl.store.repository import Repository
from hitl.schemas.registry import SchemaRegistry
from hitl.io.serialization import encode_npy
from hitl.utils.logging import configure_logging, get_logger

configure_logging("INFO")
logger = get_logger("ingest_dataset")


def parse_args():
    p = argparse.ArgumentParser(description="Ingest NPZ+CSV dataset into HITL DB")
    p.add_argument("--data-dir", default="../data", help="Directory with NPZ and CSV files")
    p.add_argument("--dataset", default="train", help="Base dataset name (train/validation/test)")
    p.add_argument("--db-path", default="db/hitl.db", help="SQLite DB path")
    p.add_argument("--source", default="manual_ingest", help="Source string for anomaly ids")
    p.add_argument("--schema-id", default=None, help="Optional schema_id to assign (skip registry.ensure)")
    p.add_argument("--limit", type=int, default=None, help="Optional sample limit for testing")
    p.add_argument("--apply-ddl", action="store_true", help="Apply DDL if needed")
    return p.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    npz_path = data_dir / f"{args.dataset}_anomalies.npz"
    csv_path = data_dir / f"{args.dataset}_anomalies_meta.csv"

    if not npz_path.exists():
        raise SystemExit(f"NPZ not found: {npz_path}")
    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path}")

    logger.info(f"Loading {npz_path}")
    data = np.load(npz_path, allow_pickle=True)
    X = data["X"]
    meta = pd.read_csv(csv_path)

    if args.limit is not None:
        X = X[: args.limit]
        meta = meta.iloc[: args.limit]

    logger.info(f"Samples: {len(X)}, sample shape: {X[0].shape}, dtype: {X.dtype}")

    db = SQLite(args.db_path)
    repo = Repository(db)
    registry = SchemaRegistry(repo)

    if args.apply_ddl:
        try:
            db._apply_ddl()
            logger.info("Applied DDL to database")
        except Exception:
            logger.debug("DDL apply failed or already applied", exc_info=True)

    # determine schema
    sample_shape = X[0].shape
    dtype = str(X.dtype)

    if args.schema_id:
        schema_id = args.schema_id
        logger.info(f"Using provided schema_id: {schema_id}")
    else:
        info = registry.ensure(sample_shape, dtype)
        schema_id = info["schema_id"]
        logger.info(f"Registered schema: {schema_id}")

    # Insert rows
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc).isoformat()
    for i in range(len(X)):
        anomaly_id = f"{args.source}-{i:08d}"
        occurred_at = now
        repo.upsert_anomaly(anomaly_id=anomaly_id, occurred_at=occurred_at, source=args.source, schema_id=schema_id, created_at=now, updated_at=now)
        blob = encode_npy(X[i])
        repo.put_vector(anomaly_id=anomaly_id, schema_id=schema_id, blob=blob, created_at=now)

        if (i + 1) % 1000 == 0:
            logger.info(f"Inserted {i+1}/{len(X)}")

    logger.info(f"Ingest complete: {len(X)} samples -> schema: {schema_id}")
    print(schema_id)


if __name__ == "__main__":
    main()
