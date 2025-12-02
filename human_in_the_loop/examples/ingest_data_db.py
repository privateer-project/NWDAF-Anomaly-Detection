#!/usr/bin/env python3
"""Ingest train anomalies into the HITL DB and submit feedback.

This is a split-out portion of `run_train_eval.py` that only ingests the
`train_anomalies.npz` / `train_anomalies_meta.csv` into the DB and then
submits automatic feedback labels (TP/FP/TN/FN) based on `is_anomaly`
predictions and `true_label` ground-truth in the metadata CSV.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from hitl import HITL
from hitl.settings import Config, paths, validate_config

from hitl.utils.logging import configure_logging, get_logger


configure_logging("INFO")
logger = get_logger("ingest_data_db")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="../data")
    p.add_argument("--db-path", default="db/hitl.db")
    p.add_argument("--apply-ddl", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--artifacts-dir", default="artifacts_demo")
    return p.parse_args()


def main():
    args = parse_args()

    data_dir = Path(args.data_dir)
    npz_path = data_dir / "final_anomalies.npz"
    csv_path = data_dir / "final_anomalies_meta.csv"

    if not npz_path.exists() or not csv_path.exists():
        raise SystemExit(f"Missing data files in {data_dir}; expected final_anomalies.npz and final_anomalies_meta.csv")
    logger.info("Loading train data")
    npz = np.load(npz_path, allow_pickle=True)
    X = npz["X"]
    meta = pd.read_csv(csv_path)

    if args.limit is not None:
        X = X[: args.limit]
        meta = meta.iloc[: args.limit]

    n = len(X)
    logger.info("Train samples: %d, sample shape: %s", n, X[0].shape)

    # Setup config and HITL
    cfg = Config(sqlite_path=str(Path(args.db_path)), artifacts_dir=str(Path(args.artifacts_dir)))
    validate_config(cfg)
    paths(cfg)

    hitl = HITL(config=cfg)

    if args.apply_ddl:
        try:
            hitl.db._apply_ddl()
            logger.info("Applied DDL")
        except Exception:
            logger.debug("DDL apply failed/ignored", exc_info=True)

    # Register schema (use registry)
    sample_shape = tuple(X[0].shape)
    dtype = str(X.dtype)
    info = hitl.registry.ensure(sample_shape, dtype)
    schema_id = info["schema_id"]
    logger.info("Using schema: %s", schema_id)

    # Ingest train set into DB using HITL core API
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc).isoformat()

    logger.info("Ingesting into DB via HITL.upsert_anomaly")
    for i in range(n):
        aid = f"train-{i:08d}"
        anomaly = {
            "anomaly_id": aid,
            "occurred_at": now,
            "source": "train_npz",
            "created_at": now,
            "updated_at": now,
        }
        # HITL.upsert_anomaly will register schema and store the tensor blob
        hitl.upsert_anomaly(anomaly=anomaly, tensor=X[i], dtype=dtype)
        if (i + 1) % 1000 == 0:
            logger.info("Inserted %d/%d samples", i + 1, n)

    logger.info("Ingest complete: %d samples", n)

    # Use predicted detection flags from the metadata CSV's `is_anomaly` column
    if "is_anomaly" in meta.columns:
        preds = [1 if str(x).lower() in ("1", "true", "yes") else 0 for x in meta["is_anomaly"].values]
    else:
        logger.warning("Metadata column 'is_anomaly' not found in train metadata; defaulting to no detections")
        preds = [0] * n

    # Submit feedback based on true_label and predicted label
    logger.info("Submitting feedback for train anomalies")
    for i in range(n):
        aid = f"train-{i:08d}"
        true_label = int(meta.iloc[i]["true_label"]) if "true_label" in meta.columns else int(meta.iloc[i].get("is_anomaly", 0))
        pred = int(preds[i]) if preds is not None else 0

        if pred == 1 and true_label == 1:
            label = "TP"
        elif pred == 1 and true_label == 0:
            label = "FP"
        elif pred == 0 and true_label == 0:
            label = "TN"
        else:
            label = "FN"

        uid = "script"
        fid = hitl.submit_feedback(aid, label, uid, confidence=None, note="auto-label-from-ground-truth")
        
        if (i + 1) % 1000 == 0:
            logger.info("Inserted %d/%d feedback entries", i + 1, n)

    logger.info("Feedback submission complete")
    print(schema_id)


if __name__ == "__main__":
    main()
