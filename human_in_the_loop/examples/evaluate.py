#!/usr/bin/env python3
"""Evaluate live model on validation dataset (split from run_train_eval).

This script loads `validation_anomalies.npz` and `validation_anomalies_meta.csv`
and runs `HITL.filter_predict` on each sample. Results are saved to
`examples/validation_results.json` (counts and per-sample results).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from hitl import HITL
from hitl.settings import Config, paths, validate_config
from hitl.utils.logging import configure_logging, get_logger


configure_logging("INFO")
logger = get_logger("evaluate")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="../data")
    p.add_argument("--db-path", default="db/hitl.db")
    p.add_argument("--artifacts-dir", default="artifacts_demo")
    return p.parse_args()


def main():
    args = parse_args()

    data_dir = Path(args.data_dir)
    val_npz = data_dir / "validation_anomalies.npz"
    val_csv = data_dir / "validation_anomalies_meta.csv"
    if not val_npz.exists() or not val_csv.exists():
        logger.error("Validation files missing in %s", data_dir)
        raise SystemExit(1)

    val_data = np.load(val_npz, allow_pickle=True)
    X_val = val_data["X"]
    meta_val = pd.read_csv(val_csv)

    cfg = Config(sqlite_path=str(Path(args.db_path)), artifacts_dir=str(Path(args.artifacts_dir)))
    validate_config(cfg)
    paths(cfg)

    hitl = HITL(config=cfg)

    results = []
    counts = {"TP": 0, "FP": 0, "TN": 0, "FN": 0}
    
    print(meta_val.shape)
    print(meta_val.head())
    return

    for i in range(len(X_val)):
        sample = X_val[i]
        pred = hitl.filter_predict(tensor=sample)
        predicted_label = int(pred["label"]) if pred and "label" in pred else 0
        true_label = int(meta_val.iloc[i]["true_label"]) if "true_label" in meta_val.columns else int(meta_val.iloc[i].get("is_anomaly", 0))

        if predicted_label == 1 and true_label == 1:
            k = "TP"
        elif predicted_label == 1 and true_label == 0:
            k = "FP"
        elif predicted_label == 0 and true_label == 0:
            k = "TN"
        else:
            k = "FN"

        counts[k] += 1

        results.append(
            {
                "index": i,
                "predicted": predicted_label,
                "true": int(true_label),
                "label": k,
                "score": float(pred.get("score", 0.0)) if pred else 0.0,
                "model_version": pred.get("model_version") if pred else None,
            }
        )

    total = sum(counts.values())
    logger.info("Evaluation complete. Samples: %d", total)
    logger.info("Counts: %s", counts)
    if total > 0:
        precision = counts["TP"] / (counts["TP"] + counts["FP"]) if (counts["TP"] + counts["FP"]) > 0 else 0.0
        recall = counts["TP"] / (counts["TP"] + counts["FN"]) if (counts["TP"] + counts["FN"]) > 0 else 0.0
        logger.info("Precision: %.4f, Recall: %.4f", precision, recall)

    out_path = Path(__file__).parent / "validation_results.json"
    with out_path.open("w") as fh:
        json.dump({"counts": counts, "results": results}, fh, indent=2)

    logger.info("Saved validation results to %s", out_path)


if __name__ == "__main__":
    main()
