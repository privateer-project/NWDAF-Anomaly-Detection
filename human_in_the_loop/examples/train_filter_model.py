#!/usr/bin/env python3
"""Train a filter model (split from run_train_eval).

This script trains using the HITL high-level API. It accepts training
parameters and a `schema_id` to train on (if omitted the registry will be
queried). After training the model is published and set as the live model.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from hitl import HITL
from hitl.settings import Config, paths, validate_config
from hitl.utils.logging import configure_logging, get_logger


configure_logging("INFO", dev_mode=True)
logger = get_logger("train_filter_model")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["dense", "conv1d"], default=None, help="Model mode to train (overrides config)")
    p.add_argument("--db-path", default="db/hitl.db")
    p.add_argument("--artifacts-dir", default="artifacts_demo")
    p.add_argument("--schema-id", default=None)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val-split", type=float, default=0.1)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--percentile", type=float, default=99.5)
    p.add_argument("--only-false-positives", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    cfg = Config(sqlite_path=str(Path(args.db_path)), artifacts_dir=str(Path(args.artifacts_dir)))
    validate_config(cfg)
    paths(cfg)

    hitl = HITL(config=cfg)

    # Allow overriding mode from CLI (useful when dataset shape requires conv1d)
    if args.mode:
        hitl.config.mode = args.mode
    
    train_params = {
        "mode": hitl.config.mode,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "val_split": args.val_split,
        "patience": args.patience,
        "percentile": args.percentile,
        "only_false_positives": True,
    }
    
    print("Training with parameters:", train_params)
    # return

    try:
        model_version = hitl.train_model(schema_id=args.schema_id, params=train_params)
        logger.info("Trained and published model: %s", model_version)
        hitl.set_live_model(model_version)
        print(model_version)
    except Exception as e:
        logger.exception("Training failed: %s", e)


if __name__ == "__main__":
    main()
