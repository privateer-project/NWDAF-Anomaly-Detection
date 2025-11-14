#!/usr/bin/env python3
"""
Detect schema for a dataset (NPZ) and print shape/dtype/ndim/numel.

Usage:
  python scripts/detect_schema.py --dataset test
  python scripts/detect_schema.py --npz path/to/file.npz --json

By default it looks for `../data/test_anomalies.npz` relative to the
`human_in_the_loop` directory (or `validation_anomalies.npz` when
`--dataset validation` is used).
"""

import argparse
import json
from pathlib import Path
from typing import Tuple

import numpy as np


def compute_schema_id(shape: Tuple[int, ...], dtype: str) -> str:
    shape_str = "x".join(map(str, shape))
    return f"fp_{shape_str}_{dtype}"


def detect_from_npz(npz_path: Path) -> dict:
    data = np.load(npz_path)
    if "X" not in data:
        raise ValueError(f"NPZ at {npz_path} does not contain 'X' array")

    X = data["X"]
    if X.ndim == 0:
        raise ValueError("Empty X array")

    sample_shape = tuple(X[0].shape)
    ndim = len(sample_shape)
    numel = int(np.prod(sample_shape))
    dtype = str(X.dtype)
    count = int(X.shape[0])

    return {
        "schema_id": compute_schema_id(sample_shape, dtype),
        "sample_shape": sample_shape,
        "ndim": ndim,
        "numel": numel,
        "dtype": dtype,
        "sample_count": count,
        "npz_path": str(npz_path),
    }


def parse_args():
    p = argparse.ArgumentParser(description="Detect dataset schema from NPZ")
    p.add_argument("--data-dir", type=str, default="../data", help="Directory containing npz files")
    p.add_argument("--dataset", choices=["test", "validation"], default="test", help="Which dataset NPZ to inspect")
    p.add_argument("--npz", type=str, default=None, help="Direct path to .npz file (overrides --data-dir/--dataset)")
    p.add_argument("--json", action="store_true", help="Output JSON instead of human-readable")
    return p.parse_args()


def main():
    args = parse_args()

    if args.npz:
        npz_path = Path(args.npz)
    else:
        name = f"{args.dataset}_anomalies.npz"
        npz_path = Path(args.data_dir) / name

    if not npz_path.exists():
        raise SystemExit(f"NPZ file not found: {npz_path}")

    info = detect_from_npz(npz_path)

    if args.json:
        print(json.dumps(info, indent=2))
    else:
        print("Dataset schema detection")
        print("------------------------")
        print(f"NPZ:          {info['npz_path']}")
        print(f"Samples:      {info['sample_count']}")
        print(f"Sample shape: {info['sample_shape']} (ndim={info['ndim']})")
        print(f"Numel:        {info['numel']}")
        print(f"Dtype:        {info['dtype']}")
        print(f"Schema ID:    {info['schema_id']}")


if __name__ == "__main__":
    main()
