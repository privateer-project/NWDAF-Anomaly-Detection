#!/usr/bin/env python3
"""Split an NPZ dataset and its metadata CSV into train/validation parts.

This script expects files like:
  <data-dir>/<dataset>_anomalies.npz
  <data-dir>/<dataset>_anomalies_meta.csv

It will produce:
  <out-dir>/<train_name>_anomalies.npz
  <out-dir>/<train_name>_anomalies_meta.csv
  <out-dir>/<val_name>_anomalies.npz
  <out-dir>/<val_name>_anomalies_meta.csv

By default it splits `validation` into `train` (90%) and `validation` (10%),
keeping the same NPZ keys and CSV columns.
"""
from __future__ import annotations

import argparse
import os
from typing import Dict, Any

import numpy as np
import pandas as pd


def load_npz(npz_path: str) -> Dict[str, np.ndarray]:
    with np.load(npz_path, allow_pickle=True) as data:
        return {k: data[k] for k in data.files}


def save_npz(npz_path: str, arrays: Dict[str, np.ndarray]) -> None:
    # preserve compressed form
    np.savez_compressed(npz_path, **arrays)


def split_indices(n: int, train_frac: float, seed: int | None = None):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_train = int(np.floor(n * train_frac))
    train_idx = idx[:n_train]
    val_idx = idx[n_train:]
    return train_idx, val_idx


def subset_arrays(arrays: Dict[str, np.ndarray], indices: np.ndarray) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for k, v in arrays.items():
        a = np.asarray(v)
        if a.ndim == 0:
            # scalar: keep as-is
            out[k] = a
        else:
            out[k] = a[indices]
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Split NPZ+CSV dataset into train/validation (90/10) preserving format")
    p.add_argument("--data-dir", default="../data", help="Directory containing NPZ and CSV files")
    p.add_argument("--dataset", default="validation", help="Base dataset name (without suffix). e.g. 'validation' -> 'validation_anomalies.npz'")
    p.add_argument("--out-dir", default=None, help="Output directory (defaults to --data-dir)")
    p.add_argument("--train-frac", type=float, default=0.9, help="Fraction of samples to allocate to train (default 0.9)")
    p.add_argument("--seed", type=int, default=42, help="RNG seed for reproducible split")
    p.add_argument("--train-name", default="train", help="Base name for train outputs (default: train)")
    p.add_argument("--val-name", default="validation", help="Base name for val outputs (default: validation)")
    p.add_argument("--npz-suffix", default="_anomalies.npz", help="Suffix for npz files")
    p.add_argument("--meta-suffix", default="_anomalies_meta.csv", help="Suffix for metadata CSV files")

    args = p.parse_args(argv)

    data_dir = os.path.abspath(args.data_dir)
    out_dir = os.path.abspath(args.out_dir) if args.out_dir else data_dir
    os.makedirs(out_dir, exist_ok=True)

    npz_path = os.path.join(data_dir, args.dataset + args.npz_suffix)
    meta_path = os.path.join(data_dir, args.dataset + args.meta_suffix)

    if not os.path.exists(npz_path):
        raise SystemExit(f"NPZ file not found: {npz_path}")
    if not os.path.exists(meta_path):
        raise SystemExit(f"Metadata CSV not found: {meta_path}")

    arrays = load_npz(npz_path)
    meta = pd.read_csv(meta_path)

    # Determine N from the metadata or first array
    n_meta = len(meta)
    # try to find array-length-consistent N
    candidate_ns = []
    for v in arrays.values():
        a = np.asarray(v)
        if a.ndim == 0:
            continue
        candidate_ns.append(a.shape[0])

    if candidate_ns:
        # ensure all candidate_ns are equal
        if len(set(candidate_ns)) != 1:
            raise SystemExit(f"Inconsistent array lengths in NPZ: {set(candidate_ns)}")
        n_arrays = candidate_ns[0]
    else:
        n_arrays = n_meta

    if n_arrays != n_meta:
        raise SystemExit(f"Length mismatch: NPZ arrays length={n_arrays}, CSV rows={n_meta}")

    train_idx, val_idx = split_indices(n_arrays, args.train_frac, seed=args.seed)

    train_arrays = subset_arrays(arrays, train_idx)
    val_arrays = subset_arrays(arrays, val_idx)

    train_meta = meta.iloc[train_idx].reset_index(drop=True)
    val_meta = meta.iloc[val_idx].reset_index(drop=True)

    train_npz = os.path.join(out_dir, args.train_name + args.npz_suffix)
    val_npz = os.path.join(out_dir, args.val_name + args.npz_suffix)
    train_csv = os.path.join(out_dir, args.train_name + args.meta_suffix)
    val_csv = os.path.join(out_dir, args.val_name + args.meta_suffix)

    save_npz(train_npz, train_arrays)
    save_npz(val_npz, val_arrays)
    train_meta.to_csv(train_csv, index=False)
    val_meta.to_csv(val_csv, index=False)

    print(f"Wrote train NPZ: {train_npz} ({len(train_idx)} samples)")
    print(f"Wrote val NPZ:   {val_npz} ({len(val_idx)} samples)")
    print(f"Wrote train CSV: {train_csv}")
    print(f"Wrote val CSV:   {val_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
