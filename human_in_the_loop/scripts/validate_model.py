#!/usr/bin/env python3
"""
Validate trained autoencoder model.

This script loads validation data (validation_anomalies.npz) and evaluates
the trained model's reconstruction errors on both False Positives and True Positives
to verify that the model can distinguish between them.

Usage:
    python scripts/validate_model.py --model-version conv1d_8x77_20250313_120000 --data-dir ../data
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hitl.artifacts.manager import Artifacts
from hitl.models.ae import build_model
from hitl.utils.logging import get_logger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Validate trained autoencoder model"
    )
    parser.add_argument(
        "--model-version",
        type=str,
        required=True,
        help="Model version to validate (e.g., conv1d_8x77_20250313_120000)"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="../data",
        help="Directory containing validation_anomalies.npz"
    )
    parser.add_argument(
        "--artifacts-dir",
        type=str,
        default="artifacts",
        help="Artifacts directory"
    )
    parser.add_argument(
        "--use-test-data",
        action="store_true",
        help="Use test data instead of validation data"
    )
    return parser.parse_args()


def load_validation_data(data_dir: str, use_test: bool = False):
    """
    Load validation or test data.
    
    Args:
        data_dir: Data directory path
        use_test: Use test data instead of validation
    
    Returns:
        Tuple of (X_fp, X_tp) arrays
    """
    logger = get_logger("validate")
    
    # Choose dataset
    filename = "test_anomalies" if use_test else "validation_anomalies"
    data_path = Path(data_dir) / f"{filename}.npz"
    meta_path = Path(data_dir) / f"{filename}_meta.csv"
    
    logger.info(f"Loading data from {data_path}")
    
    if not data_path.exists():
        logger.error(f"Data file not found: {data_path}")
        sys.exit(1)
    
    data = np.load(data_path)
    meta = pd.read_csv(meta_path)
    
    # Split FPs and TPs
    fp_mask = (meta['is_anomaly']) & (meta['true_label'] == 0)
    tp_mask = (meta['is_anomaly']) & (meta['true_label'] == 1)
    
    X_fp = data['X'][fp_mask]
    X_tp = data['X'][tp_mask]
    
    logger.info(f"Loaded {len(X_fp)} False Positives, {len(X_tp)} True Positives")
    logger.info(f"Shape: {X_fp[0].shape if len(X_fp) > 0 else 'N/A'}")
    
    return X_fp, X_tp


def compute_reconstruction_errors(model, X, device):
    """
    Compute reconstruction errors for samples.
    
    Args:
        model: Trained autoencoder
        X: Input data (N, *) - raw, no preprocessing
        device: Torch device
    
    Returns:
        Array of reconstruction errors
    """
    model.eval()
    errors = []
    
    with torch.no_grad():
        # Convert to tensor (no normalization)
        X_tensor = torch.from_numpy(X).float().to(device)
        
        # Reconstruct
        X_recon = model(X_tensor)
        
        # Compute MSE per sample
        mse = ((X_tensor - X_recon) ** 2).mean(dim=tuple(range(1, X_tensor.ndim)))
        errors = mse.cpu().numpy()
    
    return errors


def validate_model(
    model_version: str,
    data_dir: str,
    artifacts_dir: str,
    use_test: bool
):
    """
    Validate trained model on FP/TP data.
    
    Args:
        model_version: Model version to load
        data_dir: Data directory
        artifacts_dir: Artifacts directory
        use_test: Use test data instead of validation
    """
    logger = get_logger("validate")
    
    # Load model artifacts
    logger.info(f"Loading model: {model_version}")
    artifacts = Artifacts(artifacts_dir)
    
    model_dir = Path(artifacts_dir) / model_version
    if not model_dir.exists():
        logger.error(f"Model not found: {model_dir}")
        sys.exit(1)
    
    # Load config
    config = artifacts.load_config(model_version)
    logger.info(f"Model config: {config}")
    
    # Load threshold
    threshold = artifacts.load_threshold(model_version)
    logger.info(f"Threshold: {threshold['value']:.6f} (p{threshold['percentile']})")
    
    # Build model
    mode = config['mode']
    input_shape = tuple(config['input_shape'])
    logger.info(f"Building {mode} model with input shape {input_shape}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(mode, input_shape).to(device)
    
    # Load weights
    state_dict = artifacts.load_model(model_version)
    model.load_state_dict(state_dict)
    logger.info(f"Model loaded on {device}")
    
    # Load validation data
    X_fp, X_tp = load_validation_data(data_dir, use_test)
    
    if len(X_fp) == 0 and len(X_tp) == 0:
        logger.error("No validation data found!")
        sys.exit(1)
    
    # Transpose for Conv1d if needed
    if mode == "conv1d":
        X_fp = np.transpose(X_fp, (0, 2, 1)) if len(X_fp) > 0 else X_fp
        X_tp = np.transpose(X_tp, (0, 2, 1)) if len(X_tp) > 0 else X_tp
        logger.info(f"Transposed to Conv1d format: {X_fp[0].shape if len(X_fp) > 0 else 'N/A'}")
    
    # Compute reconstruction errors
    logger.info("\nComputing reconstruction errors...")
    
    errors_fp = compute_reconstruction_errors(model, X_fp, device) if len(X_fp) > 0 else np.array([])
    errors_tp = compute_reconstruction_errors(model, X_tp, device) if len(X_tp) > 0 else np.array([])
    
    # Analyze results
    logger.info("\n" + "="*60)
    logger.info("VALIDATION RESULTS")
    logger.info("="*60)
    
    if len(errors_fp) > 0:
        logger.info(f"\nFalse Positives (n={len(errors_fp)}):")
        logger.info(f"  Mean error: {errors_fp.mean():.6f}")
        logger.info(f"  Std error:  {errors_fp.std():.6f}")
        logger.info(f"  Min error:  {errors_fp.min():.6f}")
        logger.info(f"  Max error:  {errors_fp.max():.6f}")
        logger.info(f"  Median:     {np.median(errors_fp):.6f}")
        
        # Count how many FPs are below threshold (should be filtered)
        below_thresh = (errors_fp < threshold['value']).sum()
        pct_below = 100 * below_thresh / len(errors_fp)
        logger.info(f"  Below threshold: {below_thresh}/{len(errors_fp)} ({pct_below:.1f}%)")
        logger.info("    → These would be FILTERED OUT (correct)")
    
    if len(errors_tp) > 0:
        logger.info(f"\nTrue Positives (n={len(errors_tp)}):")
        logger.info(f"  Mean error: {errors_tp.mean():.6f}")
        logger.info(f"  Std error:  {errors_tp.std():.6f}")
        logger.info(f"  Min error:  {errors_tp.min():.6f}")
        logger.info(f"  Max error:  {errors_tp.max():.6f}")
        logger.info(f"  Median:     {np.median(errors_tp):.6f}")
        
        # Count how many TPs are above threshold (should be kept)
        above_thresh = (errors_tp >= threshold['value']).sum()
        pct_above = 100 * above_thresh / len(errors_tp)
        logger.info(f"  Above threshold: {above_thresh}/{len(errors_tp)} ({pct_above:.1f}%)")
        logger.info("    → These would be KEPT (correct)")
    
    # Overall assessment
    logger.info(f"\nThreshold: {threshold['value']:.6f}")
    logger.info("\nInterpretation:")
    logger.info("  • Low error (< threshold) → Similar to FP → FILTER OUT")
    logger.info("  • High error (≥ threshold) → Not like FP → KEEP as potential real anomaly")
    
    # Separation quality
    if len(errors_fp) > 0 and len(errors_tp) > 0:
        logger.info("\nSeparation Quality:")
        gap = errors_tp.mean() - errors_fp.mean()
        logger.info(f"  Mean gap (TP - FP): {gap:.6f}")
        
        if gap > 0:
            logger.info("  ✓ Good! TPs have higher error than FPs")
        else:
            logger.info("  ⚠ Warning: TPs have lower/similar error to FPs")
            logger.info("    Model may need more training or different architecture")


def main():
    """Main entry point."""
    args = parse_args()
    
    try:
        validate_model(
            model_version=args.model_version,
            data_dir=args.data_dir,
            artifacts_dir=args.artifacts_dir,
            use_test=args.use_test
        )
    except Exception as e:
        logger = get_logger("validate")
        logger.error(f"Validation failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
