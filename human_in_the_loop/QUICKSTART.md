# Quick Start Guide

This guide will help you train and evaluate an autoencoder model on your anomaly data.

## Prerequisites

```bash
# Make sure you're in the human_in_the_loop directory
cd /home/agaros/mywork/NWDAF-Anomaly-Detection/human_in_the_loop

# Verify tests pass
make test
```

## Your Data

Your data is located in `../data/`:
- **Shape**: `(12394, 77, 8)` - 12,394 samples, 77 timesteps, 8 features
- **Type**: `float32`
- **Metadata**: Includes timestamps, true labels, and anomaly flags

## Training Options

### Option 1: Conv1D Mode (Recommended for Time-Series)

Best for preserving temporal patterns in your 2D time-series data.

```bash
python scripts/train_and_evaluate.py \
    --mode conv1d \
    --use-validation \
    --epochs 50 \
    --batch-size 32 \
    --percentile 95.0
```

### Option 2: Dense Mode (For Flattened Vectors)

Treats the data as flat 1D vectors (77 × 8 = 616 features).

```bash
python scripts/train_and_evaluate.py \
    --mode dense \
    --use-validation \
    --epochs 50 \
    --batch-size 32 \
    --percentile 95.0
```

## Command Line Options

```
--mode              : 'conv1d' or 'dense' (default: conv1d)
--use-validation    : Use validation_anomalies.npz (default: test_anomalies.npz)
--data-dir          : Data directory (default: ../data)
--db-path           : SQLite database path (default: data/hitl.db)
--artifacts-dir     : Model artifacts directory (default: artifacts)
--epochs            : Training epochs (default: 50)
--batch-size        : Batch size (default: 32)
--lr                : Learning rate (default: 0.001)
--percentile        : Anomaly threshold percentile (default: 95.0)
```

## What the Script Does

1. **Loads Data**: Reads `.npz` file and metadata
2. **Database Setup**: Inserts anomalies into SQLite database
3. **Schema Registration**: Registers data shape and dtype
4. **Training**: 
   - Splits data into train/validation (90/10)
   - Trains autoencoder with early stopping
   - Computes reconstruction threshold at specified percentile
5. **Artifact Storage**: Saves model weights, config, and threshold
6. **Evaluation**: Computes reconstruction errors and classification metrics
7. **Report**: Prints detailed statistics

## Expected Output

```
==============================================================
TRAIN AND EVALUATE WORKFLOW
==============================================================
Mode: conv1d
Dataset: validation
...

==============================================================
LOADING DATA
==============================================================
Loaded X with shape: (12394, 77, 8), dtype: float32
...

==============================================================
TRAINING MODEL
==============================================================
Epoch 1/50: train_loss=0.123456, val_loss=0.234567
Epoch 2/50: train_loss=0.098765, val_loss=0.198765
...

✓ Model trained and published: AE-2025.11.06-1

==============================================================
EVALUATING MODEL
==============================================================
Reconstruction Errors:
  Mean:   0.012345
  Std:    0.023456
  Min:    0.000123
  Max:    0.456789
  Median: 0.009876

Threshold: 0.045678
Predictions: 623/12394 classified as anomalous (5.03%)

Error Percentiles:
  p 50.0: 0.009876
  p 75.0: 0.015432
  p 90.0: 0.028765
  p 95.0: 0.045678
  p 99.0: 0.098765
  p 99.5: 0.123456

==============================================================
WORKFLOW COMPLETE
==============================================================
Model Version: AE-2025.11.06-1
Data Samples: 12394
Mean Error: 0.012345
Threshold: 0.045678
Anomaly Rate: 5.03% (623/12394)

Artifacts saved to: artifacts/AE-2025.11.06-1
Database: data/hitl.db
```

## After Training

### Inspect Model Artifacts

```bash
ls -lh artifacts/AE-2025.11.06-1/
# Output:
# model.pt        - PyTorch state dict
# config.json     - Model configuration
# threshold.json  - Anomaly threshold
```

### View Config

```bash
cat artifacts/AE-2025.11.06-1/config.json
```

### Use the Model

The trained model can be used for:
- Real-time anomaly detection
- Batch prediction on new data
- Integration with the inference server (see `hitl/inference/serve.py`)

## Troubleshooting

### Database Already Populated

If you see "UNIQUE constraint failed", remove the old database:
```bash
rm data/hitl.db
```

### CUDA Out of Memory

Reduce batch size:
```bash
python scripts/train_and_evaluate.py --mode conv1d --batch-size 16
```

### Training Too Slow

Reduce epochs:
```bash
python scripts/train_and_evaluate.py --mode conv1d --epochs 20
```

### Poor Results

Try different percentiles or modes:
```bash
# More sensitive (lower threshold)
python scripts/train_and_evaluate.py --mode conv1d --percentile 90.0

# Less sensitive (higher threshold)
python scripts/train_and_evaluate.py --mode conv1d --percentile 99.0
```

## Next Steps

1. **Experiment with Hyperparameters**:
   - Try different `--percentile` values (90, 95, 99)
   - Adjust `--epochs` (20, 50, 100)
   - Change `--batch-size` (16, 32, 64)

2. **Compare Modes**:
   - Run both `--mode conv1d` and `--mode dense`
   - Compare reconstruction errors and anomaly rates

3. **Evaluate on Test Set**:
   ```bash
   # Train on validation, evaluate on validation
   python scripts/train_and_evaluate.py --mode conv1d --use-validation
   
   # Train on test, evaluate on test
   python scripts/train_and_evaluate.py --mode conv1d
   ```

4. **Integrate with Production**:
   - Use the trained model in your inference pipeline
   - See `scripts/validate_model.py` for evaluation on separate datasets
   - Implement real-time prediction using `hitl/inference/serve.py`

## Architecture Notes

- **No Preprocessing**: The system trains on raw data (no normalization/scaling)
- **Shape Convention**: 
  - Storage: `(T, F)` for time-series
  - Training: `(N, F, T)` for Conv1d models (transposed automatically)
- **Early Stopping**: Training stops if validation loss doesn't improve for 10 epochs
- **Threshold**: Computed from training set reconstruction errors at specified percentile

## Getting Help

```bash
# Show all available options
python scripts/train_and_evaluate.py --help

# Run unit tests
make test

# Check code quality
make check
```
