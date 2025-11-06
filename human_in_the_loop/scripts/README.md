# Training Scripts for False Positive Filtering

This directory contains scripts for ingesting False Positive data and training autoencoder models using the HITL database workflow.

## Overview

The workflow consists of three main steps:

1. **Ingest** - Load FP data from NPZ files into the database
2. **Train** - Train autoencoder model on ingested data
3. **Validate** - Evaluate model performance on validation data

## Scripts

### 1. `ingest_fp_data.py` - Bulk Data Ingestion

Loads False Positive samples from `test_anomalies.npz` and ingests them into the HITL database.

**Usage:**
```bash
cd human_in_the_loop

# Ingest all False Positives
python scripts/ingest_fp_data.py \
    --data-dir ../data \
    --db hitl.db \
    --schema-id fp_77x8_timeseries \
    --source upstream_detector

# Test with limited samples
python scripts/ingest_fp_data.py \
    --data-dir ../data \
    --db hitl.db \
    --limit 100
```

**Arguments:**
- `--data-dir`: Directory containing `test_anomalies.npz` and `test_anomalies_meta.csv`
- `--db`: SQLite database path (default: `hitl.db`)
- `--schema-id`: Schema identifier for this data type (default: `fp_77x8_timeseries`)
- `--source`: Source system identifier (default: `upstream_detector`)
- `--limit`: Limit number of samples (useful for testing)

**Output:**
- Registers schema in `feature_schemas` table
- Inserts anomaly records in `anomalies` table
- Stores tensor blobs in `raw_vectors` table

---

### 2. `train_model.py` - Model Training

Trains an autoencoder model on ingested False Positive data using the standard HITL workflow.

**Usage:**
```bash
cd human_in_the_loop

# Train with default parameters
python scripts/train_model.py \
    --schema-id fp_77x8_timeseries \
    --db hitl.db \
    --mode conv1d

# Train with custom hyperparameters
python scripts/train_model.py \
    --schema-id fp_77x8_timeseries \
    --db hitl.db \
    --mode conv1d \
    --epochs 100 \
    --batch-size 64 \
    --lr 0.0005 \
    --val-split 0.15 \
    --patience 15
```

**Arguments:**
- `--schema-id`: Schema ID to train on (from ingestion) **[REQUIRED]**
- `--db`: Database path (default: `hitl.db`)
- `--artifacts-dir`: Directory for model artifacts (default: `artifacts`)
- `--mode`: Model architecture (`dense` or `conv1d`, default: `conv1d`)
- `--epochs`: Number of training epochs (default: 50)
- `--batch-size`: Training batch size (default: 32)
- `--lr`: Learning rate (default: 0.001)
- `--val-split`: Validation split ratio (default: 0.1)
- `--patience`: Early stopping patience (default: 10)
- `--percentile`: Threshold percentile (default: 99.5)

**Output:**
- Creates model version directory in `artifacts/`
- Saves: `model.pt`, `config.json`, `scaler.npz`, `threshold.json`
- Registers model in `trained_models` table

---

### 3. `validate_model.py` - Model Validation

Evaluates trained model on validation data to verify it can distinguish between False Positives and True Positives.

**Usage:**
```bash
cd human_in_the_loop

# Validate on validation set
python scripts/validate_model.py \
    --model-version conv1d_8x77_20250313_120000 \
    --data-dir ../data

# Validate on test set
python scripts/validate_model.py \
    --model-version conv1d_8x77_20250313_120000 \
    --data-dir ../data \
    --use-test-data
```

**Arguments:**
- `--model-version`: Model version directory name **[REQUIRED]**
- `--data-dir`: Directory containing validation/test NPZ files (default: `../data`)
- `--artifacts-dir`: Artifacts directory (default: `artifacts`)
- `--use-test-data`: Use test data instead of validation data

**Output:**
- Reconstruction error statistics for FPs and TPs
- Percentage of samples above/below threshold
- Separation quality metrics

**Expected Results:**
- **False Positives**: Low reconstruction error (< threshold) → Would be filtered
- **True Positives**: High reconstruction error (≥ threshold) → Would be kept

---

## Complete Workflow Example

```bash
cd human_in_the_loop

# Step 1: Ingest False Positive data
echo "Step 1: Ingesting False Positives..."
python scripts/ingest_fp_data.py \
    --data-dir ../data \
    --db hitl.db \
    --schema-id fp_77x8_timeseries

# Step 2: Train autoencoder model
echo "Step 2: Training model..."
python scripts/train_model.py \
    --schema-id fp_77x8_timeseries \
    --db hitl.db \
    --mode conv1d \
    --epochs 50 \
    --batch-size 32

# Note the model version from training output (e.g., conv1d_8x77_20250313_120000)

# Step 3: Validate model
echo "Step 3: Validating model..."
python scripts/validate_model.py \
    --model-version conv1d_8x77_20250313_120000 \
    --data-dir ../data

echo "Done! Model is ready for deployment."
```

---

## Data Format Requirements

### Input NPZ File
- **File**: `test_anomalies.npz` or `validation_anomalies.npz`
- **Keys**:
  - `X`: Array of shape `(N, T, F)` - N samples, T timesteps, F features
  - `y`: Labels (optional)
  - `is_anom`: Boolean array indicating anomaly detections
  - `ts_ns`: Timestamps in nanoseconds

### Metadata CSV
- **File**: `test_anomalies_meta.csv` or `validation_anomalies_meta.csv`
- **Columns**:
  - `timestamp_ns`: Timestamp in nanoseconds
  - `timestamp`: Human-readable timestamp
  - `true_label`: Ground truth label (0=normal, 1=anomaly)
  - `is_anomaly`: Detector prediction (True/False)

### False Positive Definition
A False Positive is a sample where:
- `is_anomaly == True` (detector predicted anomaly)
- `true_label == 0` (but was actually normal)

---

## Architecture: Conv1d for Time-Series

For time-series data with shape `(T, F)`:
- Input shape: `(77, 8)` - 77 timesteps × 8 features
- Mode: `conv1d` (1D convolutions along time axis)
- Automatic transpose: `(N, T, F)` → `(N, F, T)` for PyTorch Conv1d

The Conv1d architecture uses:
- 1D convolutions to capture temporal patterns
- Encoder: 8 → 16 → 32 channels
- Decoder: 32 → 16 → 8 channels
- Kernel size: 3, stride: 1

---

## Threshold Interpretation

**Standard Anomaly Detection:**
- High reconstruction error → Anomaly

**False Positive Filtering (this use case):**
- Model trained on False Positives (samples that look normal)
- **Low reconstruction error** → Similar to FP → **FILTER OUT**
- **High reconstruction error** → Not like FP → **KEEP** as potential real anomaly

The threshold is set at the 99.5th percentile of training reconstruction errors.

---

## Troubleshooting

### Issue: Schema not found
```
Error: Schema not found: fp_77x8_timeseries
```
**Solution**: Run ingestion script first to register schema

### Issue: No data ingested
```
Loaded 0 False Positive samples
```
**Solution**: Check that `test_anomalies_meta.csv` has correct column names (`is_anomaly`, `true_label`)

### Issue: Shape mismatch
```
RuntimeError: shape mismatch
```
**Solution**: Ensure `--mode conv1d` for shape `(77, 8)` data

### Issue: CUDA out of memory
```
RuntimeError: CUDA out of memory
```
**Solution**: Reduce `--batch-size` or train on CPU (automatic fallback)

---

## Next Steps After Training

1. **Deploy model**: Copy `artifacts/{model_version}/` to production
2. **Inference**: Use HITL API or inference module
3. **Monitor**: Track filter rates and feedback
4. **Retrain**: Periodically retrain with new feedback data

---

## File Structure After Workflow

```
human_in_the_loop/
├── hitl.db                          # SQLite database with ingested data
├── artifacts/
│   └── conv1d_8x77_20250313_120000/ # Model artifacts
│       ├── model.pt                 # Model weights
│       ├── config.json              # Model configuration
│       ├── scaler.npz               # Normalization parameters
│       └── threshold.json           # Decision threshold
└── scripts/
    ├── ingest_fp_data.py
    ├── train_model.py
    └── validate_model.py
```
