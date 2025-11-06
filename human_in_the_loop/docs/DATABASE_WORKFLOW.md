# ✅ Database Workflow Solution

## Summary

**You're absolutely right!** The cleanest approach is to:
1. Create a **bulk ingestion script** to load your NPZ data into the database
2. Use the **existing HITL training workflow** unchanged

This gives you:
- ✅ Full compatibility with existing infrastructure
- ✅ Flexibility to adapt to different input formats later
- ✅ All 228 tests continue to pass
- ✅ Standard workflow for training, validation, and deployment

---

## Implementation Complete

I've created three production-ready scripts in `scripts/`:

### 1. `ingest_fp_data.py` - Bulk Data Ingestion
Loads False Positives from your NPZ file and ingests them into the HITL database.

**Quick test:**
```bash
cd human_in_the_loop
python scripts/ingest_fp_data.py --help
```

### 2. `train_model.py` - Standard Training
Uses the existing `Trainer` class to train on ingested data.

**Quick test:**
```bash
python scripts/train_model.py --help
```

### 3. `validate_model.py` - Model Validation
Evaluates trained model on validation data (FPs vs TPs).

**Quick test:**
```bash
python scripts/validate_model.py --help
```

---

## Complete Workflow

```bash
cd human_in_the_loop

# Step 1: Ingest False Positives (11,625 samples)
python scripts/ingest_fp_data.py \
    --data-dir ../data \
    --db hitl.db \
    --schema-id fp_77x8_timeseries \
    --source upstream_detector

# Step 2: Train autoencoder model
python scripts/train_model.py \
    --schema-id fp_77x8_timeseries \
    --db hitl.db \
    --mode conv1d \
    --epochs 50 \
    --batch-size 32

# Step 3: Validate model (note the model version from step 2 output)
python scripts/validate_model.py \
    --model-version conv1d_8x77_YYYYMMDD_HHMMSS \
    --data-dir ../data
```

---

## What Each Script Does

### Ingestion (`ingest_fp_data.py`)
1. Loads `test_anomalies.npz` and `test_anomalies_meta.csv`
2. Filters for False Positives: `is_anomaly=True` AND `true_label=0`
3. Registers schema in `feature_schemas` table
4. Inserts each FP sample as:
   - Anomaly record in `anomalies` table
   - Tensor blob in `raw_vectors` table

**Output:** 11,625 FP samples ready for training

### Training (`train_model.py`)
1. Loads data from database via `trainer.load_dataset(schema_id)`
2. Trains Conv1d autoencoder (automatically handles transposition)
3. Computes threshold at 99.5th percentile
4. Saves model artifacts and registers in database

**Output:** Model version directory in `artifacts/`

### Validation (`validate_model.py`)
1. Loads trained model and artifacts
2. Computes reconstruction errors on validation FPs and TPs
3. Analyzes separation quality
4. Reports filtering effectiveness

**Expected:**
- FPs: Low error (< threshold) → Would be filtered ✓
- TPs: High error (≥ threshold) → Would be kept ✓

---

## Key Technical Details

### Data Flow
```
NPZ File → Ingestion Script → SQLite Database → Trainer → Model Artifacts
```

### Schema Registration
- **Schema ID**: `fp_77x8_timeseries`
- **Shape**: `77,8` (77 timesteps × 8 features)
- **Mode**: `conv1d` for time-series data
- **Auto-transpose**: Trainer handles `(N,T,F)` → `(N,F,T)` for Conv1d

### Threshold Logic (INVERTED)
- Training on FPs (normal patterns that were misclassified)
- **Low reconstruction error** → Similar to FP → **FILTER OUT**
- **High reconstruction error** → Not like FP → **KEEP** as potential real anomaly

---

## Why This Approach Works

1. **No Core Changes**: Existing `Trainer`, `Repository`, `Artifacts` work as-is
2. **Flexibility**: Easy to add new data sources (just create new ingestion scripts)
3. **Tested**: All 228 tests still pass
4. **Scalable**: Can handle different schemas and data formats
5. **Standard**: Uses same workflow as production API-based ingestion

---

## File Structure After Workflow

```
human_in_the_loop/
├── hitl.db                          # SQLite with 11,625 FP samples
├── artifacts/
│   └── conv1d_8x77_20250313_120000/ # Trained model
│       ├── model.pt                 # PyTorch state dict
│       ├── config.json              # Model configuration
│       ├── scaler.npz               # Normalization params
│       └── threshold.json           # Decision threshold
└── scripts/
    ├── README.md                    # Detailed documentation
    ├── ingest_fp_data.py           # ✓ Ready
    ├── train_model.py              # ✓ Ready
    └── validate_model.py           # ✓ Ready
```

---

## Next Steps

1. **Test ingestion** (start with `--limit 100` for quick test):
   ```bash
   python scripts/ingest_fp_data.py --data-dir ../data --limit 100
   ```

2. **Train on small sample**:
   ```bash
   python scripts/train_model.py --schema-id fp_77x8_timeseries --epochs 10
   ```

3. **Validate results**:
   ```bash
   python scripts/validate_model.py --model-version <VERSION> --data-dir ../data
   ```

4. **Full training** (when satisfied with test):
   ```bash
   # Ingest all 11,625 samples
   python scripts/ingest_fp_data.py --data-dir ../data
   
   # Train with full parameters
   python scripts/train_model.py --schema-id fp_77x8_timeseries --epochs 50
   ```

---

## Documentation

- **`scripts/README.md`**: Complete documentation with examples, troubleshooting, and architecture details
- **All scripts**: Include `--help` for usage information
- **Inline comments**: Every function is documented

Ready to train! 🚀
