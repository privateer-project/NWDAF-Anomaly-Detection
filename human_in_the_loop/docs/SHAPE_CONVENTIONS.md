# Shape Conventions in HITL System

This document clarifies the tensor shape conventions used throughout the HITL system, particularly for Conv1d mode where shapes differ between storage and processing.

## Overview

The HITL system stores tensors in a **human-readable format** (time-first) but processes them in **PyTorch's expected format** (channels-first) for Conv1d mode. This requires shape transformations at specific points in the pipeline.

**Important: No data preprocessing/normalization is applied. All training and inference is performed on raw vectors from the database.**

---

## Dense Mode (1D Vectors)

Dense mode is straightforward with no transformations needed.

### Storage Format (per sample)
- **Shape**: `(D,)` where D = number of features
- **Example**: `(84,)` for 84 features

### Training Format (batched)
- **Shape**: `(N, D)` where N = batch size, D = features
- **Example**: `(128, 84)` for batch of 128 samples with 84 features

### No Transformation Required
- Stacking individual samples naturally creates the correct batch format
- NumPy: `X = np.stack(vectors, axis=0)` → `(N, D)`

---

## Conv1d Mode (2D Time-Series)

Conv1d mode requires careful attention to shape transformations due to different conventions between storage and PyTorch.

### Storage Format (per sample)
- **Shape**: `(T, F)` where T = timesteps, F = features/channels
- **Convention**: Time-first (intuitive, matches how humans think about time-series)
- **Example**: `(240, 84)` for 240 timesteps with 84 features each
- **Storage**: `.npy` BLOBs in SQLite database, shape preserved

### Training Format (batched)
- **Shape**: `(N, F, T)` where N = batch size, F = features/channels, T = timesteps
- **Convention**: Channels-first (PyTorch Conv1d requirement)
- **Example**: `(128, 84, 240)` for batch of 128 samples, 84 channels, 240 time points
- **Reason**: PyTorch `Conv1d` expects `(Batch, Channels, Length)` format

### Transformation Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    STORAGE TO TRAINING                          │
└─────────────────────────────────────────────────────────────────┘

Individual Samples (from DB):
  Sample 1: (240, 84) - Storage format (T, F)
  Sample 2: (240, 84)
  ...
  Sample N: (240, 84)

         ↓ np.stack(vectors, axis=0)

Stacked Array:
  X: (N, 240, 84) - Still in storage format (N, T, F)

         ↓ np.transpose(X, (0, 2, 1))

Training Array:
  X: (N, 84, 240) - Training format (N, F, T)

         ↓ PyTorch Conv1d

Model processes: (Batch, Channels=84, Length=240)
```

---

## Implementation Locations

### 1. Data Loading - `hitl/training/trainer.py`

**Method**: `Trainer.load_dataset(schema_id, mode)`

```python
# Stack individual samples
X = np.stack(vectors, axis=0)  # (N, T, F) for conv1d

# Transform for Conv1d mode
if mode == "conv1d" and X.ndim == 3:
    X = np.transpose(X, (0, 2, 1))  # (N, T, F) -> (N, F, T)
```

**Status**: ✅ Implemented

---

### 2. Training - `hitl/training/trainer.py`

**No normalization or preprocessing is applied.**

Training is performed directly on raw data from the database. The only transformations are:
- Train/val splitting
- Shape transposition for Conv1d mode (to match PyTorch format)

**Method**: `Trainer.fit(X, params)`

```python
# Split data
X_train, X_val = self._split_data(X, val_split)

# For Conv1d: X is already (N, F, T) from load_dataset
# Train directly on raw data - no normalization
```

**Status**: ✅ Implemented

---

### 3. Model Input - `hitl/models/ae.py`

**Class**: `Conv1dAE`

```python
def __init__(self, num_features: int, seq_len: int, ...):
    """
    Input format: (B, F, T) where B=batch, F=features, T=time
    PyTorch Conv1d expects channels-first format.
    """
```

**Method**: `build_model(mode, input_shape)`

```python
if mode == "conv1d":
    # input_shape comes from storage format
    seq_len, num_features = input_shape  # (T, F) from storage
    
    # But model expects (F, T) ordering in forward pass
    return Conv1dAE(
        num_features=num_features,  # F (channels)
        seq_len=seq_len,            # T (length)
    )
```

**Status**: ✅ Works correctly after data is transposed

---

### 4. Inference - `hitl/inference/serve.py`

**Class**: `LiveModel` (TO BE IMPLEMENTED)

**Method**: `predict_tensor(arr)`

```python
def predict_tensor(self, arr: np.ndarray) -> PredictResult:
    """
    Predict on a single sample.
    
    Args:
        arr: NumPy array in STORAGE format (raw data, no preprocessing)
            - Dense: (D,)
            - Conv1d: (T, F)
    
    Process:
        1. For Conv1d: transpose (T, F) -> (F, T)
        2. Add batch dimension: (F, T) -> (1, F, T)
        3. Forward pass (on raw data, no preprocessing)
        4. Compute reconstruction error
    """
    if self._config["mode"] == "conv1d":
        # Transform from storage format to model format
        arr = arr.T  # (T, F) -> (F, T)
    
    # Add batch dimension
    arr = arr[np.newaxis, ...]  # (F, T) -> (1, F, T)
    
    # Forward pass on raw data - no normalization
    # ... rest of prediction logic
```

**Status**: ⚠️ NOT YET IMPLEMENTED (stub with TODO)

---

## Model Architecture Details

### Conv1d Expectations

PyTorch `Conv1d` layer signature:
```python
nn.Conv1d(in_channels, out_channels, kernel_size, ...)
```

- Expects input: `(N, C_in, L)` where
  - N = batch size
  - C_in = number of input channels (our features)
  - L = sequence length (our timesteps)

- In our case: `(N, F, T)`
  - F = features = channels
  - T = timesteps = sequence length

### Why Channels-First?

PyTorch uses channels-first to optimize GPU memory access patterns:
- Convolution kernels operate on channels (features)
- Having channels contiguous in memory improves cache locality
- This is standard in PyTorch for all convolutional layers

---

## Testing Checklist

When testing Conv1d mode, verify:

- ✅ Input data stored as `(T, F)` in database
- ✅ Batch loaded as `(N, T, F)` initially
- ✅ Data transposed to `(N, F, T)` before training
- ✅ Scaler computed over axes `(0, 2)` for shape `(N, F, T)`
- ✅ Model forward pass works with `(N, F, T)` input
- ✅ Inference transposes single sample `(T, F)` → `(F, T)` before adding batch dim
- ✅ Reconstruction output shape matches input shape

---

## Quick Reference

| Context | Dense Mode | Conv1d Mode |
|---------|------------|-------------|
| **Single Sample (storage)** | `(D,)` | `(T, F)` |
| **Batch (after stacking)** | `(N, D)` | `(N, T, F)` |
| **Batch (for training)** | `(N, D)` | `(N, F, T)` ⚠️ |
| **Transformation needed?** | ❌ No | ✅ Yes: transpose |
| **Preprocessing?** | ❌ No | ❌ No |

⚠️ = Requires transpose operation

**Important:** Training and inference are performed on raw data with no normalization or preprocessing.

---

## Common Pitfalls

### ❌ Don't: Assume stacking creates the right shape for Conv1d
```python
X = np.stack(vectors, axis=0)  # (N, T, F) - WRONG for Conv1d!
model(torch.tensor(X))  # Will fail or produce wrong results
```

### ✅ Do: Explicitly transpose for Conv1d
```python
X = np.stack(vectors, axis=0)  # (N, T, F)
if mode == "conv1d":
    X = X.transpose(0, 2, 1)  # (N, F, T) - CORRECT!
model(torch.tensor(X))  # Works correctly
```

### ❌ Don't: Forget to transpose in inference
```python
# Single sample from DB: (240, 84)
arr = arr[np.newaxis, ...]  # (1, 240, 84) - WRONG shape!
```

### ✅ Do: Transpose before adding batch dimension
```python
# Single sample from DB: (240, 84) in storage format
arr = arr.T  # (84, 240) - model format
arr = arr[np.newaxis, ...]  # (1, 84, 240) - CORRECT!
```

---

## Related Files

- **This document**: `docs/SHAPE_CONVENTIONS.md`
- **Training**: `hitl/training/trainer.py`
- **Models**: `hitl/models/ae.py`
- **Inference**: `hitl/inference/serve.py`
- **Serialization**: `hitl/io/serialization.py`
- **Architecture spec**: `specs/architecture.md`

---

## Version History

- **2025-11-06**: Initial documentation, fixed transpose in trainer.py
