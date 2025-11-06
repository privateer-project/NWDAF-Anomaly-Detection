# Conv1d Shape Handling Fix - Implementation Summary

**Date**: 2025-11-06  
**Issue**: Critical misalignment between storage format and model expectations for Conv1d mode  
**Status**: ✅ FIXED

---

## Problem Description

The architecture specifications defined that Conv1d tensors should be stored in `(T, F)` format (time-first), which is intuitive for users. However, PyTorch's `Conv1d` layer requires `(B, F, T)` format (channels-first) for batch processing.

**The original code was missing the transpose operation** to convert from storage format to model format, which would have caused:
- Shape mismatch errors during training
- Incorrect model behavior (treating time as channels and vice versa)
- Silent failures in reconstruction quality

---

## Solution Implemented

### 1. Data Loading Transpose ✅

**File**: `hitl/training/trainer.py`

**Changes**:
- Modified `load_dataset()` method to accept `mode` parameter
- Added transpose operation for Conv1d mode: `(N, T, F)` → `(N, F, T)`
- Added detailed docstring explaining storage vs training formats
- Added logging for the transpose operation

**Code**:
```python
def load_dataset(self, schema_id: str, mode: str) -> tuple[np.ndarray, list[str]]:
    # ... load vectors from database ...
    X = np.stack(vectors, axis=0)  # Shape: (N, T, F) for conv1d
    
    # NEW: Transpose for Conv1d mode
    if mode == "conv1d" and X.ndim == 3:
        X = np.transpose(X, (0, 2, 1))  # (N, T, F) -> (N, F, T)
        self.logger.info("Transposed data for Conv1d", ...)
    
    return X, ids
```

- Updated `train_and_publish()` to pass `mode` to `load_dataset()`

---

### 2. Comprehensive Documentation ✅

**File**: `docs/SHAPE_CONVENTIONS.md` (NEW)

**Contents**:
- Clear explanation of storage vs processing formats
- Visual diagram showing transformation pipeline
- Implementation locations for all shape transformations
- Testing checklist
- Common pitfalls and how to avoid them
- Quick reference table

**Key Conventions Documented**:

| Context | Dense Mode | Conv1d Mode |
|---------|------------|-------------|
| Single Sample (storage) | `(D,)` | `(T, F)` |
| Batch (after stacking) | `(N, D)` | `(N, T, F)` |
| Batch (for training) | `(N, D)` | `(N, F, T)` ⚠️ |

---

### 3. Inference Guidance ✅

**File**: `hitl/inference/serve.py`

**Changes**:
- Added detailed TODO implementation guide in `predict_tensor()` docstring
- Documented the required transpose operation for single-sample inference
- Updated `_normalize()` documentation with shape expectations
- Included code snippets showing exact implementation needed

**Guided Implementation**:
```python
# TODO: When implementing predict_tensor()
if self._config["mode"] == "conv1d" and arr.ndim == 2:
    arr = arr.T  # (T, F) -> (F, T)
arr = arr[np.newaxis, ...]  # Add batch dimension
```

---

### 4. Model Documentation Updates ✅

**File**: `hitl/models/ae.py`

**Changes**:
- Enhanced `Conv1dAE` class docstring with shape convention section
- Clarified `__init__()` parameters and input format expectations
- Updated `forward()` method with explicit shape requirements
- Enhanced `build_model()` factory function with shape convention notes

**Key Clarifications**:
- Storage format: `(T, F)` - human-readable
- Model format: `(B, F, T)` - PyTorch requirement
- Transformation happens in data loading, not in model code

---

## Files Modified

1. ✅ `hitl/training/trainer.py`
   - Added `mode` parameter to `load_dataset()`
   - Implemented transpose for Conv1d
   - Updated `train_and_publish()` call

2. ✅ `docs/SHAPE_CONVENTIONS.md` (NEW)
   - Comprehensive shape convention documentation

3. ✅ `hitl/inference/serve.py`
   - Added TODO implementation guides
   - Updated docstrings with shape expectations

4. ✅ `hitl/models/ae.py`
   - Enhanced all Conv1d-related docstrings
   - Clarified shape expectations throughout

---

## Verification Steps

### For Training (Already Fixed):
1. ✅ Data loaded from DB as `(N, T, F)`
2. ✅ Transposed to `(N, F, T)` for Conv1d mode
3. ✅ Passed correctly to `Conv1dAE` model
4. ✅ Scaler computed over correct axes `(0, 2)`
5. ✅ Normalization broadcasts correctly

### For Inference (Documented, Not Yet Implemented):
1. ⏳ Single sample loaded as `(T, F)`
2. ⏳ Transpose to `(F, T)`
3. ⏳ Add batch dimension: `(1, F, T)`
4. ⏳ Normalize with proper broadcasting
5. ⏳ Forward pass through model
6. ⏳ Compute reconstruction error

---

## Testing Recommendations

### Unit Tests to Add:

```python
def test_conv1d_shape_transformation():
    """Test that Conv1d data is properly transposed."""
    trainer = Trainer(...)
    
    # Mock data in storage format
    vectors = [np.random.randn(240, 84) for _ in range(10)]
    # Expected: (10, 240, 84) after stacking
    # Expected: (10, 84, 240) after transpose
    
    X, ids = trainer.load_dataset(schema_id, mode="conv1d")
    
    assert X.shape == (10, 84, 240), f"Expected (10, 84, 240), got {X.shape}"

def test_conv1d_training_pipeline():
    """Integration test for full Conv1d pipeline."""
    # Create test data in storage format (T, F)
    # Train model
    # Verify model can process the data
    # Verify reconstruction shape matches input shape

def test_conv1d_inference_transpose():
    """Test inference with proper transpose (when implemented)."""
    # Load single sample: (T, F)
    # Predict
    # Verify correct transpose applied
    # Verify output is valid
```

---

## Impact Assessment

### ✅ Positive Impacts:
- **Critical bug fixed**: Conv1d mode will now work correctly
- **No API changes**: External interface remains the same
- **Better documentation**: Clear guidance for future developers
- **Improved code clarity**: Explicit shape handling is now visible

### ⚠️ No Breaking Changes:
- Storage format unchanged: Still `(T, F)`
- Database schema unchanged
- API contracts unchanged
- Only internal data transformation added

### 📋 Follow-up Work:
- Implement `LiveModel` class in `serve.py` following the TODO guides
- Add unit tests for shape transformations
- Add integration tests for Conv1d end-to-end pipeline
- Consider adding shape validation at API boundaries

---

## Related Documentation

- **Shape Conventions**: `docs/SHAPE_CONVENTIONS.md`
- **Architecture Spec**: `specs/architecture.md`
- **System Spec**: `specs/spec.md`

---

## Commit Message Suggestion

```
fix(conv1d): Add missing transpose for Conv1d data shape handling

Critical fix for Conv1d mode to properly transform data from storage 
format (T, F) to PyTorch Conv1d format (B, F, T).

Changes:
- Add transpose in trainer.py load_dataset() for Conv1d mode
- Create comprehensive shape convention documentation
- Add implementation guides in inference serve.py
- Update model docstrings for clarity

Fixes shape mismatch that would cause training failures in Conv1d mode.

See docs/SHAPE_CONVENTIONS.md for complete details.
```

---

## Sign-off

**Implementation**: ✅ Complete  
**Documentation**: ✅ Complete  
**Testing**: ⏳ Pending (guides provided)  
**Code Review**: ⏳ Ready for review

The Conv1d shape handling issue has been resolved with proper transpose operations 
and comprehensive documentation to prevent similar issues in the future.
