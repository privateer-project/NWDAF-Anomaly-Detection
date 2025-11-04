"""Tests for IO Serialization."""

from __future__ import annotations

import numpy as np
import pytest

from hitl.errors import ShapeMismatch, UnsupportedShape
from hitl.io.serialization import (
    decode_npy,
    encode_npy,
    ensure_shape,
    estimate_blob_size,
    tensor_from_list,
    tensor_to_list,
    validate_tensor,
)


class TestEncodeDecodeNpy:
    """Test .npy encoding and decoding."""

    def test_encode_npy_creates_blob(self):
        """Test encoding NumPy array to .npy bytes."""
        arr = np.array([1.0, 2.0, 3.0], dtype="float32")
        blob = encode_npy(arr, dtype="float32")
        
        assert isinstance(blob, bytes)
        assert len(blob) > 0
        # .npy format starts with magic string
        assert blob[:6] == b"\x93NUMPY"

    def test_decode_npy_returns_array(self):
        """Test decoding .npy bytes to NumPy array."""
        arr = np.array([1.0, 2.0, 3.0], dtype="float32")
        blob = encode_npy(arr)
        
        result = decode_npy(blob)
        
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        np.testing.assert_array_equal(result, arr)

    def test_encode_decode_roundtrip_1d(self):
        """Test 1D array round-trip preserves data."""
        original = np.array([1.5, 2.7, 3.9, 4.1], dtype="float32")
        blob = encode_npy(original)
        restored = decode_npy(blob)
        
        np.testing.assert_array_almost_equal(restored, original, decimal=6)
        assert restored.shape == original.shape
        assert restored.dtype == original.dtype

    def test_encode_decode_roundtrip_2d(self):
        """Test 2D array round-trip preserves data."""
        original = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype="float32")
        blob = encode_npy(original)
        restored = decode_npy(blob)
        
        np.testing.assert_array_almost_equal(restored, original, decimal=6)
        assert restored.shape == original.shape
        assert restored.dtype == original.dtype

    def test_encode_casts_dtype(self):
        """Test dtype conversion during encoding."""
        # Start with float64
        arr = np.array([1.0, 2.0, 3.0], dtype="float64")
        
        # Encode as float32
        blob = encode_npy(arr, dtype="float32")
        restored = decode_npy(blob)
        
        assert restored.dtype == np.float32

    def test_encode_handles_non_contiguous(self):
        """Test encoding handles non-contiguous arrays."""
        # Create non-contiguous array (transpose)
        arr = np.array([[1, 2, 3], [4, 5, 6]], dtype="float32").T
        assert not arr.flags.c_contiguous
        
        # Should still encode successfully
        blob = encode_npy(arr)
        restored = decode_npy(blob)
        
        np.testing.assert_array_equal(restored, arr)

    def test_encode_rejects_3d(self):
        """Test 3D arrays are rejected."""
        arr = np.ones((2, 3, 4), dtype="float32")
        
        with pytest.raises(UnsupportedShape, match="must be 1D or 2D"):
            encode_npy(arr)

    def test_decode_invalid_blob_raises(self):
        """Test decoding invalid blob raises error."""
        invalid_blob = b"not a valid npy format"
        
        with pytest.raises((ValueError, OSError)):
            decode_npy(invalid_blob)


class TestEnsureShape:
    """Test shape validation and enforcement."""

    def test_ensure_shape_passes_matching(self):
        """Test ensure_shape accepts matching shape."""
        arr = np.array([1.0, 2.0, 3.0], dtype="float32")
        result = ensure_shape(arr, expected_shape=(3,))
        
        assert result is arr  # Should return same object
        np.testing.assert_array_equal(result, arr)

    def test_ensure_shape_raises_mismatch(self):
        """Test ensure_shape raises on mismatch."""
        arr = np.array([1.0, 2.0, 3.0], dtype="float32")
        
        with pytest.raises(ShapeMismatch, match="expected.*got"):
            ensure_shape(arr, expected_shape=(4,))

    def test_ensure_shape_2d(self):
        """Test ensure_shape with 2D arrays."""
        arr = np.ones((10, 8), dtype="float32")
        result = ensure_shape(arr, expected_shape=(10, 8))
        
        assert result is arr

    def test_ensure_shape_mismatch_provides_details(self):
        """Test error message provides shape details."""
        arr = np.ones((10, 8), dtype="float32")
        
        with pytest.raises(ShapeMismatch) as exc_info:
            ensure_shape(arr, expected_shape=(10, 4))
        
        error_msg = str(exc_info.value)
        assert "(10, 8)" in error_msg
        assert "(10, 4)" in error_msg


class TestValidateTensor:
    """Test tensor validation."""

    def test_validate_tensor_accepts_valid_1d(self):
        """Test valid 1D tensors pass validation."""
        arr = np.array([1.0, 2.0, 3.0], dtype="float32")
        validate_tensor(arr)  # Should not raise

    def test_validate_tensor_accepts_valid_2d(self):
        """Test valid 2D tensors pass validation."""
        arr = np.ones((10, 8), dtype="float32")
        validate_tensor(arr)  # Should not raise

    def test_validate_tensor_rejects_nan(self):
        """Test NaN values are caught."""
        arr = np.array([1.0, np.nan, 3.0], dtype="float32")
        
        with pytest.raises(ValueError, match="NaN"):
            validate_tensor(arr)

    def test_validate_tensor_rejects_inf(self):
        """Test Inf values are caught."""
        arr = np.array([1.0, np.inf, 3.0], dtype="float32")
        
        with pytest.raises(ValueError, match="Inf"):
            validate_tensor(arr)

    def test_validate_tensor_rejects_negative_inf(self):
        """Test -Inf values are caught."""
        arr = np.array([1.0, -np.inf, 3.0], dtype="float32")
        
        with pytest.raises(ValueError, match="Inf"):
            validate_tensor(arr)

    def test_validate_tensor_rejects_empty(self):
        """Test empty arrays are caught."""
        arr = np.array([], dtype="float32")
        
        with pytest.raises(ValueError, match="empty"):
            validate_tensor(arr)

    def test_validate_tensor_rejects_3d(self):
        """Test 3D+ arrays are rejected."""
        arr = np.ones((2, 3, 4), dtype="float32")
        
        with pytest.raises(UnsupportedShape, match="must be 1D or 2D"):
            validate_tensor(arr)

    def test_validate_tensor_rejects_0d(self):
        """Test 0D scalars are rejected."""
        arr = np.array(42.0, dtype="float32")
        
        with pytest.raises(UnsupportedShape, match="must be 1D or 2D"):
            validate_tensor(arr)


class TestTensorFromList:
    """Test list to array conversion."""

    def test_tensor_from_list_1d(self):
        """Test converting 1D list to array."""
        data = [1.0, 2.0, 3.0]
        arr = tensor_from_list(data, dtype="float32")
        
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (3,)
        assert arr.dtype == np.float32
        np.testing.assert_array_equal(arr, [1.0, 2.0, 3.0])

    def test_tensor_from_list_2d(self):
        """Test converting 2D list to array."""
        data = [[1.0, 2.0], [3.0, 4.0]]
        arr = tensor_from_list(data, dtype="float32")
        
        assert arr.shape == (2, 2)
        assert arr.dtype == np.float32
        np.testing.assert_array_equal(arr, [[1.0, 2.0], [3.0, 4.0]])

    def test_tensor_from_list_casts_dtype(self):
        """Test dtype conversion from integers."""
        data = [1, 2, 3]
        arr = tensor_from_list(data, dtype="float32")
        
        assert arr.dtype == np.float32
        np.testing.assert_array_equal(arr, [1.0, 2.0, 3.0])

    def test_tensor_from_list_ragged_raises(self):
        """Test ragged arrays are rejected."""
        data = [[1.0, 2.0], [3.0]]  # Inconsistent inner lengths
        
        # NumPy will create object array, which fails validation
        with pytest.raises((ValueError, UnsupportedShape)):
            tensor_from_list(data, dtype="float32")

    def test_tensor_from_list_validates(self):
        """Test validation is applied to result."""
        # Data with NaN should fail validation
        data = [1.0, float("nan"), 3.0]
        
        with pytest.raises(ValueError, match="NaN"):
            tensor_from_list(data, dtype="float32")


class TestTensorToList:
    """Test array to list conversion."""

    def test_tensor_to_list_1d(self):
        """Test converting 1D array to list."""
        arr = np.array([1.0, 2.0, 3.0], dtype="float32")
        lst = tensor_to_list(arr)
        
        assert isinstance(lst, list)
        assert lst == [1.0, 2.0, 3.0]

    def test_tensor_to_list_2d(self):
        """Test converting 2D array to list."""
        arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype="float32")
        lst = tensor_to_list(arr)
        
        assert isinstance(lst, list)
        assert isinstance(lst[0], list)
        assert lst == [[1.0, 2.0], [3.0, 4.0]]

    def test_tensor_to_list_preserves_values(self):
        """Test values are preserved in conversion."""
        arr = np.array([1.5, 2.7, 3.9], dtype="float32")
        lst = tensor_to_list(arr)
        
        for i, val in enumerate(lst):
            assert abs(val - arr[i]) < 1e-6


class TestEstimateBlobSize:
    """Test blob size estimation."""

    def test_estimate_blob_size_1d(self):
        """Test size estimation for 1D array."""
        size = estimate_blob_size((128,), dtype="float32")
        
        # 128 elements * 4 bytes + 128 header = 640
        assert size == 640

    def test_estimate_blob_size_2d(self):
        """Test size estimation for 2D array."""
        size = estimate_blob_size((10, 8), dtype="float32")
        
        # 10 * 8 = 80 elements * 4 bytes + 128 header = 448
        assert size == 448

    def test_estimate_blob_size_float64(self):
        """Test size estimation with different dtype."""
        size = estimate_blob_size((100,), dtype="float64")
        
        # 100 elements * 8 bytes + 128 header = 928
        assert size == 928

    def test_estimate_blob_size_matches_actual(self):
        """Test estimation is close to actual size."""
        shape = (50,)
        dtype = "float32"
        
        estimated = estimate_blob_size(shape, dtype=dtype)
        
        # Create actual blob
        arr = np.zeros(shape, dtype=dtype)
        blob = encode_npy(arr, dtype=dtype)
        actual = len(blob)
        
        # Should be close (within 200 bytes for header variance)
        assert abs(estimated - actual) < 200

