# Tests for IO Serialization
#
# Test coverage:
# - .npy encoding and decoding
# - Shape validation and enforcement
# - Tensor validation (NaN, Inf, empty)
# - List to array conversion
# - Array to list conversion
# - Blob size estimation
#
# Test functions to implement:
#
# def test_encode_npy_creates_blob():
#   """Test encoding NumPy array to .npy bytes."""
#
# def test_decode_npy_returns_array():
#   """Test decoding .npy bytes to NumPy array."""
#
# def test_encode_decode_roundtrip_1d():
#   """Test 1D array round-trip preserves data."""
#
# def test_encode_decode_roundtrip_2d():
#   """Test 2D array round-trip preserves data."""
#
# def test_encode_casts_dtype():
#   """Test dtype conversion during encoding."""
#
# def test_ensure_shape_passes_matching():
#   """Test ensure_shape accepts matching shape."""
#
# def test_ensure_shape_raises_mismatch():
#   """Test ensure_shape raises on mismatch."""
#
# def test_validate_tensor_accepts_valid():
#   """Test valid tensors pass validation."""
#
# def test_validate_tensor_rejects_nan():
#   """Test NaN values are caught."""
#
# def test_validate_tensor_rejects_inf():
#   """Test Inf values are caught."""
#
# def test_validate_tensor_rejects_empty():
#   """Test empty arrays are caught."""
#
# def test_validate_tensor_rejects_3d():
#   """Test 3D+ arrays are rejected."""
#
# def test_tensor_from_list_1d():
#   """Test converting 1D list to array."""
#
# def test_tensor_from_list_2d():
#   """Test converting 2D list to array."""
#
# def test_tensor_from_list_ragged_raises():
#   """Test ragged arrays are rejected."""
#
# def test_tensor_to_list_1d():
#   """Test converting 1D array to list."""
#
# def test_tensor_to_list_2d():
#   """Test converting 2D array to list."""
#
# def test_estimate_blob_size():
#   """Test blob size estimation."""
