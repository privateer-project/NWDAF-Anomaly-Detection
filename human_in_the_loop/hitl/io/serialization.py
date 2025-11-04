# Tensor Serialization to/from .npy Format
#
# This module handles conversion between NumPy arrays and .npy format bytes
# for BLOB storage in SQLite, with shape and dtype validation.
#
# Functions to implement:
#
# - def encode_npy(arr: np.ndarray, dtype: str = "float32") -> bytes:
#   """
#   Encode NumPy array to .npy format bytes for database storage.
#   
#   Args:
#     - arr: NumPy array (1D or 2D)
#     - dtype: Target dtype string (e.g., "float32")
#   
#   Returns: .npy format bytes
#   
#   Algorithm:
#     1. Validate array is contiguous in memory
#        If not: arr = np.ascontiguousarray(arr)
#     2. Cast to target dtype: arr = arr.astype(dtype)
#     3. Validate shape is 1D or 2D
#     4. Use BytesIO + np.save() to get .npy bytes
#     5. Return bytes
#   
#   Raises:
#     - UnsupportedShape: if not 1D or 2D
#     - ValueError: if dtype conversion fails
#   
#   .npy format preserves:
#     - Shape
#     - Dtype
#     - Byte order
#     - Array data
#   
#   Example:
#     arr = np.array([1.0, 2.0, 3.0])
#     blob = encode_npy(arr, dtype="float32")
#     # blob is bytes in .npy format
#   """
#
# - def decode_npy(blob: bytes) -> np.ndarray:
#   """
#   Decode .npy format bytes back to NumPy array.
#   
#   Args:
#     - blob: .npy format bytes from database
#   
#   Returns: NumPy array with original shape and dtype
#   
#   Algorithm:
#     1. Wrap blob in BytesIO
#     2. Use np.load() with allow_pickle=False for security
#     3. Validate result is ndarray
#     4. Return array
#   
#   Raises:
#     - ValueError: if blob is not valid .npy format
#     - TypeError: if result is not ndarray
#   
#   Example:
#     arr = decode_npy(blob)
#     # arr is NumPy array with preserved shape/dtype
#   """
#
# - def ensure_shape(arr: np.ndarray, expected_shape: tuple[int, ...]) -> np.ndarray:
#   """
#   Validate array has expected shape.
#   
#   Args:
#     - arr: NumPy array to check
#     - expected_shape: Required shape tuple
#   
#   Returns: arr (unchanged) if shape matches
#   
#   Raises:
#     - ShapeMismatch: if arr.shape != expected_shape
#   
#   Error should include:
#     - Expected shape
#     - Actual shape
#     - Helpful message about schema mismatch
#   
#   Example:
#     arr = decode_npy(blob)
#     arr = ensure_shape(arr, expected_shape=(128,))
#   """
#
# - def validate_tensor(arr: np.ndarray) -> None:
#   """
#   Validate tensor for HITL system requirements.
#   
#   Args:
#     - arr: NumPy array to validate
#   
#   Raises:
#     - UnsupportedShape: if not 1D or 2D
#     - ValueError: if contains NaN or Inf
#     - ValueError: if empty (0 elements)
#   
#   Checks:
#     - Shape is 1D (D,) or 2D (T, F)
#     - No NaN values: not np.any(np.isnan(arr))
#     - No Inf values: not np.any(np.isinf(arr))
#     - Non-empty: arr.size > 0
#   
#   Used before encoding to catch bad data early.
#   """
#
# - def tensor_from_list(
#     data: list[float] | list[list[float]],
#     dtype: str = "float32"
#   ) -> np.ndarray:
#   """
#   Convert Python list to NumPy array for API inputs.
#   
#   Args:
#     - data: Nested list (1D or 2D)
#     - dtype: Target NumPy dtype
#   
#   Returns: NumPy array with validated shape
#   
#   Algorithm:
#     1. Convert to NumPy: arr = np.array(data, dtype=dtype)
#     2. Validate with validate_tensor()
#     3. Return array
#   
#   Handles:
#     - 1D: [1.0, 2.0, 3.0] → shape (3,)
#     - 2D: [[1, 2], [3, 4]] → shape (2, 2)
#   
#   Raises:
#     - UnsupportedShape: if 3D+ or scalar
#     - ValueError: if ragged arrays (inconsistent inner lengths)
#   
#   Example:
#     arr = tensor_from_list([1.0, 2.0, 3.0])
#   """
#
# - def tensor_to_list(arr: np.ndarray) -> list[float] | list[list[float]]:
#   """
#   Convert NumPy array to Python list for API outputs.
#   
#   Args:
#     - arr: NumPy array (1D or 2D)
#   
#   Returns: Nested Python list
#   
#   Uses arr.tolist() which preserves structure
#   
#   Example:
#     arr = np.array([1.0, 2.0])
#     lst = tensor_to_list(arr)  # [1.0, 2.0]
#   """
#
# - def estimate_blob_size(shape: tuple[int, ...], dtype: str = "float32") -> int:
#   """
#   Estimate .npy blob size in bytes for capacity planning.
#   
#   Args:
#     - shape: Tensor dimensions
#     - dtype: NumPy dtype string
#   
#   Returns: Approximate bytes (header + data)
#   
#   Algorithm:
#     - Compute numel = product of shape
#     - Get itemsize for dtype (e.g., float32 = 4 bytes)
#     - Data size = numel * itemsize
#     - .npy header ~= 128 bytes (format overhead)
#     - Total = data_size + 128
#   
#   Example:
#     size = estimate_blob_size((128,), "float32")
#     # = 128 * 4 + 128 = 640 bytes
#   """
#
# Usage patterns:
#   # Store tensor
#   arr = np.random.randn(128).astype("float32")
#   blob = encode_npy(arr, dtype="float32")
#   repo.put_vector(anomaly_id, schema_id, blob, now_iso())
#   
#   # Load tensor
#   schema_id, blob = repo.get_vector(anomaly_id)
#   arr = decode_npy(blob)
#   schema = registry.get(schema_id)
#   arr = ensure_shape(arr, expected_shape=schema["shape"])
#   
#   # From API
#   tensor_data = [1.0, 2.0, 3.0]
#   arr = tensor_from_list(tensor_data, dtype="float32")
#   blob = encode_npy(arr)
