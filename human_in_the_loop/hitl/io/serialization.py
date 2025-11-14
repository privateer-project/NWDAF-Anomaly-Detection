"""
Tensor Serialization to/from .npy Format

This module handles conversion between NumPy arrays and .npy format bytes
for BLOB storage in SQLite, with shape and dtype validation.
"""

from __future__ import annotations

from io import BytesIO

import numpy as np

from hitl.errors import ShapeMismatch, UnsupportedShape

__all__ = [
    "encode_npy",
    "decode_npy",
    "ensure_shape",
    "validate_tensor",
    "tensor_from_list",
    "tensor_to_list",
    "estimate_blob_size",
    "validate_array",
    "safe_cast",
]


def encode_npy(arr: np.ndarray, dtype: str = "float32") -> bytes:
    """
    Encode NumPy array to .npy format bytes for database storage.

    Args:
        arr: NumPy array (1D or 2D)
        dtype: Target dtype string (e.g., "float32")

    Returns:
        .npy format bytes

    Raises:
        UnsupportedShape: if not 1D or 2D
        ValueError: if dtype conversion fails

    Example:
        >>> arr = np.array([1.0, 2.0, 3.0])
        >>> blob = encode_npy(arr, dtype="float32")
    """
    # Ensure contiguous memory layout
    if not arr.flags.c_contiguous:
        arr = np.ascontiguousarray(arr)

    # Cast to target dtype
    arr = arr.astype(dtype)

    # Validate shape is 1D or 2D
    if arr.ndim not in (1, 2):
        raise UnsupportedShape(
            f"Array must be 1D or 2D, got shape {arr.shape} (ndim={arr.ndim})"
        )

    # Serialize to .npy format
    buffer = BytesIO()
    np.save(buffer, arr)
    return buffer.getvalue()


def decode_npy(blob: bytes) -> np.ndarray:
    """
    Decode .npy format bytes back to NumPy array.

    Args:
        blob: .npy format bytes from database

    Returns:
        NumPy array with original shape and dtype

    Raises:
        ValueError: if blob is not valid .npy format
        TypeError: if result is not ndarray

    Example:
        >>> arr = decode_npy(blob)
    """
    buffer = BytesIO(blob)

    # Load with allow_pickle=False for security
    result = np.load(buffer, allow_pickle=False)

    # Validate result is ndarray
    if not isinstance(result, np.ndarray):
        raise TypeError(f"Expected ndarray, got {type(result).__name__}")

    return result


def ensure_shape(arr: np.ndarray, expected_shape: tuple[int, ...]) -> np.ndarray:
    """
    Validate array has expected shape.

    Args:
        arr: NumPy array to check
        expected_shape: Required shape tuple

    Returns:
        arr (unchanged) if shape matches

    Raises:
        ShapeMismatch: if arr.shape != expected_shape

    Example:
        >>> arr = decode_npy(blob)
        >>> arr = ensure_shape(arr, expected_shape=(128,))
    """
    if arr.shape != expected_shape:
        raise ShapeMismatch(
            f"Shape mismatch: expected {expected_shape}, got {arr.shape}. "
            "This tensor may not match the registered schema."
        )
    return arr


def validate_tensor(arr: np.ndarray) -> None:
    """
    Validate tensor for HITL system requirements.

    Args:
        arr: NumPy array to validate

    Raises:
        UnsupportedShape: if not 1D or 2D
        ValueError: if contains NaN, Inf, or is empty

    Checks:
        - Shape is 1D (D,) or 2D (T, F)
        - No NaN values
        - No Inf values
        - Non-empty (size > 0)
    """
    # Check shape
    if arr.ndim not in (1, 2):
        raise UnsupportedShape(
            f"Tensor must be 1D or 2D, got shape {arr.shape} (ndim={arr.ndim})"
        )

    # Check non-empty
    if arr.size == 0:
        raise ValueError("Tensor cannot be empty (size=0)")

    # Check for NaN
    if np.any(np.isnan(arr)):
        raise ValueError("Tensor contains NaN values")

    # Check for Inf
    if np.any(np.isinf(arr)):
        raise ValueError("Tensor contains Inf values")


def tensor_from_list(
    data: list[float] | list[list[float]], dtype: str = "float32"
) -> np.ndarray:
    """
    Convert Python list to NumPy array for API inputs.

    Args:
        data: Nested list (1D or 2D)
        dtype: Target NumPy dtype

    Returns:
        NumPy array with validated shape

    Raises:
        UnsupportedShape: if 3D+ or scalar
        ValueError: if ragged arrays or invalid data

    Example:
        >>> arr = tensor_from_list([1.0, 2.0, 3.0])
        >>> arr.shape
        (3,)
    """
    # Convert to NumPy array
    arr = np.array(data, dtype=dtype)

    # Validate with tensor requirements
    validate_tensor(arr)

    return arr


def validate_array(arr: np.ndarray) -> None:
    """Backward-compatible wrapper for array validation used in dev scripts.

    Raises `InvalidArray` (alias of ValidationError) on failure.
    """
    # Reuse validate_tensor behavior
    from hitl.errors import InvalidArray

    try:
        validate_tensor(arr)
    except Exception as e:
        # Wrap in InvalidArray for older tests
        raise InvalidArray(str(e)) from e


def safe_cast(data: list | np.ndarray, target_dtype: str = "float32", expected_shape: tuple[int, ...] | None = None) -> np.ndarray:
    """Convert lists to NumPy arrays and validate shape/dtype.

    Args:
        data: Python list (1D or 2D) or ndarray
        target_dtype: dtype string
        expected_shape: optional shape to assert

    Returns:
        NumPy ndarray with target dtype

    Raises:
        InvalidArray if validation fails
    """
    from hitl.errors import InvalidArray

    try:
        if isinstance(data, np.ndarray):
            arr = data.astype(target_dtype)
        else:
            arr = tensor_from_list(data, dtype=target_dtype)

        if expected_shape is not None and tuple(arr.shape) != tuple(expected_shape):
            raise InvalidArray(f"Expected shape {expected_shape}, got {arr.shape}")

        return arr
    except Exception as e:
        if isinstance(e, InvalidArray):
            raise
        raise InvalidArray(str(e)) from e


def tensor_to_list(arr: np.ndarray) -> list[float] | list[list[float]]:
    """
    Convert NumPy array to Python list for API outputs.

    Args:
        arr: NumPy array (1D or 2D)

    Returns:
        Nested Python list

    Example:
        >>> arr = np.array([1.0, 2.0])
        >>> lst = tensor_to_list(arr)
        >>> lst
        [1.0, 2.0]
    """
    return arr.tolist()


def estimate_blob_size(shape: tuple[int, ...], dtype: str = "float32") -> int:
    """
    Estimate .npy blob size in bytes for capacity planning.

    Args:
        shape: Tensor dimensions
        dtype: NumPy dtype string

    Returns:
        Approximate bytes (header + data)

    Example:
        >>> size = estimate_blob_size((128,), "float32")
        >>> size
        640
    """
    # Compute number of elements
    numel = 1
    for dim in shape:
        numel *= dim

    # Get itemsize for dtype
    itemsize = np.dtype(dtype).itemsize

    # Data size
    data_size = numel * itemsize

    # .npy header overhead (~128 bytes)
    header_size = 128

    return data_size + header_size
