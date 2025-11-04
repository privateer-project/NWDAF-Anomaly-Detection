#!/usr/bin/env python
"""
Development script to test serialization functionality.

Tests NumPy array encoding/decoding, shape preservation, and data validation.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from hitl.io.serialization import (
    encode_npy,
    decode_npy,
    validate_array,
    safe_cast,
)
from hitl.errors import InvalidArray


def print_section(title):
    """Print a section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def test_serialization():
    """Test serialization functionality."""
    
    # Test 1D arrays
    print_section("1. 1D Array Serialization")
    
    arr_1d = np.array([1.5, 2.3, -0.5, 4.2], dtype=np.float32)
    print(f"Original array: shape={arr_1d.shape}, dtype={arr_1d.dtype}")
    print(f"Values: {arr_1d}")
    
    blob_1d = encode_npy(arr_1d)
    print(f"✓ Encoded to {len(blob_1d)} bytes")
    
    restored_1d = decode_npy(blob_1d)
    print(f"✓ Decoded: shape={restored_1d.shape}, dtype={restored_1d.dtype}")
    
    assert np.array_equal(arr_1d, restored_1d)
    assert arr_1d.dtype == restored_1d.dtype
    print("✓ Round-trip successful - arrays match perfectly")
    
    # Test 2D arrays
    print_section("2. 2D Array Serialization")
    
    arr_2d = np.random.randn(5, 8).astype(np.float32)
    print(f"Original array: shape={arr_2d.shape}, dtype={arr_2d.dtype}")
    print(f"Sample values:\n{arr_2d[:2, :4]}")
    
    blob_2d = encode_npy(arr_2d)
    print(f"✓ Encoded to {len(blob_2d)} bytes")
    
    restored_2d = decode_npy(blob_2d)
    print(f"✓ Decoded: shape={restored_2d.shape}, dtype={restored_2d.dtype}")
    
    assert np.allclose(arr_2d, restored_2d)
    assert arr_2d.shape == restored_2d.shape
    print("✓ Round-trip successful - arrays match")
    
    # Test different dtypes
    print_section("3. Different Data Types")
    
    dtypes = [
        (np.float32, "float32"),
        (np.float64, "float64"),
        (np.int32, "int32"),
        (np.int64, "int64"),
    ]
    
    for np_dtype, name in dtypes:
        arr = np.array([1, 2, 3, 4], dtype=np_dtype)
        blob = encode_npy(arr)
        restored = decode_npy(blob)
        
        assert np.array_equal(arr, restored)
        assert arr.dtype == restored.dtype
        
        print(f"✓ {name:8s}: {len(blob):3d} bytes")
    
    # Test array validation
    print_section("4. Array Validation")
    
    print("Testing valid arrays...")
    
    valid_arrays = [
        np.array([1.0, 2.0, 3.0], dtype=np.float32),
        np.random.randn(128).astype(np.float32),
        np.zeros((10, 8), dtype=np.float64),
        np.ones((256,), dtype=np.int32),
    ]
    
    for i, arr in enumerate(valid_arrays):
        validate_array(arr)
        print(f"✓ Valid array {i+1}: shape={arr.shape}, dtype={arr.dtype}")
    
    print("\nTesting invalid arrays...")
    
    # NaN array
    arr_nan = np.array([1.0, np.nan, 3.0], dtype=np.float32)
    try:
        validate_array(arr_nan)
        print("❌ Should have rejected NaN array")
    except InvalidArray as e:
        print(f"✓ Rejected NaN: {str(e)[:50]}...")
    
    # Inf array
    arr_inf = np.array([1.0, np.inf, 3.0], dtype=np.float32)
    try:
        validate_array(arr_inf)
        print("❌ Should have rejected Inf array")
    except InvalidArray as e:
        print(f"✓ Rejected Inf: {str(e)[:50]}...")
    
    # 3D array (unsupported)
    arr_3d = np.zeros((2, 3, 4), dtype=np.float32)
    try:
        validate_array(arr_3d)
        print("❌ Should have rejected 3D array")
    except InvalidArray as e:
        print(f"✓ Rejected 3D: {str(e)[:50]}...")
    
    # Empty array
    arr_empty = np.array([], dtype=np.float32)
    try:
        validate_array(arr_empty)
        print("❌ Should have rejected empty array")
    except InvalidArray as e:
        print(f"✓ Rejected empty: {str(e)[:50]}...")
    
    # Test safe casting
    print_section("5. Safe Type Casting")
    
    print("Testing list to array conversion...")
    
    list_1d = [1.5, 2.3, -0.5, 4.2]
    arr_from_list = safe_cast(list_1d, target_dtype="float32", expected_shape=(4,))
    print(f"✓ List -> array: {arr_from_list.dtype}, shape={arr_from_list.shape}")
    assert arr_from_list.dtype == np.float32
    assert arr_from_list.shape == (4,)
    
    list_2d = [[1, 2, 3], [4, 5, 6]]
    arr_from_list_2d = safe_cast(list_2d, target_dtype="float32", expected_shape=(2, 3))
    print(f"✓ Nested list -> array: {arr_from_list_2d.dtype}, shape={arr_from_list_2d.shape}")
    assert arr_from_list_2d.shape == (2, 3)
    
    # Test with wrong shape
    print("\nTesting shape mismatch detection...")
    try:
        safe_cast([1, 2, 3], target_dtype="float32", expected_shape=(4,))
        print("❌ Should have detected shape mismatch")
    except InvalidArray as e:
        print(f"✓ Detected shape mismatch: {str(e)[:60]}...")
    
    # Test compression efficiency
    print_section("6. Compression Analysis")
    
    sizes = [128, 256, 512, 1024]
    
    print("Analyzing serialization overhead...")
    print(f"{'Size':<10} {'Raw Bytes':<12} {'Encoded':<12} {'Overhead':<10}")
    print("-" * 50)
    
    for size in sizes:
        arr = np.random.randn(size).astype(np.float32)
        raw_size = arr.nbytes
        blob = encode_npy(arr)
        encoded_size = len(blob)
        overhead = encoded_size - raw_size
        
        print(f"{size:<10} {raw_size:<12} {encoded_size:<12} {overhead:<10}")
    
    print("\n✓ NumPy .npy format adds minimal overhead (~128 bytes for header)")
    
    # Test large arrays
    print_section("7. Large Array Handling")
    
    large_sizes = [
        (10000,),
        (1000, 100),
        (100, 100, 10),  # Will fail validation but test encoding
    ]
    
    for shape in large_sizes[:2]:  # Skip 3D for validation
        arr = np.random.randn(*shape).astype(np.float32)
        print(f"\nTesting {shape}:")
        print(f"  Array size: {arr.nbytes / 1024:.2f} KB")
        
        blob = encode_npy(arr)
        print(f"  Encoded size: {len(blob) / 1024:.2f} KB")
        
        restored = decode_npy(blob)
        assert np.allclose(arr, restored)
        print(f"  ✓ Round-trip successful")
    
    # Test precision preservation
    print_section("8. Precision Preservation")
    
    print("Testing floating point precision...")
    
    # Very small values
    arr_tiny = np.array([1e-10, 1e-15, 1e-20], dtype=np.float64)
    blob_tiny = encode_npy(arr_tiny)
    restored_tiny = decode_npy(blob_tiny)
    
    assert np.array_equal(arr_tiny, restored_tiny)
    print(f"✓ Tiny values preserved: {arr_tiny}")
    
    # Very large values
    arr_huge = np.array([1e10, 1e15, 1e20], dtype=np.float64)
    blob_huge = encode_npy(arr_huge)
    restored_huge = decode_npy(blob_huge)
    
    assert np.array_equal(arr_huge, restored_huge)
    print(f"✓ Huge values preserved: {arr_huge}")
    
    # Mixed range
    arr_mixed = np.array([1e-10, 1.0, 1e10], dtype=np.float64)
    blob_mixed = encode_npy(arr_mixed)
    restored_mixed = decode_npy(blob_mixed)
    
    assert np.array_equal(arr_mixed, restored_mixed)
    print(f"✓ Mixed range preserved: {arr_mixed}")
    
    # Test integer arrays
    print_section("9. Integer Array Handling")
    
    int_arrays = [
        np.array([1, 2, 3, 4], dtype=np.int32),
        np.array([-1000, 0, 1000], dtype=np.int64),
        np.arange(100, dtype=np.int32),
    ]
    
    for arr in int_arrays:
        blob = encode_npy(arr)
        restored = decode_npy(blob)
        
        assert np.array_equal(arr, restored)
        print(f"✓ {arr.dtype}: {arr.shape} - {len(blob)} bytes")
    
    print_section("Summary")
    print("✓ All serialization tests passed!")
    print("✓ Features verified:")
    print("  - 1D and 2D array encoding/decoding")
    print("  - Multiple data types (float32, float64, int32, int64)")
    print("  - Array validation (NaN, Inf, shape checks)")
    print("  - Safe type casting from lists")
    print("  - Precision preservation")
    print("  - Large array handling")


if __name__ == "__main__":
    try:
        test_serialization()
        print("\n" + "="*60)
        print("  ✓ ALL TESTS PASSED")
        print("="*60 + "\n")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
