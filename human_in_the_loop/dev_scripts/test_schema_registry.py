#!/usr/bin/env python
"""
Development script to test schema registry functionality.

Tests schema registration, validation, and retrieval for different tensor shapes.
"""

import sys
import tempfile
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from hitl.schemas.registry import SchemaRegistry, shape_to_str, str_to_shape
from hitl.store.sqlite import SQLite
from hitl.store.repository import Repository
from hitl.io.serialization import encode_npy
from hitl.utils.time import now_iso
from hitl.errors import UnsupportedShape


def print_section(title):
    """Print a section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def test_schema_registry():
    """Test schema registry functionality."""

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"

        print_section("1. Schema Registry Initialization")
        print(f"Creating database at: {db_path}")

        sqlite = SQLite(str(db_path))
        repo = Repository(sqlite)
        registry = SchemaRegistry(repo)

        print("✓ Schema registry initialized")

        # Test 1D schema (dense mode)
        print_section("2. 1D Schema Registration (Dense Mode)")

        shape_1d = (128,)
        print(f"Registering 1D schema: {shape_1d}")

        schema_1d = registry.ensure(shape=shape_1d, dtype="float32")
        print(f"✓ Schema created: {schema_1d['schema_id'][:16]}...")
        print(f"  - Shape: {schema_1d['shape']}")
        print(f"  - Ndim: {schema_1d['ndim']}")
        print(f"  - Numel: {schema_1d['numel']}")
        print(f"  - Dtype: {schema_1d['dtype']}")

        # Test idempotency
        schema_1d_again = registry.ensure(shape=shape_1d, dtype="float32")
        assert schema_1d["schema_id"] == schema_1d_again["schema_id"]
        print("✓ Idempotency verified - same schema_id returned")

        # Test 2D schema (conv1d mode)
        print_section("3. 2D Schema Registration (Conv1d Mode)")

        shape_2d = (10, 8)
        print(f"Registering 2D schema: {shape_2d}")

        schema_2d = registry.ensure(shape=shape_2d, dtype="float32")
        print(f"✓ Schema created: {schema_2d['schema_id'][:16]}...")
        print(f"  - Shape: {schema_2d['shape']}")
        print(f"  - Ndim: {schema_2d['ndim']}")
        print(f"  - Numel: {schema_2d['numel']}")
        print(f"  - Dtype: {schema_2d['dtype']}")

        # Verify different schemas have different IDs
        assert schema_1d["schema_id"] != schema_2d["schema_id"]
        print("✓ Different shapes produce different schema IDs")

        # Test schema retrieval
        print_section("4. Schema Retrieval")

        retrieved = registry.get(schema_1d["schema_id"])
        print(f"✓ Retrieved schema by ID: {retrieved['shape']}")
        assert retrieved == schema_1d

        # Test listing all schemas
        all_schemas = registry.list_all()
        print(f"✓ Listed all schemas: {len(all_schemas)} total")
        for s in all_schemas:
            print(f"  - {s['shape']} ({s['dtype']}) = {s['schema_id'][:16]}...")

        # Test shape validation
        print_section("5. Shape Validation")

        print("Testing valid shapes...")
        valid_shapes = [
            (64,),
            (256,),
            (16, 16),
            (100, 128),
        ]

        for shape in valid_shapes:
            registry.validate_shape(shape)
            print(f"✓ Valid: {shape}")

        print("\nTesting invalid shapes...")
        invalid_shapes = [
            (2, 3, 4),  # 3D
            (0,),  # Zero dimension
            (),  # Empty
            (10, 0),  # Zero in 2D
        ]

        for shape in invalid_shapes:
            try:
                registry.validate_shape(shape)
                print(f"❌ Should have rejected: {shape}")
            except UnsupportedShape as e:
                print(f"✓ Rejected: {shape} - {str(e)[:50]}...")

        # Test dtype validation
        print_section("6. Dtype Validation")

        print("Testing valid dtypes...")
        valid_dtypes = ["float32", "float64", "int32", "int64"]

        for dtype in valid_dtypes:
            registry.validate_dtype(dtype)
            print(f"✓ Valid: {dtype}")

        print("\nTesting invalid dtypes...")
        invalid_dtypes = ["float16", "uint8", "notadtype"]

        for dtype in invalid_dtypes:
            try:
                registry.validate_dtype(dtype)
                print(f"❌ Should have rejected: {dtype}")
            except Exception:
                print(f"✓ Rejected: {dtype}")

        # Test schema from anomaly
        print_section("7. Schema from Anomaly")

        # Create anomaly with vector
        anomaly_id = "test-anomaly-1"
        arr = np.random.randn(128).astype("float32")
        blob = encode_npy(arr)

        now = now_iso()
        repo.upsert_anomaly(
            anomaly_id=anomaly_id,
            occurred_at=now,
            source="test",
            schema_id=schema_1d["schema_id"],
            created_at=now,
            updated_at=now,
        )

        repo.put_vector(
            anomaly_id=anomaly_id,
            schema_id=schema_1d["schema_id"],
            blob=blob,
            created_at=now,
        )

        print(f"Created anomaly with vector: {anomaly_id}")

        # Get schema from anomaly
        schema_from_anomaly = registry.from_anomaly(anomaly_id)
        print(f"✓ Retrieved schema from anomaly: {schema_from_anomaly['shape']}")
        assert schema_from_anomaly["schema_id"] == schema_1d["schema_id"]

        # Test error case: anomaly without vector
        anomaly_id_no_vector = "test-anomaly-no-vector"
        repo.upsert_anomaly(
            anomaly_id=anomaly_id_no_vector,
            occurred_at=now,
            source="test",
            schema_id=schema_1d["schema_id"],
            created_at=now,
            updated_at=now,
        )

        try:
            registry.from_anomaly(anomaly_id_no_vector)
            print("❌ Should have raised ValueError")
        except ValueError:
            print("✓ Correctly raised error for missing vector")

        # Test shape serialization
        print_section("8. Shape Serialization")

        test_shapes = [
            (128,),
            (10, 8),
            (256,),
            (100, 128),
        ]

        print("Testing shape round-trip serialization...")
        for shape in test_shapes:
            shape_str = shape_to_str(shape)
            restored = str_to_shape(shape_str)
            assert restored == shape
            print(f"✓ {shape} -> '{shape_str}' -> {restored}")

        # Test deterministic schema IDs
        print_section("9. Schema ID Determinism")

        print("Creating multiple registries...")
        # Create new registry with same database
        registry2 = SchemaRegistry(repo)

        # Ensure same schema
        schema_test = registry2.ensure(shape=(128,), dtype="float32")

        print("✓ Schema ID is deterministic:")
        print(f"  First:  {schema_1d['schema_id']}")
        print(f"  Second: {schema_test['schema_id']}")
        assert schema_1d["schema_id"] == schema_test["schema_id"]

        print_section("Summary")
        print("✓ All schema registry tests passed!")
        print(f"✓ Total schemas registered: {len(registry.list_all())}")
        print("✓ Schemas tested:")
        for s in registry.list_all():
            print(f"  - {s['shape']} ({s['dtype']})")


if __name__ == "__main__":
    try:
        test_schema_registry()
        print("\n" + "=" * 60)
        print("  ✓ ALL TESTS PASSED")
        print("=" * 60 + "\n")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
