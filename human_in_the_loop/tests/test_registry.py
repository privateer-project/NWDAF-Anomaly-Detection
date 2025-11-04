"""Tests for Schema Registry."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from hitl.errors import SchemaNotFound, UnsupportedShape, ValidationError
from hitl.io.serialization import encode_npy
from hitl.schemas.registry import SchemaRegistry, shape_to_str, str_to_shape
from hitl.store.repository import Repository
from hitl.store.sqlite import SQLite
from hitl.utils.ids import schema_id
from hitl.utils.time import now_iso


@pytest.fixture
def temp_db():
    """Create temporary directory for test database."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def repository(temp_db):
    """Create repository with test database."""
    db_path = temp_db / "test.db"
    sqlite = SQLite(str(db_path))
    return Repository(sqlite)


@pytest.fixture
def registry(repository):
    """Create schema registry."""
    return SchemaRegistry(repository)


@pytest.fixture
def sample_vector_1d():
    """Create sample 1D vector."""
    arr = np.array([1.0, 2.0, 3.0], dtype="float32")
    return encode_npy(arr)


@pytest.fixture
def sample_vector_2d():
    """Create sample 2D vector."""
    arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype="float32")
    return encode_npy(arr)


class TestSchemaCreation:
    """Test schema creation and ID generation."""

    def test_ensure_creates_new_schema(self, registry):
        """Test creating new schema for unique shape."""
        shape = (128,)
        dtype = "float32"
        
        schema = registry.ensure(shape=shape, dtype=dtype)
        
        # schema_id is a SHA-1 hex digest (40 characters)
        assert len(schema["schema_id"]) == 40
        assert schema["shape"] == shape
        assert schema["ndim"] == 1
        assert schema["numel"] == 128
        assert schema["dtype"] == dtype

    def test_ensure_returns_existing_schema(self, registry):
        """Test idempotent behavior - same shape returns same schema_id."""
        shape = (100,)
        dtype = "float32"
        
        # Create first time
        schema1 = registry.ensure(shape=shape, dtype=dtype)
        
        # Create again - should return same
        schema2 = registry.ensure(shape=shape, dtype=dtype)
        
        assert schema1["schema_id"] == schema2["schema_id"]
        assert schema1 == schema2

    def test_ensure_1d_shape_valid(self, registry):
        """Test 1D shapes are accepted."""
        shapes = [(10,), (128,), (256,), (1024,)]
        
        for shape in shapes:
            schema = registry.ensure(shape=shape, dtype="float32")
            assert schema["ndim"] == 1
            assert schema["numel"] == shape[0]

    def test_ensure_2d_shape_valid(self, registry):
        """Test 2D shapes are accepted."""
        shapes = [(10, 8), (16, 16), (100, 128)]
        
        for shape in shapes:
            schema = registry.ensure(shape=shape, dtype="float32")
            assert schema["ndim"] == 2
            assert schema["numel"] == shape[0] * shape[1]

    def test_ensure_different_shapes_different_ids(self, registry):
        """Test different shapes produce different IDs."""
        schema1 = registry.ensure(shape=(128,), dtype="float32")
        schema2 = registry.ensure(shape=(256,), dtype="float32")
        
        assert schema1["schema_id"] != schema2["schema_id"]

    def test_ensure_different_dtypes_different_ids(self, registry):
        """Test different dtypes produce different IDs."""
        schema1 = registry.ensure(shape=(128,), dtype="float32")
        schema2 = registry.ensure(shape=(128,), dtype="float64")
        
        assert schema1["schema_id"] != schema2["schema_id"]


class TestShapeValidation:
    """Test shape validation."""

    def test_ensure_rejects_3d_shape(self, registry):
        """Test 3D+ shapes raise UnsupportedShape."""
        shape = (10, 8, 5)
        
        with pytest.raises(UnsupportedShape, match="must be 1D or 2D"):
            registry.ensure(shape=shape, dtype="float32")

    def test_ensure_rejects_zero_dimension(self, registry):
        """Test shapes with 0 values are rejected."""
        shapes = [(0,), (10, 0), (0, 8)]
        
        for shape in shapes:
            with pytest.raises(UnsupportedShape, match="positive"):
                registry.ensure(shape=shape, dtype="float32")

    def test_ensure_rejects_negative_dimension(self, registry):
        """Test shapes with negative values are rejected."""
        shape = (10, -5)
        
        with pytest.raises(UnsupportedShape, match="positive"):
            registry.ensure(shape=shape, dtype="float32")

    def test_ensure_rejects_empty_shape(self, registry):
        """Test empty shape (0D scalar) is rejected."""
        shape = ()
        
        with pytest.raises(UnsupportedShape, match="empty"):
            registry.ensure(shape=shape, dtype="float32")

    def test_validate_shape_provides_helpful_messages(self, registry):
        """Test error messages explain supported shapes."""
        shape = (2, 3, 4)
        
        with pytest.raises(UnsupportedShape) as exc_info:
            registry.validate_shape(shape)
        
        error_msg = str(exc_info.value)
        assert "dense" in error_msg or "conv1d" in error_msg


class TestDtypeValidation:
    """Test dtype validation."""

    def test_validate_dtype_accepts_valid(self, registry):
        """Test valid dtype strings are accepted."""
        valid_dtypes = ["float32", "float64", "int32", "int64"]
        
        for dtype in valid_dtypes:
            registry.validate_dtype(dtype)  # Should not raise

    def test_validate_dtype_rejects_invalid(self, registry):
        """Test invalid dtype strings raise ValidationError."""
        invalid_dtypes = ["float16", "uint8", "complex64", "notadtype"]
        
        for dtype in invalid_dtypes:
            with pytest.raises(ValidationError):
                registry.validate_dtype(dtype)

    def test_validate_dtype_rejects_unsupported(self, registry):
        """Test unsupported but valid NumPy dtypes raise ValidationError."""
        # These are valid NumPy dtypes but not supported by our system
        unsupported = ["bool", "int8", "uint32"]
        
        for dtype in unsupported:
            with pytest.raises(ValidationError, match="Unsupported"):
                registry.validate_dtype(dtype)


class TestSchemaRetrieval:
    """Test schema retrieval."""

    def test_get_retrieves_schema_by_id(self, registry):
        """Test schema retrieval by ID."""
        # Create schema
        shape = (64,)
        created = registry.ensure(shape=shape, dtype="float32")
        
        # Retrieve by ID
        retrieved = registry.get(created["schema_id"])
        
        assert retrieved == created

    def test_get_raises_if_not_found(self, registry):
        """Test SchemaNotFound for invalid ID."""
        with pytest.raises(SchemaNotFound, match="Schema not found"):
            registry.get("schema_nonexistent")

    def test_from_anomaly_returns_schema(self, registry, repository, sample_vector_1d):
        """Test getting schema from anomaly's vector."""
        # Create schema and anomaly
        shape = (3,)
        schema = registry.ensure(shape=shape, dtype="float32")
        
        anomaly_id = "A1"
        now = now_iso()
        repository.upsert_anomaly(
            anomaly_id=anomaly_id,
            occurred_at=now,
            source="test",
            schema_id=schema["schema_id"],
            created_at=now,
            updated_at=now,
        )
        
        # Store vector
        repository.put_vector(
            anomaly_id=anomaly_id,
            schema_id=schema["schema_id"],
            blob=sample_vector_1d,
            created_at=now,
        )
        
        # Retrieve schema from anomaly
        retrieved = registry.from_anomaly(anomaly_id)
        
        assert retrieved["schema_id"] == schema["schema_id"]
        assert retrieved["shape"] == shape

    def test_from_anomaly_raises_if_no_vector(self, registry, repository):
        """Test ValueError if anomaly has no vector."""
        # Create schema first
        shape = (64,)
        schema = registry.ensure(shape=shape, dtype="float32")
        
        # Create anomaly without vector
        anomaly_id = "A2"
        now = now_iso()
        repository.upsert_anomaly(
            anomaly_id=anomaly_id,
            occurred_at=now,
            source="test",
            schema_id=schema["schema_id"],
            created_at=now,
            updated_at=now,
        )
        
        with pytest.raises(ValueError, match="No vector stored"):
            registry.from_anomaly(anomaly_id)

    def test_list_all_returns_schemas(self, registry):
        """Test listing all registered schemas."""
        # Create multiple schemas
        shapes = [(128,), (256,), (10, 8)]
        created = []
        
        for shape in shapes:
            schema = registry.ensure(shape=shape, dtype="float32")
            created.append(schema)
        
        # List all
        all_schemas = registry.list_all()
        
        assert len(all_schemas) == len(shapes)
        
        # Check all created schemas are in list
        all_ids = {s["schema_id"] for s in all_schemas}
        created_ids = {s["schema_id"] for s in created}
        assert all_ids == created_ids

    def test_list_all_empty_when_no_schemas(self, registry):
        """Test listing returns empty list when no schemas exist."""
        schemas = registry.list_all()
        assert schemas == []


class TestSchemaMetadata:
    """Test schema metadata computation."""

    def test_compute_metadata_1d(self, registry):
        """Test metadata for 1D shape."""
        schema = registry.ensure(shape=(100,), dtype="float32")
        
        assert schema["ndim"] == 1
        assert schema["numel"] == 100

    def test_compute_metadata_2d(self, registry):
        """Test metadata for 2D shape."""
        schema = registry.ensure(shape=(10, 8), dtype="float32")
        
        assert schema["ndim"] == 2
        assert schema["numel"] == 80

    def test_compute_metadata_various_shapes(self, registry):
        """Test metadata computation for various shapes."""
        test_cases = [
            ((50,), 1, 50),
            ((16, 16), 2, 256),
            ((100, 128), 2, 12800),
        ]
        
        for shape, expected_ndim, expected_numel in test_cases:
            schema = registry.ensure(shape=shape, dtype="float32")
            assert schema["ndim"] == expected_ndim
            assert schema["numel"] == expected_numel


class TestShapeSerialization:
    """Test shape string conversion."""

    def test_shape_to_str_1d(self):
        """Test 1D shape to string."""
        shape = (128,)
        s = shape_to_str(shape)
        
        assert isinstance(s, str)
        assert "128" in s

    def test_shape_to_str_2d(self):
        """Test 2D shape to string."""
        shape = (10, 8)
        s = shape_to_str(shape)
        
        assert isinstance(s, str)
        assert "10" in s
        assert "8" in s

    def test_str_to_shape_json(self):
        """Test parsing JSON format."""
        s = "[128]"
        shape = str_to_shape(s)
        
        assert shape == (128,)

    def test_str_to_shape_json_2d(self):
        """Test parsing JSON format 2D."""
        s = "[10, 8]"
        shape = str_to_shape(s)
        
        assert shape == (10, 8)

    def test_str_to_shape_csv(self):
        """Test parsing comma-separated format."""
        s = "10,8"
        shape = str_to_shape(s)
        
        assert shape == (10, 8)

    def test_shape_to_str_and_back(self):
        """Test shape serialization round-trip."""
        shapes = [(128,), (10, 8), (256,), (100, 128)]
        
        for original_shape in shapes:
            s = shape_to_str(original_shape)
            restored_shape = str_to_shape(s)
            assert restored_shape == original_shape


class TestSchemaIdDeterminism:
    """Test schema ID generation."""

    def test_schema_id_deterministic(self):
        """Test same shape+dtype produces same schema_id."""
        shape = (128,)
        dtype = "float32"
        
        id1 = schema_id(shape, dtype)
        id2 = schema_id(shape, dtype)
        
        assert id1 == id2

    def test_schema_id_different_for_different_shapes(self):
        """Test different shapes produce different IDs."""
        id1 = schema_id((128,), "float32")
        id2 = schema_id((256,), "float32")
        
        assert id1 != id2

    def test_schema_id_different_for_different_dtypes(self):
        """Test different dtypes produce different IDs."""
        id1 = schema_id((128,), "float32")
        id2 = schema_id((128,), "float64")
        
        assert id1 != id2

