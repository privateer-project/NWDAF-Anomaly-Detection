# Tests for Schema Registry
#
# Test coverage:
# - Schema creation and ID generation
# - Shape validation (1D and 2D)
# - Dtype validation
# - Schema retrieval and listing
# - Error cases (unsupported shapes, invalid dtypes)
#
# Test functions to implement:
#
# def test_ensure_creates_new_schema(registry):
#   """Test creating new schema for unique shape."""
#
# def test_ensure_returns_existing_schema(registry):
#   """Test idempotent behavior - same shape returns same schema_id."""
#
# def test_ensure_1d_shape_valid(registry):
#   """Test 1D shapes are accepted."""
#
# def test_ensure_2d_shape_valid(registry):
#   """Test 2D shapes are accepted."""
#
# def test_ensure_rejects_3d_shape(registry):
#   """Test 3D+ shapes raise UnsupportedShape."""
#
# def test_ensure_rejects_zero_dimension(registry):
#   """Test shapes with 0 values are rejected."""
#
# def test_validate_dtype_accepts_valid(registry):
#   """Test valid dtype strings are accepted."""
#
# def test_validate_dtype_rejects_invalid(registry):
#   """Test invalid dtype strings raise ValidationError."""
#
# def test_get_retrieves_schema_by_id(registry):
#   """Test schema retrieval by ID."""
#
# def test_get_raises_if_not_found(registry):
#   """Test SchemaNotFound for invalid ID."""
#
# def test_from_anomaly_returns_schema(registry, repository, sample_vector_1d):
#   """Test getting schema from anomaly's vector."""
#
# def test_list_all_returns_schemas(registry):
#   """Test listing all registered schemas."""
#
# def test_schema_id_deterministic():
#   """Test same shape+dtype produces same schema_id."""
#
# def test_shape_to_str_and_back():
#   """Test shape serialization round-trip."""
