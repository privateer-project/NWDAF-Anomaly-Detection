# Feature Schema Registry
#
# This module manages feature vector schemas, ensuring consistent
# tensor shapes and types throughout the system.
#
# Class to implement:
#
# class SchemaRegistry:
#   """Registry for feature tensor schemas with validation."""
#   
#   def __init__(self, repository: Repository):
#     """
#     Initialize registry with repository for persistence.
#     Args:
#       - repository: Repository instance for database access
#     """
#     self.repo = repository
#   
#   def ensure(self, shape: tuple[int, ...], dtype: str = "float32") -> SchemaInfo:
#     """
#     Ensure schema exists, creating if needed.
#     
#     This is the main entry point for schema registration.
#     
#     Args:
#       - shape: Tensor dimensions as tuple (e.g., (128,) or (10, 8))
#       - dtype: NumPy dtype string (default: "float32")
#     
#     Returns: SchemaInfo dict with schema_id and metadata
#     
#     Algorithm:
#       1. Validate shape is supported (1D or 2D only for MVP)
#       2. Compute schema_id from shape + dtype using ids.schema_id()
#       3. Check if schema_id exists in database
#       4. If not, insert new schema record
#       5. Return SchemaInfo dict
#     
#     Raises:
#       - UnsupportedShape: if shape not (D,) or (T,F)
#       - ValidationError: if dtype invalid
#       - DBError: on database errors
#     
#     Example:
#       schema = registry.ensure(shape=(128,), dtype="float32")
#       # Returns: {"schema_id": "schema_abc123...", "shape": (128,),
#       #           "ndim": 1, "numel": 128, "dtype": "float32"}
#     """
#   
#   def get(self, schema_id: str) -> SchemaInfo:
#     """
#     Retrieve schema by ID.
#     
#     Args:
#       - schema_id: Schema identifier
#     
#     Returns: SchemaInfo dict
#     Raises: SchemaNotFound if ID doesn't exist
#     """
#   
#   def from_anomaly(self, anomaly_id: str) -> SchemaInfo:
#     """
#     Get schema for an anomaly's stored vector.
#     
#     Args:
#       - anomaly_id: Anomaly identifier
#     
#     Returns: SchemaInfo dict
#     
#     Algorithm:
#       1. Fetch vector record from database (includes schema_id)
#       2. Get schema by schema_id
#       3. Return SchemaInfo
#     
#     Raises:
#       - ValueError: if anomaly_id not found or has no vector
#       - SchemaNotFound: if schema_id invalid (shouldn't happen)
#     """
#   
#   def list_all(self) -> list[SchemaInfo]:
#     """
#     List all registered schemas.
#     
#     Returns: List of SchemaInfo dicts
#     Useful for training when schema_id not specified
#     """
#   
#   def validate_shape(self, shape: tuple[int, ...]) -> None:
#     """
#     Validate that shape is supported (1D or 2D).
#     
#     Args:
#       - shape: Tensor dimensions
#     
#     Raises: UnsupportedShape with detailed message
#     
#     Validation rules:
#       - Must be tuple of ints
#       - Must be 1D: (D,) where D > 0
#       - OR 2D: (T, F) where T > 0 and F > 0
#       - No 0 dimensions allowed
#       - No 3D+ tensors (MVP limitation)
#     
#     Error messages should explain:
#       - What shape was provided
#       - What shapes are supported
#       - Why this matters (dense vs conv1d modes)
#     """
#   
#   def validate_dtype(self, dtype: str) -> None:
#     """
#     Validate NumPy dtype string.
#     
#     Args:
#       - dtype: NumPy dtype string
#     
#     Raises: ValidationError if invalid
#     
#     Supported dtypes for MVP:
#       - "float32" (primary)
#       - "float64"
#       - "int32", "int64" (if needed for counts)
#     
#     Can use np.dtype(dtype) to validate, catch TypeError/ValueError
#     """
#   
#   def _compute_metadata(self, shape: tuple[int, ...]) -> dict:
#     """
#     Compute ndim and numel from shape.
#     
#     Args:
#       - shape: Tensor dimensions
#     
#     Returns: dict with "ndim" and "numel"
#     
#     Example:
#       shape=(128,) → {"ndim": 1, "numel": 128}
#       shape=(10, 8) → {"ndim": 2, "numel": 80}
#     """
#   
#   def _row_to_schema_info(self, row: sqlite3.Row) -> SchemaInfo:
#     """
#     Convert database row to SchemaInfo dict.
#     
#     Args:
#       - row: Row from feature_schemas table
#     
#     Returns: SchemaInfo TypedDict
#     
#     Note: shape column is stored as string (JSON or comma-separated)
#     Parse back to tuple of ints
#     """
#
# Helper functions (module-level):
#
# - def shape_to_str(shape: tuple[int, ...]) -> str:
#   Convert shape tuple to string for database storage
#   Options:
#     - JSON: json.dumps(shape)
#     - Comma-separated: ",".join(map(str, shape))
#   Recommend JSON for clarity
#
# - def str_to_shape(s: str) -> tuple[int, ...]:
#   Parse stored shape string back to tuple
#   Handle both JSON and comma-separated formats
#   Return tuple of ints
#
# Usage patterns:
#   registry = SchemaRegistry(repo)
#   
#   # Register schema from tensor
#   arr = np.random.randn(128)
#   schema = registry.ensure(shape=arr.shape, dtype=str(arr.dtype))
#   
#   # Get schema for existing anomaly
#   schema = registry.from_anomaly("A1")
#   
#   # Training: resolve schema
#   schemas = registry.list_all()
#   if len(schemas) == 1:
#     schema = schemas[0]
#   else:
#     # Need user to specify which schema
#     raise SchemaNotFound("Multiple schemas exist, specify schema_id")
