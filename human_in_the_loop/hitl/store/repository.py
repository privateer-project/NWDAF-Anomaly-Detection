# Repository Layer - High-Level Database Operations
#
# This module provides a repository pattern over SQLite for domain objects.
# All database operations should go through this layer.
#
# Class to implement:
#
# class Repository:
#   """High-level database operations for HITL domain objects."""
#   
#   def __init__(self, sqlite: SQLite):
#     """
#     Initialize repository with SQLite connection manager.
#     Args:
#       - sqlite: SQLite instance for database access
#     """
#     self.db = sqlite
#   
#   # ===== ANOMALIES =====
#   
#   def upsert_anomaly(
#     self,
#     anomaly_id: str,
#     occurred_at: str,
#     source: str,
#     created_at: str,
#     updated_at: str
#   ) -> None:
#     """
#     Insert or update anomaly record.
#     
#     SQL: INSERT OR REPLACE INTO anomalies ...
#     On conflict, update updated_at timestamp.
#     
#     Args: anomaly metadata fields
#     Returns: None
#     Raises: DBError on failure
#     """
#   
#   def get_anomaly(self, anomaly_id: str) -> sqlite3.Row | None:
#     """
#     Fetch anomaly by ID.
#     
#     SQL: SELECT * FROM anomalies WHERE anomaly_id = ?
#     Returns: Row or None if not found
#     """
#   
#   def list_anomalies(
#     self,
#     limit: int = 100,
#     offset: int = 0,
#     source: str | None = None
#   ) -> list[sqlite3.Row]:
#     """
#     List anomalies with pagination and optional filtering.
#     
#     SQL: SELECT * FROM anomalies WHERE ... ORDER BY occurred_at DESC
#     Args:
#       - limit: max rows to return
#       - offset: skip this many rows
#       - source: filter by source if provided
#     """
#   
#   # ===== FEEDBACK =====
#   
#   def insert_feedback(
#     self,
#     feedback_id: str,
#     anomaly_id: str,
#     user_id: str,
#     label: str,
#     confidence: float | None,
#     note: str | None,
#     created_at: str
#   ) -> None:
#     """
#     Insert feedback record.
#     
#     SQL: INSERT INTO feedback ...
#     Validate anomaly_id exists (FK constraint will catch this)
#     
#     Args: feedback fields
#     Returns: None
#     Raises: DBError if anomaly_id invalid or other error
#     """
#   
#   def latest_feedback(self, anomaly_id: str) -> sqlite3.Row | None:
#     """
#     Get most recent feedback for anomaly.
#     
#     SQL: SELECT * FROM feedback WHERE anomaly_id = ?
#          ORDER BY created_at DESC LIMIT 1
#     Returns: Row or None
#     """
#   
#   def list_feedback(
#     self,
#     anomaly_id: str | None = None,
#     label: str | None = None
#   ) -> list[sqlite3.Row]:
#     """
#     List feedback records with optional filters.
#     
#     Args:
#       - anomaly_id: filter by specific anomaly
#       - label: filter by label type (TP, FP, etc.)
#     """
#   
#   # ===== FEATURE SCHEMAS =====
#   
#   def get_schema(self, schema_id: str) -> sqlite3.Row | None:
#     """
#     Fetch schema by ID.
#     
#     SQL: SELECT * FROM feature_schemas WHERE schema_id = ?
#     Returns: Row with shape, ndim, numel, dtype or None
#     """
#   
#   def insert_schema(
#     self,
#     schema_id: str,
#     shape: str,  # JSON or comma-separated tuple string
#     ndim: int,
#     numel: int,
#     dtype: str,
#     created_at: str
#   ) -> None:
#     """
#     Insert new schema (idempotent).
#     
#     SQL: INSERT OR IGNORE INTO feature_schemas ...
#     If schema_id exists, this is a no-op.
#     """
#   
#   def list_schemas(self) -> list[sqlite3.Row]:
#     """
#     List all registered schemas.
#     
#     SQL: SELECT * FROM feature_schemas ORDER BY created_at DESC
#     """
#   
#   # ===== RAW VECTORS =====
#   
#   def put_vector(
#     self,
#     anomaly_id: str,
#     schema_id: str,
#     blob: bytes,
#     created_at: str
#   ) -> None:
#     """
#     Store tensor blob for anomaly (upsert).
#     
#     SQL: INSERT OR REPLACE INTO raw_vectors ...
#     Args:
#       - anomaly_id: FK to anomalies
#       - schema_id: FK to feature_schemas
#       - blob: .npy format bytes
#       - created_at: timestamp
#     """
#   
#   def get_vector(self, anomaly_id: str) -> tuple[str, bytes] | None:
#     """
#     Retrieve tensor blob for anomaly.
#     
#     SQL: SELECT schema_id, tensor_blob FROM raw_vectors
#          WHERE anomaly_id = ?
#     Returns: (schema_id, blob) tuple or None
#     """
#   
#   def iter_vectors(self, schema_id: str) -> Generator[tuple[str, bytes], None, None]:
#     """
#     Iterate over all vectors for a schema (for training).
#     
#     SQL: SELECT anomaly_id, tensor_blob FROM raw_vectors
#          WHERE schema_id = ? ORDER BY created_at
#     Yields: (anomaly_id, blob) tuples
#     
#     Implementation note: Use cursor as iterator to avoid loading
#     all blobs into memory at once.
#     """
#   
#   def count_vectors(self, schema_id: str) -> int:
#     """
#     Count vectors for a schema.
#     
#     SQL: SELECT COUNT(*) FROM raw_vectors WHERE schema_id = ?
#     Returns: integer count
#     """
#   
#   # ===== MODELS =====
#   
#   def insert_model(
#     self,
#     model_version: str,
#     kind: str,
#     artifact_path: str,
#     created_at: str
#   ) -> None:
#     """
#     Register trained model.
#     
#     SQL: INSERT INTO models ...
#     Args:
#       - model_version: unique identifier (e.g., "AE-2025.11.04-1")
#       - kind: "dense" or "conv1d"
#       - artifact_path: relative path to artifacts directory
#       - created_at: timestamp
#     """
#   
#   def get_model(self, model_version: str) -> sqlite3.Row | None:
#     """
#     Fetch model metadata by version.
#     
#     SQL: SELECT * FROM models WHERE model_version = ?
#     """
#   
#   def list_models(self, kind: str | None = None) -> list[sqlite3.Row]:
#     """
#     List all models, optionally filtered by kind.
#     
#     SQL: SELECT * FROM models WHERE ... ORDER BY created_at DESC
#     """
#   
#   # ===== SETTINGS =====
#   
#   def set_setting(self, key: str, value: str) -> None:
#     """
#     Set a configuration value (upsert).
#     
#     SQL: INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)
#     """
#   
#   def get_setting(self, key: str) -> str | None:
#     """
#     Get a configuration value.
#     
#     SQL: SELECT value FROM settings WHERE key = ?
#     Returns: value string or None
#     """
#   
#   def set_live_model(self, model_version: str) -> None:
#     """
#     Set the live model version.
#     
#     Validates that model_version exists in models table.
#     Then: self.set_setting("live_model_version", model_version)
#     Raises: DBError if model_version not found
#     """
#   
#   def get_live_model(self) -> str | None:
#     """
#     Get current live model version.
#     
#     Returns: self.get_setting("live_model_version")
#     """
#
# Usage patterns:
#   db = SQLite("hitl.db")
#   repo = Repository(db)
#   
#   # Insert anomaly
#   repo.upsert_anomaly(
#     anomaly_id="A1",
#     occurred_at="2025-11-04T10:00:00Z",
#     source="unit-1",
#     created_at=now_iso(),
#     updated_at=now_iso()
#   )
#   
#   # Store vector
#   repo.put_vector("A1", schema_id, blob, now_iso())
#   
#   # Training: iterate vectors
#   for aid, blob in repo.iter_vectors(schema_id):
#     tensor = decode_npy(blob)
#     # ... accumulate for training
#   
#   # Set live model
#   repo.set_live_model("AE-2025.11.04-1")
