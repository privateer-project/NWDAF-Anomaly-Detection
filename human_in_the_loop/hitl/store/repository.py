"""
Repository layer for HITL system.

This module provides high-level database operations following the repository
pattern. All database access should go through this layer.

Classes:
    Repository: High-level CRUD operations for all domain objects
"""

import sqlite3
from typing import Generator

from .sqlite import SQLite
from ..errors import DBError


class Repository:
    """
    High-level database operations for HITL domain objects.

    Provides CRUD operations for anomalies, feedback, schemas, vectors,
    models, and settings.

    Example:
        >>> db = SQLite("hitl.db")
        >>> repo = Repository(db)
        >>> repo.upsert_anomaly("A1", "2025-11-04T10:00:00Z", "sensor-1",
        ...                     "schema123", "2025-11-04T10:00:01Z", "2025-11-04T10:00:01Z")
    """

    def __init__(self, sqlite: SQLite):
        """
        Initialize repository with SQLite connection manager.

        Args:
            sqlite: SQLite instance for database access
        """
        self.db = sqlite

    # =========================================================================
    # ANOMALIES
    # =========================================================================

    def upsert_anomaly(
        self,
        anomaly_id: str,
        occurred_at: str,
        source: str,
        schema_id: str,
        created_at: str,
        updated_at: str,
    ) -> None:
        """
        Insert or update anomaly record.

        Args:
            anomaly_id: Unique anomaly identifier
            occurred_at: ISO timestamp when anomaly occurred
            source: Source system identifier
            schema_id: Schema ID for this anomaly's tensor
            created_at: ISO timestamp when record created
            updated_at: ISO timestamp when record updated

        Raises:
            DBError: On database errors
        """
        sql = """
        INSERT INTO anomalies (anomaly_id, occurred_at, source, schema_id, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(anomaly_id) DO UPDATE SET
            occurred_at = excluded.occurred_at,
            source = excluded.source,
            schema_id = excluded.schema_id,
            updated_at = excluded.updated_at
        """
        self.db.execute(
            sql, (anomaly_id, occurred_at, source, schema_id, created_at, updated_at)
        )

    def get_anomaly(self, anomaly_id: str) -> sqlite3.Row | None:
        """
        Fetch anomaly by ID.

        Args:
            anomaly_id: Anomaly identifier

        Returns:
            Row with anomaly data or None if not found
        """
        sql = "SELECT * FROM anomalies WHERE anomaly_id = ?"
        return self.db.fetchone(sql, (anomaly_id,))

    def list_anomalies(
        self,
        limit: int = 100,
        offset: int = 0,
        source: str | None = None,
    ) -> list[sqlite3.Row]:
        """
        List anomalies with pagination and optional filtering.

        Args:
            limit: Maximum rows to return
            offset: Skip this many rows
            source: Filter by source if provided

        Returns:
            List of anomaly rows
        """
        if source:
            sql = """
            SELECT * FROM anomalies
            WHERE source = ?
            ORDER BY occurred_at DESC
            LIMIT ? OFFSET ?
            """
            return self.db.fetchall(sql, (source, limit, offset))
        else:
            sql = """
            SELECT * FROM anomalies
            ORDER BY occurred_at DESC
            LIMIT ? OFFSET ?
            """
            return self.db.fetchall(sql, (limit, offset))

    # =========================================================================
    # FEEDBACK
    # =========================================================================

    def insert_feedback(
        self,
        feedback_id: str,
        anomaly_id: str,
        user_id: str,
        label: str,
        confidence: float | None,
        note: str | None,
        created_at: str,
    ) -> None:
        """
        Insert feedback record.

        Args:
            feedback_id: Unique feedback identifier
            anomaly_id: Associated anomaly ID (FK)
            user_id: User who provided feedback
            label: Feedback label (e.g., "TP", "FP")
            confidence: Optional confidence score (0.0-1.0)
            note: Optional text note
            created_at: ISO timestamp

        Raises:
            DBError: If anomaly_id invalid or other error
        """
        sql = """
        INSERT INTO feedback (feedback_id, anomaly_id, user_id, label, confidence, note, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        self.db.execute(
            sql, (feedback_id, anomaly_id, user_id, label, confidence, note, created_at)
        )

    def latest_feedback(self, anomaly_id: str) -> sqlite3.Row | None:
        """
        Get most recent feedback for anomaly.

        Args:
            anomaly_id: Anomaly identifier

        Returns:
            Most recent feedback row or None
        """
        sql = """
        SELECT * FROM feedback
        WHERE anomaly_id = ?
        ORDER BY created_at DESC
        LIMIT 1
        """
        return self.db.fetchone(sql, (anomaly_id,))

    def list_feedback(
        self,
        anomaly_id: str | None = None,
        label: str | None = None,
    ) -> list[sqlite3.Row]:
        """
        List feedback records with optional filters.

        Args:
            anomaly_id: Filter by specific anomaly
            label: Filter by label type

        Returns:
            List of feedback rows
        """
        if anomaly_id and label:
            sql = """
            SELECT * FROM feedback
            WHERE anomaly_id = ? AND label = ?
            ORDER BY created_at DESC
            """
            return self.db.fetchall(sql, (anomaly_id, label))
        elif anomaly_id:
            sql = """
            SELECT * FROM feedback
            WHERE anomaly_id = ?
            ORDER BY created_at DESC
            """
            return self.db.fetchall(sql, (anomaly_id,))
        elif label:
            sql = """
            SELECT * FROM feedback
            WHERE label = ?
            ORDER BY created_at DESC
            """
            return self.db.fetchall(sql, (label,))
        else:
            sql = "SELECT * FROM feedback ORDER BY created_at DESC"
            return self.db.fetchall(sql)

    # =========================================================================
    # FEATURE SCHEMAS
    # =========================================================================

    def get_schema(self, schema_id: str) -> sqlite3.Row | None:
        """
        Fetch schema by ID.

        Args:
            schema_id: Schema identifier

        Returns:
            Row with schema data or None
        """
        sql = "SELECT * FROM feature_schemas WHERE schema_id = ?"
        return self.db.fetchone(sql, (schema_id,))

    def insert_schema(
        self,
        schema_id: str,
        shape: str,
        ndim: int,
        numel: int,
        dtype: str,
        created_at: str,
    ) -> None:
        """
        Insert new schema (idempotent).

        Args:
            schema_id: Unique schema identifier
            shape: Shape as string (e.g., "128" or "8,128")
            ndim: Number of dimensions
            numel: Total number of elements
            dtype: NumPy dtype string
            created_at: ISO timestamp
        """
        sql = """
        INSERT OR IGNORE INTO feature_schemas (schema_id, shape, ndim, numel, dtype, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """
        self.db.execute(sql, (schema_id, shape, ndim, numel, dtype, created_at))

    def list_schemas(self) -> list[sqlite3.Row]:
        """
        List all registered schemas.

        Returns:
            List of schema rows
        """
        sql = "SELECT * FROM feature_schemas ORDER BY created_at DESC"
        return self.db.fetchall(sql)

    # =========================================================================
    # RAW VECTORS
    # =========================================================================

    def put_vector(
        self,
        anomaly_id: str,
        schema_id: str,
        blob: bytes,
        created_at: str,
    ) -> None:
        """
        Store tensor blob for anomaly (upsert).

        Args:
            anomaly_id: Anomaly identifier (FK)
            schema_id: Schema identifier (FK)
            blob: Tensor in .npy format
            created_at: ISO timestamp
        """
        sql = """
        INSERT INTO raw_vectors (anomaly_id, schema_id, tensor_blob, created_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(anomaly_id) DO UPDATE SET
            schema_id = excluded.schema_id,
            tensor_blob = excluded.tensor_blob
        """
        self.db.execute(sql, (anomaly_id, schema_id, blob, created_at))

    def get_vector(self, anomaly_id: str) -> tuple[str, bytes] | None:
        """
        Retrieve tensor blob for anomaly.

        Args:
            anomaly_id: Anomaly identifier

        Returns:
            (schema_id, blob) tuple or None
        """
        sql = "SELECT schema_id, tensor_blob FROM raw_vectors WHERE anomaly_id = ?"
        row = self.db.fetchone(sql, (anomaly_id,))
        if row:
            return (row["schema_id"], row["tensor_blob"])
        return None

    def iter_vectors(self, schema_id: str) -> Generator[tuple[str, bytes], None, None]:
        """
        Iterate over all vectors for a schema (for training).

        Args:
            schema_id: Schema identifier

        Yields:
            (anomaly_id, blob) tuples

        Note:
            Uses cursor iterator to avoid loading all blobs into memory.
        """
        sql = """
        SELECT anomaly_id, tensor_blob FROM raw_vectors
        WHERE schema_id = ?
        ORDER BY created_at
        """
        conn = self.db.connect()
        try:
            cursor = conn.execute(sql, (schema_id,))
            for row in cursor:
                yield (row["anomaly_id"], row["tensor_blob"])
        finally:
            conn.close()

    def count_vectors(self, schema_id: str) -> int:
        """
        Count vectors for a schema.

        Args:
            schema_id: Schema identifier

        Returns:
            Number of vectors
        """
        sql = "SELECT COUNT(*) FROM raw_vectors WHERE schema_id = ?"
        row = self.db.fetchone(sql, (schema_id,))
        return row[0] if row else 0

    # =========================================================================
    # MODELS
    # =========================================================================

    def insert_model(
        self,
        model_version: str,
        kind: str,
        schema_id: str,
        artifact_path: str,
        created_at: str,
    ) -> None:
        """
        Register trained model.

        Args:
            model_version: Unique model identifier (e.g., "AE-2025.11.04-1")
            kind: Model architecture ("dense" or "conv1d")
            schema_id: Schema this model was trained on
            artifact_path: Relative path to artifacts directory
            created_at: ISO timestamp
        """
        sql = """
        INSERT INTO models (model_version, kind, schema_id, artifact_path, created_at)
        VALUES (?, ?, ?, ?, ?)
        """
        self.db.execute(
            sql, (model_version, kind, schema_id, artifact_path, created_at)
        )

    def get_model(self, model_version: str) -> sqlite3.Row | None:
        """
        Fetch model metadata by version.

        Args:
            model_version: Model identifier

        Returns:
            Row with model data or None
        """
        sql = "SELECT * FROM models WHERE model_version = ?"
        return self.db.fetchone(sql, (model_version,))

    def list_models(self, kind: str | None = None) -> list[sqlite3.Row]:
        """
        List all models, optionally filtered by kind.

        Args:
            kind: Filter by architecture type

        Returns:
            List of model rows
        """
        if kind:
            sql = """
            SELECT * FROM models
            WHERE kind = ?
            ORDER BY created_at DESC
            """
            return self.db.fetchall(sql, (kind,))
        else:
            sql = "SELECT * FROM models ORDER BY created_at DESC"
            return self.db.fetchall(sql)

    # =========================================================================
    # SETTINGS
    # =========================================================================

    def set_setting(self, key: str, value: str) -> None:
        """
        Set a configuration value (upsert).

        Args:
            key: Setting key
            value: Setting value
        """
        sql = """
        INSERT INTO settings (key, value)
        VALUES (?, ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value
        """
        self.db.execute(sql, (key, value))

    def get_setting(self, key: str) -> str | None:
        """
        Get a configuration value.

        Args:
            key: Setting key

        Returns:
            Setting value or None
        """
        sql = "SELECT value FROM settings WHERE key = ?"
        row = self.db.fetchone(sql, (key,))
        return row["value"] if row else None

    def set_live_model(self, model_version: str) -> None:
        """
        Set the live model version.

        Args:
            model_version: Model identifier

        Raises:
            DBError: If model_version not found
        """
        # Validate model exists
        if not self.get_model(model_version):
            raise DBError(f"Model not found: {model_version}")

        self.set_setting("live_model_version", model_version)

    def get_live_model(self) -> str | None:
        """
        Get current live model version.

        Returns:
            Live model version or None
        """
        return self.get_setting("live_model_version")

