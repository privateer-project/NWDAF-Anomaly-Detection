"""
Unit tests for Phase 3 database layer (DDL, SQLite, Repository).
"""

import pytest
import sys
import os
import tempfile
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hitl.store.sqlite import SQLite, row_to_dict
from hitl.store.repository import Repository
from hitl.errors import DBError
from hitl.utils.time import now_iso


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        yield db_path


@pytest.fixture
def sqlite(temp_db):
    """Create SQLite instance."""
    return SQLite(temp_db)


@pytest.fixture
def repo(sqlite):
    """Create Repository instance."""
    return Repository(sqlite)


class TestSQLite:
    """Test SQLite connection manager."""

    def test_init_creates_database(self, temp_db):
        """Test initialization creates database file."""
        assert not temp_db.exists()
        _ = SQLite(temp_db)
        assert temp_db.exists()

    def test_ddl_applied(self, sqlite):
        """Test DDL schema is applied."""
        # Check that tables exist
        tables = sqlite.fetchall(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        table_names = [row["name"] for row in tables]

        expected = [
            "anomalies",
            "feedback",
            "feature_schemas",
            "models",
            "raw_vectors",
            "settings",
        ]
        for table in expected:
            assert table in table_names

    def test_wal_mode_enabled(self, sqlite):
        """Test WAL mode is enabled."""
        row = sqlite.fetchone("PRAGMA journal_mode")
        assert row[0].upper() == "WAL"

    def test_foreign_keys_enabled(self, sqlite):
        """Test foreign keys are enabled."""
        row = sqlite.fetchone("PRAGMA foreign_keys")
        assert row[0] == 1

    def test_execute(self, sqlite):
        """Test execute method."""
        sqlite.execute(
            "INSERT INTO settings (key, value) VALUES (?, ?)",
            ("test_key", "test_value"),
        )
        row = sqlite.fetchone("SELECT value FROM settings WHERE key = ?", ("test_key",))
        assert row["value"] == "test_value"

    def test_fetchone(self, sqlite):
        """Test fetchone method."""
        sqlite.execute(
            "INSERT INTO settings (key, value) VALUES (?, ?)", ("key1", "value1")
        )
        row = sqlite.fetchone("SELECT * FROM settings WHERE key = ?", ("key1",))
        assert row is not None
        assert row["key"] == "key1"
        assert row["value"] == "value1"

    def test_fetchone_not_found(self, sqlite):
        """Test fetchone returns None when not found."""
        row = sqlite.fetchone("SELECT * FROM settings WHERE key = ?", ("nonexistent",))
        assert row is None

    def test_fetchall(self, sqlite):
        """Test fetchall method."""
        sqlite.execute("INSERT INTO settings (key, value) VALUES (?, ?)", ("k1", "v1"))
        sqlite.execute("INSERT INTO settings (key, value) VALUES (?, ?)", ("k2", "v2"))

        rows = sqlite.fetchall("SELECT * FROM settings ORDER BY key")
        assert len(rows) == 2
        assert rows[0]["key"] == "k1"
        assert rows[1]["key"] == "k2"

    def test_tx_commit(self, sqlite):
        """Test transaction commits on success."""
        with sqlite.tx() as conn:
            conn.execute(
                "INSERT INTO settings (key, value) VALUES (?, ?)",
                ("tx_key", "tx_value"),
            )

        row = sqlite.fetchone("SELECT value FROM settings WHERE key = ?", ("tx_key",))
        assert row["value"] == "tx_value"

    def test_tx_rollback(self, sqlite):
        """Test transaction rolls back on exception."""
        try:
            with sqlite.tx() as conn:
                conn.execute(
                    "INSERT INTO settings (key, value) VALUES (?, ?)",
                    ("roll_key", "roll_value"),
                )
                raise ValueError("Test error")
        except (DBError, ValueError):
            pass

        row = sqlite.fetchone("SELECT value FROM settings WHERE key = ?", ("roll_key",))
        assert row is None

    def test_row_to_dict(self, sqlite):
        """Test row_to_dict helper."""
        sqlite.execute(
            "INSERT INTO settings (key, value) VALUES (?, ?)",
            ("dict_key", "dict_value"),
        )
        row = sqlite.fetchone("SELECT * FROM settings WHERE key = ?", ("dict_key",))

        data = row_to_dict(row)
        assert isinstance(data, dict)
        assert data["key"] == "dict_key"
        assert data["value"] == "dict_value"


class TestRepository:
    """Test Repository operations."""

    def test_upsert_anomaly(self, repo):
        """Test upserting anomaly."""
        # First, create a schema
        repo.insert_schema("schema1", "128", 1, 128, "float32", now_iso())

        # Insert anomaly
        repo.upsert_anomaly(
            "A1", "2025-11-04T10:00:00Z", "sensor-1", "schema1", now_iso(), now_iso()
        )

        # Fetch back
        row = repo.get_anomaly("A1")
        assert row is not None
        assert row["anomaly_id"] == "A1"
        assert row["source"] == "sensor-1"

    def test_upsert_anomaly_update(self, repo):
        """Test updating existing anomaly."""
        # Create schema
        repo.insert_schema("schema1", "128", 1, 128, "float32", now_iso())

        # Insert
        repo.upsert_anomaly(
            "A1", "2025-11-04T10:00:00Z", "sensor-1", "schema1", now_iso(), now_iso()
        )

        # Update
        new_time = now_iso()
        repo.upsert_anomaly(
            "A1", "2025-11-04T11:00:00Z", "sensor-2", "schema1", now_iso(), new_time
        )

        # Verify update
        row = repo.get_anomaly("A1")
        assert row["source"] == "sensor-2"
        assert row["occurred_at"] == "2025-11-04T11:00:00Z"

    def test_list_anomalies(self, repo):
        """Test listing anomalies with pagination."""
        # Create schema
        repo.insert_schema("schema1", "128", 1, 128, "float32", now_iso())

        # Insert multiple
        for i in range(5):
            repo.upsert_anomaly(
                f"A{i}",
                f"2025-11-04T10:0{i}:00Z",
                "sensor-1",
                "schema1",
                now_iso(),
                now_iso(),
            )

        # List all
        rows = repo.list_anomalies(limit=10)
        assert len(rows) == 5

        # List with limit
        rows = repo.list_anomalies(limit=2)
        assert len(rows) == 2

        # List with offset
        rows = repo.list_anomalies(limit=2, offset=2)
        assert len(rows) == 2

    def test_list_anomalies_by_source(self, repo):
        """Test filtering anomalies by source."""
        # Create schema
        repo.insert_schema("schema1", "128", 1, 128, "float32", now_iso())

        # Insert with different sources
        repo.upsert_anomaly(
            "A1", "2025-11-04T10:00:00Z", "sensor-1", "schema1", now_iso(), now_iso()
        )
        repo.upsert_anomaly(
            "A2", "2025-11-04T10:01:00Z", "sensor-2", "schema1", now_iso(), now_iso()
        )
        repo.upsert_anomaly(
            "A3", "2025-11-04T10:02:00Z", "sensor-1", "schema1", now_iso(), now_iso()
        )

        # Filter by source
        rows = repo.list_anomalies(source="sensor-1")
        assert len(rows) == 2

    def test_insert_feedback(self, repo):
        """Test inserting feedback."""
        # Create schema and anomaly
        repo.insert_schema("schema1", "128", 1, 128, "float32", now_iso())
        repo.upsert_anomaly(
            "A1", "2025-11-04T10:00:00Z", "sensor-1", "schema1", now_iso(), now_iso()
        )

        # Insert feedback
        repo.insert_feedback("F1", "A1", "user1", "TP", 0.95, "Looks good", now_iso())

        # Verify
        fb = repo.latest_feedback("A1")
        assert fb is not None
        assert fb["feedback_id"] == "F1"
        assert fb["label"] == "TP"
        assert fb["confidence"] == 0.95

    def test_latest_feedback(self, repo):
        """Test getting latest feedback."""
        # Setup
        repo.insert_schema("schema1", "128", 1, 128, "float32", now_iso())
        repo.upsert_anomaly(
            "A1", "2025-11-04T10:00:00Z", "sensor-1", "schema1", now_iso(), now_iso()
        )

        # Insert multiple feedback
        repo.insert_feedback(
            "F1", "A1", "user1", "TP", 0.9, None, "2025-11-04T10:00:00Z"
        )
        repo.insert_feedback(
            "F2", "A1", "user1", "FP", 0.8, None, "2025-11-04T10:01:00Z"
        )

        # Get latest
        fb = repo.latest_feedback("A1")
        assert fb["feedback_id"] == "F2"  # Most recent

    def test_list_feedback(self, repo):
        """Test listing feedback with filters."""
        # Setup
        repo.insert_schema("schema1", "128", 1, 128, "float32", now_iso())
        repo.upsert_anomaly(
            "A1", "2025-11-04T10:00:00Z", "sensor-1", "schema1", now_iso(), now_iso()
        )
        repo.upsert_anomaly(
            "A2", "2025-11-04T10:01:00Z", "sensor-1", "schema1", now_iso(), now_iso()
        )

        # Insert feedback
        repo.insert_feedback("F1", "A1", "user1", "TP", None, None, now_iso())
        repo.insert_feedback("F2", "A1", "user2", "FP", None, None, now_iso())
        repo.insert_feedback("F3", "A2", "user1", "TP", None, None, now_iso())

        # Filter by anomaly
        rows = repo.list_feedback(anomaly_id="A1")
        assert len(rows) == 2

        # Filter by label
        rows = repo.list_feedback(label="TP")
        assert len(rows) == 2

        # Filter by both
        rows = repo.list_feedback(anomaly_id="A1", label="TP")
        assert len(rows) == 1

    def test_schema_operations(self, repo):
        """Test schema CRUD operations."""
        # Insert schema
        repo.insert_schema("schema1", "128", 1, 128, "float32", now_iso())

        # Get schema
        schema = repo.get_schema("schema1")
        assert schema is not None
        assert schema["shape"] == "128"
        assert schema["ndim"] == 1
        assert schema["numel"] == 128

        # List schemas
        schemas = repo.list_schemas()
        assert len(schemas) == 1

    def test_schema_idempotent(self, repo):
        """Test schema insert is idempotent."""
        # Insert twice
        repo.insert_schema("schema1", "128", 1, 128, "float32", now_iso())
        repo.insert_schema("schema1", "128", 1, 128, "float32", now_iso())

        # Should still be one
        schemas = repo.list_schemas()
        assert len(schemas) == 1

    def test_vector_operations(self, repo):
        """Test vector storage and retrieval."""
        # Setup
        repo.insert_schema("schema1", "128", 1, 128, "float32", now_iso())
        repo.upsert_anomaly(
            "A1", "2025-11-04T10:00:00Z", "sensor-1", "schema1", now_iso(), now_iso()
        )

        # Store vector
        blob = b"test_blob_data"
        repo.put_vector("A1", "schema1", blob, now_iso())

        # Retrieve
        result = repo.get_vector("A1")
        assert result is not None
        schema_id, retrieved_blob = result
        assert schema_id == "schema1"
        assert retrieved_blob == blob

    def test_iter_vectors(self, repo):
        """Test iterating over vectors."""
        # Setup
        repo.insert_schema("schema1", "128", 1, 128, "float32", now_iso())

        # Insert anomalies and vectors
        for i in range(3):
            aid = f"A{i}"
            repo.upsert_anomaly(
                aid,
                f"2025-11-04T10:0{i}:00Z",
                "sensor-1",
                "schema1",
                now_iso(),
                now_iso(),
            )
            repo.put_vector(aid, "schema1", f"blob{i}".encode(), now_iso())

        # Iterate
        vectors = list(repo.iter_vectors("schema1"))
        assert len(vectors) == 3
        assert all(isinstance(v[0], str) and isinstance(v[1], bytes) for v in vectors)

    def test_count_vectors(self, repo):
        """Test counting vectors."""
        # Setup
        repo.insert_schema("schema1", "128", 1, 128, "float32", now_iso())
        repo.insert_schema("schema2", "64", 1, 64, "float32", now_iso())

        # Insert vectors for different schemas
        for i in range(3):
            aid = f"A{i}"
            schema_id = "schema1" if i < 2 else "schema2"
            repo.upsert_anomaly(
                aid, now_iso(), "sensor-1", schema_id, now_iso(), now_iso()
            )
            repo.put_vector(aid, schema_id, b"blob", now_iso())

        # Count
        assert repo.count_vectors("schema1") == 2
        assert repo.count_vectors("schema2") == 1

    def test_model_operations(self, repo):
        """Test model CRUD operations."""
        # Setup schema
        repo.insert_schema("schema1", "128", 1, 128, "float32", now_iso())

        # Insert model
        repo.insert_model(
            "AE-2025.11.04-1",
            "dense",
            "schema1",
            "artifacts/AE-2025.11.04-1",
            now_iso(),
        )

        # Get model
        model = repo.get_model("AE-2025.11.04-1")
        assert model is not None
        assert model["kind"] == "dense"
        assert model["schema_id"] == "schema1"

        # List models
        models = repo.list_models()
        assert len(models) == 1

    def test_list_models_by_kind(self, repo):
        """Test filtering models by kind."""
        # Setup
        repo.insert_schema("schema1", "128", 1, 128, "float32", now_iso())

        # Insert models
        repo.insert_model("AE-2025.11.04-1", "dense", "schema1", "path1", now_iso())
        repo.insert_model("AE-2025.11.04-2", "conv1d", "schema1", "path2", now_iso())
        repo.insert_model("AE-2025.11.04-3", "dense", "schema1", "path3", now_iso())

        # Filter
        dense_models = repo.list_models(kind="dense")
        assert len(dense_models) == 2

        conv_models = repo.list_models(kind="conv1d")
        assert len(conv_models) == 1

    def test_settings_operations(self, repo):
        """Test settings key-value store."""
        # Set setting
        repo.set_setting("test_key", "test_value")

        # Get setting
        value = repo.get_setting("test_key")
        assert value == "test_value"

        # Update setting
        repo.set_setting("test_key", "new_value")
        value = repo.get_setting("test_key")
        assert value == "new_value"

    def test_live_model_operations(self, repo):
        """Test live model management."""
        # Setup
        repo.insert_schema("schema1", "128", 1, 128, "float32", now_iso())
        repo.insert_model("AE-2025.11.04-1", "dense", "schema1", "path1", now_iso())

        # Set live model
        repo.set_live_model("AE-2025.11.04-1")

        # Get live model
        live = repo.get_live_model()
        assert live == "AE-2025.11.04-1"

    def test_set_live_model_invalid(self, repo):
        """Test setting live model with invalid version."""
        with pytest.raises(DBError, match="Model not found"):
            repo.set_live_model("nonexistent-model")

    def test_foreign_key_cascade(self, repo):
        """Test foreign key cascade deletion."""
        # Setup
        repo.insert_schema("schema1", "128", 1, 128, "float32", now_iso())
        repo.upsert_anomaly(
            "A1", now_iso(), "sensor-1", "schema1", now_iso(), now_iso()
        )
        repo.put_vector("A1", "schema1", b"blob", now_iso())
        repo.insert_feedback("F1", "A1", "user1", "TP", None, None, now_iso())

        # Delete anomaly (should cascade to vector and feedback)
        repo.db.execute("DELETE FROM anomalies WHERE anomaly_id = ?", ("A1",))

        # Verify cascade
        assert repo.get_anomaly("A1") is None
        assert repo.get_vector("A1") is None
        assert repo.latest_feedback("A1") is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
