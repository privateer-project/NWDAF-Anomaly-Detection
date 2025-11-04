#!/usr/bin/env python
"""
Development script to test database layer functionality.

Tests SQLite connection, repository operations, and all CRUD functionality.
"""

import sys
import tempfile
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hitl.store.sqlite import SQLite
from hitl.store.repository import Repository
from hitl.utils.time import now_iso
from hitl.utils.ids import uuid_str


def print_section(title):
    """Print a section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def test_database():
    """Test complete database functionality."""
    
    # Create temporary database
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        
        print_section("1. Database Initialization")
        print(f"Creating database at: {db_path}")
        
        sqlite = SQLite(str(db_path))
        repo = Repository(sqlite)
        
        print("✓ Database created successfully")
        print(f"✓ WAL mode enabled: {sqlite.db.execute('PRAGMA journal_mode').fetchone()[0] == 'wal'}")
        
        # Test anomaly operations
        print_section("2. Anomaly Operations")
        
        anomaly_id = uuid_str()
        occurred_at = now_iso()
        schema_id = "schema_test123"
        
        print(f"Creating anomaly: {anomaly_id}")
        repo.upsert_anomaly(
            anomaly_id=anomaly_id,
            occurred_at=occurred_at,
            source="network-monitor",
            schema_id=schema_id,
            created_at=occurred_at,
            updated_at=occurred_at,
        )
        print("✓ Anomaly created")
        
        # Retrieve anomaly
        anomaly = repo.get_anomaly(anomaly_id)
        print(f"✓ Retrieved anomaly: {anomaly['source']}")
        
        # List anomalies
        anomalies = repo.list_anomalies(limit=10)
        print(f"✓ Listed {len(anomalies)} anomalies")
        
        # Test feedback operations
        print_section("3. Feedback Operations")
        
        feedback_id = uuid_str()
        print(f"Submitting feedback: {feedback_id}")
        
        repo.insert_feedback(
            feedback_id=feedback_id,
            anomaly_id=anomaly_id,
            user_id="admin",
            label="true",
            confidence=0.95,
            note="Confirmed network anomaly",
            created_at=now_iso(),
        )
        print("✓ Feedback submitted")
        
        # Get latest feedback
        latest = repo.latest_feedback(anomaly_id)
        print(f"✓ Latest feedback label: {latest['label']}")
        
        # List feedback
        feedback_list = repo.list_feedback(limit=10)
        print(f"✓ Listed {len(feedback_list)} feedback entries")
        
        # Test schema operations
        print_section("4. Schema Operations")
        
        print("Registering schema...")
        repo.insert_schema(
            schema_id=schema_id,
            shape="[128]",
            ndim=1,
            numel=128,
            dtype="float32",
            created_at=now_iso(),
        )
        print("✓ Schema registered")
        
        # Retrieve schema
        schema = repo.get_schema(schema_id)
        print(f"✓ Retrieved schema: shape={schema['shape']}, dtype={schema['dtype']}")
        
        # List schemas
        schemas = repo.list_schemas()
        print(f"✓ Listed {len(schemas)} schemas")
        
        # Test vector operations
        print_section("5. Vector Operations")
        
        print("Storing vector...")
        test_blob = b"test_tensor_data_here"
        repo.put_vector(
            anomaly_id=anomaly_id,
            schema_id=schema_id,
            blob=test_blob,
            created_at=now_iso(),
        )
        print("✓ Vector stored")
        
        # Retrieve vector
        retrieved_schema_id, retrieved_blob = repo.get_vector(anomaly_id)
        print(f"✓ Retrieved vector: {len(retrieved_blob)} bytes")
        assert retrieved_blob == test_blob
        
        # Count vectors
        count = repo.count_vectors(schema_id)
        print(f"✓ Vector count for schema: {count}")
        
        # Test model operations
        print_section("6. Model Operations")
        
        model_version = "AE-2025.11.04-1"
        print(f"Registering model: {model_version}")
        
        repo.insert_model(
            model_version=model_version,
            mode="dense",
            schema_id=schema_id,
            artifact_path=f"artifacts/{model_version}",
            created_at=now_iso(),
        )
        print("✓ Model registered")
        
        # Retrieve model
        model = repo.get_model(model_version)
        print(f"✓ Retrieved model: mode={model['mode']}")
        
        # List models
        models = repo.list_models()
        print(f"✓ Listed {len(models)} models")
        
        # Test settings operations
        print_section("7. Settings Operations")
        
        print("Setting configuration...")
        repo.set_setting("training_enabled", "true")
        repo.set_setting("min_feedback_count", "10")
        print("✓ Settings configured")
        
        # Get settings
        training = repo.get_setting("training_enabled")
        min_feedback = repo.get_setting("min_feedback_count")
        print(f"✓ Training enabled: {training}")
        print(f"✓ Min feedback count: {min_feedback}")
        
        # Set live model
        print("Setting live model...")
        repo.set_live_model(model_version)
        print("✓ Live model set")
        
        # Get live model
        live = repo.get_live_model()
        print(f"✓ Live model: {live}")
        assert live == model_version
        
        # Test transaction rollback
        print_section("8. Transaction Rollback")
        
        print("Testing rollback...")
        try:
            with repo.db.tx() as conn:
                conn.execute("INSERT INTO settings (key, value) VALUES (?, ?)", 
                           ("temp_key", "temp_value"))
                # Verify it's there in transaction
                result = conn.execute("SELECT value FROM settings WHERE key = ?", 
                                    ("temp_key",)).fetchone()
                print(f"  In transaction: {result[0]}")
                # Force rollback
                raise Exception("Test rollback")
        except Exception:
            pass
        
        # Verify it's not there after rollback
        temp = repo.get_setting("temp_key")
        assert temp is None
        print("✓ Rollback successful - temp setting not persisted")
        
        print_section("Summary")
        print("✓ All database tests passed!")
        print(f"✓ Database location: {db_path}")
        print(f"✓ Total anomalies: {len(repo.list_anomalies())}")
        print(f"✓ Total feedback: {len(repo.list_feedback())}")
        print(f"✓ Total schemas: {len(repo.list_schemas())}")
        print(f"✓ Total models: {len(repo.list_models())}")


if __name__ == "__main__":
    try:
        test_database()
        print("\n" + "="*60)
        print("  ✓ ALL TESTS PASSED")
        print("="*60 + "\n")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
