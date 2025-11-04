# Tests for Store Module (sqlite.py and repository.py)
#
# Test coverage:
# - SQLite connection and DDL application
# - Basic CRUD operations
# - Transaction handling
# - Repository methods for all tables
# - Foreign key constraints
# - Error handling
#
# Test classes/functions to implement:
#
# class TestSQLite:
#   """Test SQLite connection and basic operations."""
#   
#   def test_init_creates_database(temp_db):
#     """Test database file creation and schema application."""
#   
#   def test_connect_returns_connection(sqlite):
#     """Test connection creation with proper settings."""
#   
#   def test_execute_inserts_data(sqlite):
#     """Test execute method for INSERT."""
#   
#   def test_fetchone_returns_row(sqlite):
#     """Test fetchone for SELECT queries."""
#   
#   def test_fetchall_returns_list(sqlite):
#     """Test fetchall for multiple rows."""
#   
#   def test_transaction_commits_on_success(sqlite):
#     """Test transaction context manager commits."""
#   
#   def test_transaction_rolls_back_on_error(sqlite):
#     """Test transaction rollback on exception."""
#
# class TestRepository:
#   """Test Repository CRUD operations."""
#   
#   # Anomalies
#   def test_upsert_anomaly_creates_new(repository):
#     """Test creating new anomaly."""
#   
#   def test_upsert_anomaly_updates_existing(repository):
#     """Test updating existing anomaly."""
#   
#   def test_get_anomaly_returns_row(repository):
#     """Test retrieving anomaly by ID."""
#   
#   def test_get_anomaly_returns_none_if_not_found(repository):
#     """Test None return for missing anomaly."""
#   
#   # Feedback
#   def test_insert_feedback_creates_record(repository):
#     """Test feedback insertion."""
#   
#   def test_latest_feedback_returns_newest(repository):
#     """Test getting most recent feedback."""
#   
#   # Schemas
#   def test_insert_schema_creates_record(repository):
#     """Test schema insertion."""
#   
#   def test_get_schema_returns_row(repository):
#     """Test schema retrieval."""
#   
#   # Vectors
#   def test_put_vector_stores_blob(repository):
#     """Test storing tensor blob."""
#   
#   def test_get_vector_returns_blob(repository):
#     """Test retrieving tensor blob."""
#   
#   def test_iter_vectors_yields_all_for_schema(repository):
#     """Test iterating vectors by schema."""
#   
#   # Models
#   def test_insert_model_creates_record(repository):
#     """Test model registration."""
#   
#   def test_set_live_model_updates_setting(repository):
#     """Test setting live model version."""
#   
#   def test_get_live_model_returns_version(repository):
#     """Test retrieving live model version."""
#   
#   # Settings
#   def test_set_setting_stores_value(repository):
#     """Test key-value storage."""
#   
#   def test_get_setting_returns_value(repository):
#     """Test key-value retrieval."""
