# Pytest Configuration and Fixtures
#
# This file defines shared pytest fixtures and configuration
# used across all test modules.
#
# Fixtures to implement:
#
# @pytest.fixture
# def temp_db(tmp_path):
#   """
#   Provide temporary SQLite database for testing.
#   
#   Creates fresh database in tmp_path for each test.
#   Applies DDL schema.
#   Returns path to database file.
#   
#   Usage:
#     def test_something(temp_db):
#       db = SQLite(temp_db)
#       ...
#   """
#
# @pytest.fixture
# def temp_artifacts_dir(tmp_path):
#   """
#   Provide temporary artifacts directory.
#   
#   Creates empty directory for artifact storage.
#   Returns path.
#   """
#
# @pytest.fixture
# def config(temp_db, temp_artifacts_dir):
#   """
#   Provide test Config instance.
#   
#   Uses temporary paths for database and artifacts.
#   Returns Config object.
#   """
#
# @pytest.fixture
# def sqlite(temp_db):
#   """
#   Provide SQLite instance with schema applied.
#   
#   Returns SQLite object ready for use.
#   """
#
# @pytest.fixture
# def repository(sqlite):
#   """
#   Provide Repository instance.
#   
#   Wraps sqlite fixture.
#   Returns Repository object.
#   """
#
# @pytest.fixture
# def registry(repository):
#   """
#   Provide SchemaRegistry instance.
#   
#   Returns SchemaRegistry object.
#   """
#
# @pytest.fixture
# def artifacts(temp_artifacts_dir):
#   """
#   Provide Artifacts manager instance.
#   
#   Returns Artifacts object.
#   """
#
# @pytest.fixture
# def hitl(config):
#   """
#   Provide fully initialized HITL instance.
#   
#   All subsystems wired together with temp paths.
#   Returns HITL object ready for testing.
#   """
#
# @pytest.fixture
# def sample_vector_1d():
#   """
#   Provide sample 1D feature vector.
#   
#   Returns NumPy array of shape (128,).
#   """
#
# @pytest.fixture
# def sample_vector_2d():
#   """
#   Provide sample 2D feature vector.
#   
#   Returns NumPy array of shape (10, 8).
#   """
#
# @pytest.fixture
# def sample_anomaly():
#   """
#   Provide sample anomaly metadata dict.
#   
#   Returns dict with anomaly_id, occurred_at, source.
#   """
#
# @pytest.fixture
# def trained_model_artifacts(artifacts, sample_vector_1d):
#   """
#   Provide trained model artifacts for testing inference.
#   
#   Creates minimal artifacts:
#   - Dummy model state dict
#   - Scaler params
#   - Threshold
#   - Config
#   
#   Returns (model_version, artifact_path) tuple.
#   """
#
# # Pytest configuration
# # Set markers, test discovery patterns, etc.
