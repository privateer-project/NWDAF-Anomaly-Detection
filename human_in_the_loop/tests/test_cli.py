# Tests for CLI
#
# Test coverage:
# - All CLI commands
# - Argument parsing
# - File I/O (.npy files)
# - Output formatting
# - Error handling
#
# Uses typer.testing.CliRunner or subprocess for testing.
#
# Test setup:
#
# from typer.testing import CliRunner
# from hitl.cli.main import app
#
# @pytest.fixture
# def runner():
#   """Provide CLI test runner."""
#   return CliRunner()
#
# Test functions to implement:
#
# def test_init_db_command(runner, temp_db):
#   """Test hitl init-db creates database."""
#
# def test_upsert_command(runner, hitl, tmp_path):
#   """Test hitl upsert with .npy file."""
#
# def test_feedback_command(runner, hitl):
#   """Test hitl feedback."""
#
# def test_train_command(runner, hitl):
#   """Test hitl train."""
#
# def test_set_live_command(runner, hitl):
#   """Test hitl set-live."""
#
# def test_predict_command_with_npy(runner, hitl, tmp_path):
#   """Test hitl predict --npy."""
#
# def test_predict_command_with_anomaly_id(runner, hitl):
#   """Test hitl predict --anomaly."""
#
# def test_list_anomalies_command(runner, hitl):
#   """Test hitl list-anomalies."""
#
# def test_list_models_command(runner, hitl):
#   """Test hitl list-models."""
#
# def test_stats_command(runner, hitl):
#   """Test hitl stats."""
#
# def test_health_command(runner, hitl):
#   """Test hitl health."""
#
# def test_get_anomaly_command(runner, hitl):
#   """Test hitl get-anomaly."""
#
# def test_export_vector_command(runner, hitl, tmp_path):
#   """Test hitl export-vector."""
#
# def test_command_error_handling(runner):
#   """Test error messages are user-friendly."""
#
# def test_help_text(runner):
#   """Test --help shows usage information."""
