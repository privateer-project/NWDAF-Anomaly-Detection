"""
Unit tests for Phase 1 utilities (time, ids, logging, settings, errors).
"""

import pytest
from datetime import datetime, timezone
import sys
import os

# Add parent directory to path to import hitl
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from hitl.utils import time as time_utils
from hitl.utils import ids as id_utils
from hitl.utils import logging as logging_utils
from hitl import settings
from hitl import errors


class TestTimeUtils:
    """Test time utilities."""
    
    def test_utcnow(self):
        """Test utcnow returns timezone-aware datetime."""
        now = time_utils.utcnow()
        assert isinstance(now, datetime)
        assert now.tzinfo is not None
        assert now.tzinfo == timezone.utc
    
    def test_now_iso(self):
        """Test now_iso returns ISO 8601 string."""
        iso_str = time_utils.now_iso()
        assert isinstance(iso_str, str)
        assert "T" in iso_str
        assert "+" in iso_str or "Z" in iso_str
    
    def test_parse_iso_valid(self):
        """Test parse_iso with valid timestamp."""
        dt = time_utils.parse_iso("2025-11-04T10:30:45+00:00")
        assert dt.year == 2025
        assert dt.month == 11
        assert dt.day == 4
        assert dt.hour == 10
        assert dt.minute == 30
        assert dt.second == 45
        assert dt.tzinfo is not None
    
    def test_parse_iso_invalid(self):
        """Test parse_iso with invalid timestamp."""
        with pytest.raises(ValueError):
            time_utils.parse_iso("not a timestamp")
    
    def test_to_iso(self):
        """Test to_iso converts datetime to string."""
        dt = datetime(2025, 11, 4, 10, 30, 45, tzinfo=timezone.utc)
        iso_str = time_utils.to_iso(dt)
        assert "2025-11-04" in iso_str
        assert "10:30:45" in iso_str
    
    def test_validate_iso_valid(self):
        """Test validate_iso with valid timestamp."""
        assert time_utils.validate_iso("2025-11-04T10:30:45+00:00") is True
    
    def test_validate_iso_invalid(self):
        """Test validate_iso with invalid timestamp."""
        assert time_utils.validate_iso("not a timestamp") is False


class TestIdUtils:
    """Test ID utilities."""
    
    def test_uuid_str(self):
        """Test uuid_str generates unique IDs."""
        id1 = id_utils.uuid_str()
        id2 = id_utils.uuid_str()
        assert id1 != id2
        assert len(id1) == 32  # UUID4 hex without hyphens
        assert isinstance(id1, str)
    
    def test_sha1_bytes(self):
        """Test sha1_bytes generates correct hash."""
        hash1 = id_utils.sha1_bytes(b"hello")
        hash2 = id_utils.sha1_bytes(b"hello")
        hash3 = id_utils.sha1_bytes(b"world")
        
        assert hash1 == hash2  # Deterministic
        assert hash1 != hash3  # Different inputs
        assert len(hash1) == 40  # SHA-1 hex length
    
    def test_schema_id_deterministic(self):
        """Test schema_id is deterministic for same inputs."""
        id1 = id_utils.schema_id((128,), "float32")
        id2 = id_utils.schema_id((128,), "float32")
        id3 = id_utils.schema_id((64,), "float32")
        
        assert id1 == id2  # Same inputs
        assert id1 != id3  # Different inputs
        assert len(id1) == 40  # SHA-1 hex length
    
    def test_model_version(self):
        """Test model_version generates correct format."""
        dt = datetime(2025, 11, 4, tzinfo=timezone.utc)
        version = id_utils.model_version(dt, 1)
        assert version == "AE-2025.11.04-1"
        
        version2 = id_utils.model_version(dt, 2)
        assert version2 == "AE-2025.11.04-2"
    
    def test_feedback_id(self):
        """Test feedback_id generates deterministic IDs."""
        dt = datetime(2025, 11, 4, 10, 30, 0, tzinfo=timezone.utc)
        fid1 = id_utils.feedback_id("A1", "user1", dt)
        fid2 = id_utils.feedback_id("A1", "user1", dt)
        fid3 = id_utils.feedback_id("A2", "user1", dt)
        
        assert fid1 == fid2  # Same inputs
        assert fid1 != fid3  # Different anomaly_id
        assert len(fid1) == 40  # SHA-1 hex length


class TestLoggingUtils:
    """Test logging utilities."""
    
    def test_configure_logging(self):
        """Test configure_logging doesn't raise errors."""
        logging_utils.configure_logging(level="INFO", dev_mode=True)
        # Just verify it doesn't crash
        assert True
    
    def test_get_logger(self):
        """Test get_logger returns logger instance."""
        logging_utils.configure_logging(level="INFO", dev_mode=True)
        logger = logging_utils.get_logger(__name__)
        assert logger is not None
        # Test that we can log without errors
        logger.info("test message", key="value")
    
    def test_bind_and_clear_context(self):
        """Test bind_context and clear_context."""
        logging_utils.configure_logging(level="INFO", dev_mode=True)
        
        # Bind context
        logging_utils.bind_context(request_id="abc123", user_id="user1")
        
        # Clear context
        logging_utils.clear_context()
        
        # Just verify no errors
        assert True


class TestSettings:
    """Test settings module."""
    
    def test_config_dataclass(self):
        """Test Config dataclass initialization."""
        config = settings.Config()
        assert config.sqlite_path is not None
        assert config.artifacts_dir is not None
        assert config.mode in ("dense", "conv1d")
        assert config.log_level == "INFO"
        assert config.dev_mode is False
    
    def test_get_env_config(self):
        """Test get_env_config reads from environment."""
        # Set environment variable
        os.environ["HITL_MODE"] = "conv1d"
        os.environ["HITL_LOG_LEVEL"] = "DEBUG"
        
        config = settings.get_env_config()
        assert config.mode == "conv1d"
        assert config.log_level == "DEBUG"
        
        # Clean up
        del os.environ["HITL_MODE"]
        del os.environ["HITL_LOG_LEVEL"]
    
    def test_validate_config_valid(self):
        """Test validate_config with valid config."""
        config = settings.Config(mode="dense", log_level="INFO")
        settings.validate_config(config)  # Should not raise
    
    def test_validate_config_invalid_mode(self):
        """Test validate_config with invalid mode."""
        config = settings.Config(mode="invalid")  # type: ignore
        with pytest.raises(ValueError, match="Invalid mode"):
            settings.validate_config(config)
    
    def test_validate_config_invalid_log_level(self):
        """Test validate_config with invalid log level."""
        config = settings.Config(log_level="INVALID")
        with pytest.raises(ValueError, match="Invalid log_level"):
            settings.validate_config(config)


class TestErrors:
    """Test custom exceptions."""
    
    def test_hitl_error(self):
        """Test base HITLError."""
        with pytest.raises(errors.HITLError):
            raise errors.HITLError("Test error")
    
    def test_unsupported_shape(self):
        """Test UnsupportedShape exception."""
        with pytest.raises(errors.UnsupportedShape):
            raise errors.UnsupportedShape("Expected 1D or 2D")
    
    def test_shape_mismatch(self):
        """Test ShapeMismatch exception."""
        with pytest.raises(errors.ShapeMismatch):
            raise errors.ShapeMismatch("Expected (128,), got (64,)")
    
    def test_schema_not_found(self):
        """Test SchemaNotFound exception."""
        with pytest.raises(errors.SchemaNotFound):
            raise errors.SchemaNotFound("Schema abc123 not found")
    
    def test_no_live_model(self):
        """Test NoLiveModel exception."""
        with pytest.raises(errors.NoLiveModel):
            raise errors.NoLiveModel("No live model set")
    
    def test_artifact_missing(self):
        """Test ArtifactMissing exception."""
        with pytest.raises(errors.ArtifactMissing):
            raise errors.ArtifactMissing("Missing scaler.json")
    
    def test_db_error(self):
        """Test DBError exception."""
        with pytest.raises(errors.DBError):
            raise errors.DBError("Database error")
    
    def test_validation_error(self):
        """Test ValidationError exception."""
        with pytest.raises(errors.ValidationError):
            raise errors.ValidationError("Invalid data")
    
    def test_exception_hierarchy(self):
        """Test that all exceptions inherit from HITLError."""
        assert issubclass(errors.UnsupportedShape, errors.HITLError)
        assert issubclass(errors.ShapeMismatch, errors.HITLError)
        assert issubclass(errors.SchemaNotFound, errors.HITLError)
        assert issubclass(errors.NoLiveModel, errors.HITLError)
        assert issubclass(errors.ArtifactMissing, errors.HITLError)
        assert issubclass(errors.DBError, errors.HITLError)
        assert issubclass(errors.ValidationError, errors.HITLError)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
