"""
Unit tests for Phase 2 type system (TypedDicts and Pydantic models).
"""

import pytest
import sys
import os
from pydantic import ValidationError

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from hitl.types import (
    SchemaInfo,
    TrainParams,
    PredictResult,
    AnomalyRecord,
    FeedbackRecord,
    AnomalyUpsert,
    FeedbackIn,
    TrainRequest,
    PredictIn,
    PredictOut,
)


class TestTypedDicts:
    """Test TypedDict definitions."""
    
    def test_schema_info(self):
        """Test SchemaInfo TypedDict."""
        schema: SchemaInfo = {
            "schema_id": "abc123",
            "shape": (128,),
            "ndim": 1,
            "numel": 128,
            "dtype": "float32",
        }
        assert schema["schema_id"] == "abc123"
        assert schema["shape"] == (128,)
        assert schema["ndim"] == 1
        assert schema["numel"] == 128
        assert schema["dtype"] == "float32"
    
    def test_train_params(self):
        """Test TrainParams TypedDict."""
        params: TrainParams = {
            "mode": "dense",
            "epochs": 100,
            "batch_size": 32,
            "lr": 0.001,
            "val_split": 0.1,
            "patience": 10,
            "percentile": 99.5,
        }
        assert params["mode"] == "dense"
        assert params["epochs"] == 100
        assert params["lr"] == 0.001
    
    def test_predict_result(self):
        """Test PredictResult TypedDict."""
        result: PredictResult = {
            "label": 1,
            "score": 0.85,
            "threshold": 0.50,
            "model_version": "AE-2025.11.04-1",
        }
        assert result["label"] == 1
        assert result["score"] == 0.85
        assert result["threshold"] == 0.50
        assert result["model_version"] == "AE-2025.11.04-1"
    
    def test_anomaly_record(self):
        """Test AnomalyRecord TypedDict."""
        record: AnomalyRecord = {
            "anomaly_id": "A1",
            "occurred_at": "2025-11-04T10:00:00Z",
            "source": "sensor-1",
            "created_at": "2025-11-04T10:00:01Z",
            "updated_at": "2025-11-04T10:00:01Z",
        }
        assert record["anomaly_id"] == "A1"
        assert record["source"] == "sensor-1"
    
    def test_feedback_record(self):
        """Test FeedbackRecord TypedDict."""
        feedback: FeedbackRecord = {
            "feedback_id": "F1",
            "anomaly_id": "A1",
            "user_id": "user1",
            "label": "TP",
            "confidence": 0.9,
            "note": "Definitely anomalous",
            "created_at": "2025-11-04T10:00:00Z",
        }
        assert feedback["feedback_id"] == "F1"
        assert feedback["anomaly_id"] == "A1"
        assert feedback["label"] == "TP"
        assert feedback["confidence"] == 0.9


class TestPydanticModels:
    """Test Pydantic model validation."""
    
    def test_anomaly_upsert_valid(self):
        """Test AnomalyUpsert with valid data."""
        req = AnomalyUpsert(
            anomaly_id="A1",
            occurred_at="2025-11-04T10:00:00Z",
            source="sensor-1",
            tensor=[1.0, 2.0, 3.0],
        )
        assert req.anomaly_id == "A1"
        assert req.source == "sensor-1"
        assert req.tensor == [1.0, 2.0, 3.0]
        assert req.dtype == "float32"
    
    def test_anomaly_upsert_2d_tensor(self):
        """Test AnomalyUpsert with 2D tensor."""
        req = AnomalyUpsert(
            anomaly_id="A1",
            occurred_at="2025-11-04T10:00:00Z",
            source="sensor-1",
            tensor=[[1.0, 2.0], [3.0, 4.0]],
        )
        assert req.tensor == [[1.0, 2.0], [3.0, 4.0]]
    
    def test_anomaly_upsert_empty_tensor(self):
        """Test AnomalyUpsert rejects empty tensor."""
        with pytest.raises(ValidationError):
            AnomalyUpsert(
                anomaly_id="A1",
                occurred_at="2025-11-04T10:00:00Z",
                source="sensor-1",
                tensor=[],
            )
    
    def test_anomaly_upsert_invalid_2d_tensor(self):
        """Test AnomalyUpsert rejects irregular 2D tensor."""
        with pytest.raises(ValidationError):
            AnomalyUpsert(
                anomaly_id="A1",
                occurred_at="2025-11-04T10:00:00Z",
                source="sensor-1",
                tensor=[[1.0, 2.0], [3.0]],  # Irregular shape
            )
    
    def test_anomaly_upsert_missing_fields(self):
        """Test AnomalyUpsert requires all fields."""
        with pytest.raises(ValidationError):
            AnomalyUpsert(
                anomaly_id="A1",
                occurred_at="2025-11-04T10:00:00Z",
                # Missing source and tensor
            )
    
    def test_feedback_in_valid(self):
        """Test FeedbackIn with valid data."""
        feedback = FeedbackIn(
            anomaly_id="A1",
            user_id="user1",
            label="TP",
            confidence=0.95,
            note="Looks good",
        )
        assert feedback.anomaly_id == "A1"
        assert feedback.user_id == "user1"
        assert feedback.label == "TP"
        assert feedback.confidence == 0.95
        assert feedback.note == "Looks good"
    
    def test_feedback_in_optional_fields(self):
        """Test FeedbackIn with optional fields omitted."""
        feedback = FeedbackIn(
            anomaly_id="A1",
            user_id="user1",
            label="TP",
        )
        assert feedback.confidence is None
        assert feedback.note is None
    
    def test_feedback_in_invalid_confidence(self):
        """Test FeedbackIn rejects invalid confidence."""
        with pytest.raises(ValidationError):
            FeedbackIn(
                anomaly_id="A1",
                user_id="user1",
                label="TP",
                confidence=1.5,  # > 1.0
            )
        
        with pytest.raises(ValidationError):
            FeedbackIn(
                anomaly_id="A1",
                user_id="user1",
                label="TP",
                confidence=-0.1,  # < 0.0
            )
    
    def test_train_request_valid(self):
        """Test TrainRequest with valid data."""
        req = TrainRequest(
            mode="dense",
            schema_id="abc123",
            params={"epochs": 50, "batch_size": 64},
        )
        assert req.mode == "dense"
        assert req.schema_id == "abc123"
        assert req.params == {"epochs": 50, "batch_size": 64}
    
    def test_train_request_optional_fields(self):
        """Test TrainRequest with optional fields omitted."""
        req = TrainRequest(mode="conv1d")
        assert req.mode == "conv1d"
        assert req.schema_id is None
        assert req.params is None
    
    def test_train_request_invalid_mode(self):
        """Test TrainRequest rejects invalid mode."""
        with pytest.raises(ValidationError):
            TrainRequest(mode="invalid")  # type: ignore
    
    def test_predict_in_with_tensor(self):
        """Test PredictIn with tensor."""
        req = PredictIn(tensor=[1.0, 2.0, 3.0])
        assert req.tensor == [1.0, 2.0, 3.0]
        assert req.anomaly_id is None
    
    def test_predict_in_with_anomaly_id(self):
        """Test PredictIn with anomaly_id."""
        req = PredictIn(anomaly_id="A1")
        assert req.anomaly_id == "A1"
        assert req.tensor is None
    
    def test_predict_in_requires_one(self):
        """Test PredictIn requires exactly one of tensor or anomaly_id."""
        # Neither provided
        with pytest.raises(ValidationError, match="Must provide either"):
            PredictIn()
    
    def test_predict_in_rejects_both(self):
        """Test PredictIn rejects both tensor and anomaly_id."""
        with pytest.raises(ValidationError, match="Cannot provide both"):
            PredictIn(tensor=[1.0, 2.0], anomaly_id="A1")
    
    def test_predict_out_valid(self):
        """Test PredictOut with valid data."""
        result = PredictOut(
            label=1,
            score=0.85,
            threshold=0.50,
            model_version="AE-2025.11.04-1",
        )
        assert result.label == 1
        assert result.score == 0.85
        assert result.threshold == 0.50
        assert result.model_version == "AE-2025.11.04-1"
    
    def test_predict_out_invalid_label(self):
        """Test PredictOut rejects invalid label."""
        with pytest.raises(ValidationError):
            PredictOut(
                label=2,  # Must be 0 or 1
                score=0.85,
                threshold=0.50,
                model_version="AE-2025.11.04-1",
            )
        
        with pytest.raises(ValidationError):
            PredictOut(
                label=-1,  # Must be >= 0
                score=0.85,
                threshold=0.50,
                model_version="AE-2025.11.04-1",
            )
    
    def test_predict_out_negative_score(self):
        """Test PredictOut rejects negative score."""
        with pytest.raises(ValidationError):
            PredictOut(
                label=1,
                score=-0.5,  # Must be >= 0
                threshold=0.50,
                model_version="AE-2025.11.04-1",
            )


class TestPydanticSerialization:
    """Test Pydantic model serialization."""
    
    def test_anomaly_upsert_dict(self):
        """Test AnomalyUpsert to dict."""
        req = AnomalyUpsert(
            anomaly_id="A1",
            occurred_at="2025-11-04T10:00:00Z",
            source="sensor-1",
            tensor=[1.0, 2.0, 3.0],
        )
        data = req.model_dump()
        assert data["anomaly_id"] == "A1"
        assert data["tensor"] == [1.0, 2.0, 3.0]
    
    def test_predict_out_json(self):
        """Test PredictOut to JSON."""
        result = PredictOut(
            label=1,
            score=0.85,
            threshold=0.50,
            model_version="AE-2025.11.04-1",
        )
        json_str = result.model_dump_json()
        assert "label" in json_str
        assert "0.85" in json_str
    
    def test_feedback_in_from_dict(self):
        """Test FeedbackIn from dict."""
        data = {
            "anomaly_id": "A1",
            "user_id": "user1",
            "label": "TP",
            "confidence": 0.95,
        }
        feedback = FeedbackIn(**data)
        assert feedback.anomaly_id == "A1"
        assert feedback.confidence == 0.95


class TestTypeAnnotations:
    """Test type system works with type checkers."""
    
    def test_typed_dict_usage(self):
        """Test TypedDict can be used with type hints."""
        def process_schema(schema: SchemaInfo) -> int:
            return schema["numel"]
        
        schema: SchemaInfo = {
            "schema_id": "abc",
            "shape": (10,),
            "ndim": 1,
            "numel": 10,
            "dtype": "float32",
        }
        assert process_schema(schema) == 10
    
    def test_pydantic_model_usage(self):
        """Test Pydantic model can be used with type hints."""
        def process_prediction(result: PredictOut) -> bool:
            return result.label == 1
        
        result = PredictOut(
            label=1,
            score=0.85,
            threshold=0.50,
            model_version="AE-2025.11.04-1",
        )
        assert process_prediction(result) is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
