"""
Type definitions for HITL system.

This module defines type aliases, TypedDicts, and Pydantic models used throughout
the HITL system for type safety and API validation.

TypedDicts:
    - SchemaInfo: Feature schema metadata
    - TrainParams: Training configuration
    - PredictResult: Prediction output
    - AnomalyRecord: Anomaly metadata
    - FeedbackRecord: Human feedback record

Pydantic Models:
    - AnomalyUpsert: API request for upserting anomaly
    - FeedbackIn: API request for submitting feedback
    - TrainRequest: API request for training
    - PredictIn: API request for prediction
    - PredictOut: API response for prediction

Type Aliases:
    - Tensor1D, Tensor2D: NumPy array types
"""

from typing import Literal, Protocol, TypedDict, Any
from pydantic import BaseModel, Field, field_validator, model_validator

import numpy as np


# =============================================================================
# TypedDicts (for internal use, lightweight dictionaries with type hints)
# =============================================================================


class SchemaInfo(TypedDict):
    """
    Feature schema metadata.

    Describes the shape and dtype of tensors associated with a schema.

    Fields:
        schema_id: Unique identifier (SHA-1 hash of shape+dtype)
        shape: Tensor dimensions tuple (e.g., (128,) or (8, 128))
        ndim: Number of dimensions (1 or 2)
        numel: Total number of elements
        dtype: NumPy dtype string (e.g., "float32")
    """

    schema_id: str
    shape: tuple[int, ...]
    ndim: int
    numel: int
    dtype: str


class TrainParams(TypedDict, total=False):
    """
    Training configuration parameters.

    All fields are optional (total=False) and will use defaults if not provided.

    Fields:
        mode: Model architecture ("dense" or "conv1d")
        epochs: Number of training epochs (default: 100)
        batch_size: Batch size for training (default: 32)
        lr: Learning rate (default: 0.001)
        val_split: Validation data fraction (default: 0.1)
        patience: Early stopping patience (default: 10)
        percentile: Anomaly threshold percentile (default: 99.5)
    """

    mode: Literal["dense", "conv1d"]
    epochs: int
    batch_size: int
    lr: float
    val_split: float
    patience: int
    percentile: float


class PredictResult(TypedDict):
    """
    Prediction result.

    Fields:
        label: Binary label (0=normal, 1=anomaly)
        score: Reconstruction error score
        threshold: Decision threshold used
        model_version: Model version that made prediction
    """

    label: int
    score: float
    threshold: float
    model_version: str


class AnomalyRecord(TypedDict):
    """
    Anomaly metadata record.

    Fields:
        anomaly_id: Unique anomaly identifier
        occurred_at: When anomaly occurred (ISO 8601)
        source: Source system/sensor identifier
        created_at: When record was created (ISO 8601)
        updated_at: When record was last updated (ISO 8601)
    """

    anomaly_id: str
    occurred_at: str
    source: str
    created_at: str
    updated_at: str


class FeedbackRecord(TypedDict):
    """
    Human feedback record.

    Fields:
        feedback_id: Unique feedback identifier
        anomaly_id: Associated anomaly ID
        user_id: User who provided feedback
        label: Feedback label (e.g., "TP", "FP", "TN", "FN")
        confidence: Confidence score (0.0-1.0) or None
        note: Optional text note
        created_at: When feedback was submitted (ISO 8601)
    """

    feedback_id: str
    anomaly_id: str
    user_id: str
    label: str
    confidence: float | None
    note: str | None
    created_at: str


# =============================================================================
# Pydantic Models (for API validation with automatic validation)
# =============================================================================


class AnomalyUpsert(BaseModel):
    """
    Request model for upserting an anomaly.

    Attributes:
        anomaly_id: Unique identifier for the anomaly
        occurred_at: ISO 8601 timestamp when anomaly occurred
        source: Source system or sensor identifier
        tensor: 1D list (for dense) or 2D nested list (for conv1d)
        dtype: NumPy dtype string (default: "float32")

    Example:
        >>> req = AnomalyUpsert(
        ...     anomaly_id="A1",
        ...     occurred_at="2025-11-04T10:00:00Z",
        ...     source="sensor-1",
        ...     tensor=[1.0, 2.0, 3.0],
        ... )
    """

    anomaly_id: str = Field(..., min_length=1, description="Unique anomaly identifier")
    occurred_at: str = Field(..., description="ISO 8601 timestamp")
    source: str = Field(..., min_length=1, description="Source identifier")
    tensor: list[float] | list[list[float]] = Field(..., description="Feature tensor")
    dtype: str = Field("float32", description="NumPy dtype")

    @field_validator("tensor")
    @classmethod
    def validate_tensor(
        cls, v: list[float] | list[list[float]]
    ) -> list[float] | list[list[float]]:
        """Validate tensor is non-empty."""
        if not v:
            raise ValueError("Tensor cannot be empty")
        if isinstance(v[0], list):
            # 2D tensor - validate all rows have same length
            first_row = v[0]
            first_len = len(first_row) if isinstance(first_row, list) else 0
            for row in v:
                if isinstance(row, list) and len(row) != first_len:
                    raise ValueError("All rows in 2D tensor must have same length")
        return v


class FeedbackIn(BaseModel):
    """
    Request model for submitting feedback.

    Attributes:
        anomaly_id: ID of anomaly being labeled
        user_id: ID of user providing feedback
        label: Feedback label (e.g., "TP", "FP", "TN", "FN")
        confidence: Optional confidence score (0.0-1.0)
        note: Optional text note

    Example:
        >>> feedback = FeedbackIn(
        ...     anomaly_id="A1",
        ...     user_id="analyst-1",
        ...     label="TP",
        ...     confidence=0.95,
        ... )
    """

    anomaly_id: str = Field(..., min_length=1, description="Anomaly identifier")
    user_id: str = Field(..., min_length=1, description="User identifier")
    label: str = Field(..., min_length=1, description="Feedback label")
    confidence: float | None = Field(
        None, ge=0.0, le=1.0, description="Confidence score"
    )
    note: str | None = Field(None, description="Optional note")


class TrainRequest(BaseModel):
    """
    Request model for training a model.

    Attributes:
        mode: Model architecture ("dense" or "conv1d")
        schema_id: Optional schema ID to train on (if None, use all data)
        params: Optional training parameters (overrides defaults)

    Example:
        >>> req = TrainRequest(
        ...     mode="dense",
        ...     params={"epochs": 50, "batch_size": 64},
        ... )
    """

    mode: Literal["dense", "conv1d"] = Field(..., description="Model architecture")
    schema_id: str | None = Field(None, description="Schema ID filter")
    params: dict[str, Any] | None = Field(None, description="Training parameters")


class PredictIn(BaseModel):
    """
    Request model for prediction.

    Must provide exactly one of: tensor or anomaly_id.

    Attributes:
        tensor: Feature tensor (1D or 2D nested list)
        anomaly_id: ID of existing anomaly to predict on

    Example:
        >>> # Predict on new tensor
        >>> req1 = PredictIn(tensor=[1.0, 2.0, 3.0])
        >>> # Predict on existing anomaly
        >>> req2 = PredictIn(anomaly_id="A1")
    """

    tensor: list[float] | list[list[float]] | None = Field(
        None, description="Feature tensor"
    )
    anomaly_id: str | None = Field(None, description="Existing anomaly ID")

    @model_validator(mode="after")
    def validate_exactly_one(self) -> "PredictIn":
        """Ensure exactly one of tensor or anomaly_id is provided."""
        has_tensor = self.tensor is not None
        has_anomaly_id = self.anomaly_id is not None

        if not (has_tensor or has_anomaly_id):
            raise ValueError("Must provide either 'tensor' or 'anomaly_id'")

        if has_tensor and has_anomaly_id:
            raise ValueError("Cannot provide both 'tensor' and 'anomaly_id'")

        return self


class PredictOut(BaseModel):
    """
    Response model for prediction.

    Attributes:
        label: Binary label (0=normal, 1=anomaly)
        score: Reconstruction error score
        threshold: Decision threshold used
        model_version: Model version that made prediction

    Example:
        >>> result = PredictOut(
        ...     label=1,
        ...     score=0.85,
        ...     threshold=0.50,
        ...     model_version="AE-2025.11.04-1",
        ... )
    """

    label: int = Field(..., ge=0, le=1, description="Binary label")
    score: float = Field(..., ge=0.0, description="Reconstruction error")
    threshold: float = Field(..., ge=0.0, description="Decision threshold")
    model_version: str = Field(..., min_length=1, description="Model version")


# =============================================================================
# Protocols (structural typing for interfaces)
# =============================================================================


class RepositoryProtocol(Protocol):
    """
    Protocol for repository implementations.

    This defines the interface that any repository implementation must follow.
    Used for dependency injection and testing with mocks.
    """

    def upsert_anomaly(
        self,
        anomaly_id: str,
        occurred_at: str,
        source: str,
        schema_id: str,
    ) -> None:
        """Insert or update anomaly metadata."""
        ...

    def get_anomaly(self, anomaly_id: str) -> AnomalyRecord | None:
        """Retrieve anomaly metadata."""
        ...

    def insert_feedback(
        self,
        feedback_id: str,
        anomaly_id: str,
        user_id: str,
        label: str,
        confidence: float | None,
        note: str | None,
    ) -> None:
        """Insert feedback record."""
        ...

    def get_schema(self, schema_id: str) -> SchemaInfo | None:
        """Retrieve schema by ID."""
        ...

    def put_vector(self, anomaly_id: str, blob: bytes) -> None:
        """Store tensor blob."""
        ...

    def get_vector(self, anomaly_id: str) -> bytes | None:
        """Retrieve tensor blob."""
        ...


# =============================================================================
# Type Aliases
# =============================================================================


# NumPy array type aliases
Tensor1D = np.ndarray  # Shape: (D,) - 1D vector
Tensor2D = np.ndarray  # Shape: (T, F) - 2D time series
