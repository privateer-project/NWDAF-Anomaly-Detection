# API Request/Response Schemas
#
# This module defines Pydantic models for FastAPI request validation
# and response serialization.
#
# Request Models:
#
# class AnomalyUpsertRequest(BaseModel):
#   """Request model for upserting anomaly."""
#   anomaly_id: str = Field(..., description="Unique anomaly identifier")
#   occurred_at: str = Field(..., description="ISO timestamp when anomaly occurred")
#   source: str = Field(..., description="Origin/unit that detected anomaly")
#   tensor: list[float] | list[list[float]] = Field(..., description="Feature vector")
#   dtype: str = Field(default="float32", description="NumPy dtype")
#
#   # Validators:
#   # - occurred_at must be valid ISO timestamp
#   # - tensor must be 1D or 2D list
#   # - dtype must be valid
#
# class FeedbackSubmitRequest(BaseModel):
#   """Request model for submitting feedback."""
#   anomaly_id: str
#   user_id: str
#   label: str = Field(..., description="Label type: TP, FP, TN, FN, etc.")
#   confidence: float | None = Field(default=None, ge=0.0, le=1.0)
#   note: str | None = None
#
# class TrainModelRequest(BaseModel):
#   """Request model for training."""
#   mode: Literal["dense", "conv1d"] | None = None
#   schema_id: str | None = None
#   params: dict | None = None
#
#   # params can include:
#   # - epochs: int
#   # - batch_size: int
#   # - lr: float
#   # - val_split: float
#   # - patience: int
#   # - percentile: float
#
# class SetLiveModelRequest(BaseModel):
#   """Request model for setting live model."""
#   model_version: str
#
# class PredictRequest(BaseModel):
#   """Request model for prediction."""
#   tensor: list[float] | list[list[float]] | None = None
#   anomaly_id: str | None = None
#
#   # Custom validator: exactly one of tensor/anomaly_id must be provided
#   @model_validator(mode='after')
#   def check_exactly_one(self):
#     if (self.tensor is None) == (self.anomaly_id is None):
#       raise ValueError("Provide exactly one of tensor or anomaly_id")
#     return self
#
# Response Models:
#
# class AnomalyUpsertResponse(BaseModel):
#   """Response for anomaly upsert."""
#   anomaly_id: str
#   message: str = "Anomaly upserted successfully"
#
# class FeedbackSubmitResponse(BaseModel):
#   """Response for feedback submission."""
#   feedback_id: str
#   message: str = "Feedback submitted successfully"
#
# class TrainModelResponse(BaseModel):
#   """Response for model training."""
#   model_version: str
#   message: str = "Model trained successfully"
#   metrics: dict | None = None  # Optional training metrics
#
# class SetLiveModelResponse(BaseModel):
#   """Response for setting live model."""
#   model_version: str
#   message: str = "Live model updated"
#
# class PredictResponse(BaseModel):
#   """Response for prediction."""
#   label: int = Field(..., description="0=normal, 1=anomaly")
#   score: float = Field(..., description="Reconstruction error")
#   threshold: float = Field(..., description="Decision threshold")
#   model_version: str = Field(..., description="Model used for prediction")
#   is_anomaly: bool = Field(..., description="Convenience flag: label == 1")
#
# class AnomalyDetailResponse(BaseModel):
#   """Response for anomaly detail."""
#   anomaly_id: str
#   occurred_at: str
#   source: str
#   created_at: str
#   updated_at: str
#   feedback: list[dict] | None = None
#   has_vector: bool = False
#
# class ModelDetailResponse(BaseModel):
#   """Response for model detail."""
#   model_version: str
#   kind: str
#   artifact_path: str
#   created_at: str
#   is_live: bool = False
#
# class StatsResponse(BaseModel):
#   """Response for system stats."""
#   num_anomalies: int
#   num_vectors: int
#   num_schemas: int
#   num_models: int
#   num_feedback: int
#   live_model: str | None
#   schemas: list[dict]
#
# class HealthResponse(BaseModel):
#   """Response for health check."""
#   status: Literal["ok", "error"]
#   database: str
#   artifacts: str
#   live_model: str
#   errors: list[str] = Field(default_factory=list)
#   timestamp: str
#
# class ErrorResponse(BaseModel):
#   """Standard error response."""
#   error: str
#   detail: str | None = None
#   error_type: str | None = None
#
# Example Usage:
#   from hitl.api.schemas import AnomalyUpsertRequest, PredictResponse
#
#   # In FastAPI endpoint
#   @app.post("/anomalies", response_model=AnomalyUpsertResponse)
#   async def create_anomaly(req: AnomalyUpsertRequest):
#     ...
