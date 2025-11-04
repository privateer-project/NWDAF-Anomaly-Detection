# FastAPI HTTP Server
#
# This module implements the REST API for HITL system using FastAPI.
# Provides HTTP endpoints for all HITL operations.
#
# Main components:
#
# # Application setup
# app = FastAPI(
#   title="HITL Anomaly Filtering API",
#   description="Human-in-the-Loop anomaly filtering with autoencoder models",
#   version="0.1.0"
# )
#
# # Global state
# # Store HITL instance at app startup for reuse across requests
# # Use app.state or dependency injection
#
# @app.on_event("startup")
# async def startup_event():
#   """
#   Initialize HITL system on server startup.
#   
#   Responsibilities:
#     - Load config
#     - Create HITL instance
#     - Store in app.state
#     - Load live model if available
#     - Log startup info
#   """
#
# @app.on_event("shutdown")
# async def shutdown_event():
#   """
#   Clean shutdown.
#   
#   Close HITL connections.
#   """
#
# # Dependency for getting HITL instance
# def get_hitl() -> HITL:
#   """Dependency to inject HITL instance into endpoints."""
#   return app.state.hitl
#
# # ===== ENDPOINTS =====
#
# @app.get("/health", response_model=HealthResponse)
# async def health_check(hitl: HITL = Depends(get_hitl)):
#   """
#   Health check endpoint.
#   
#   Returns system health status including database,
#   artifacts, and live model state.
#   """
#
# @app.get("/stats", response_model=StatsResponse)
# async def get_stats(hitl: HITL = Depends(get_hitl)):
#   """
#   Get system statistics.
#   
#   Returns counts of anomalies, models, feedback, etc.
#   """
#
# @app.post("/anomalies", response_model=AnomalyUpsertResponse, status_code=201)
# async def upsert_anomaly(
#   req: AnomalyUpsertRequest,
#   hitl: HITL = Depends(get_hitl)
# ):
#   """
#   Insert or update anomaly with feature vector.
#   
#   Request body:
#     {
#       "anomaly_id": "A1",
#       "occurred_at": "2025-11-04T10:00:00Z",
#       "source": "unit-1",
#       "tensor": [1.0, 2.0, ...],
#       "dtype": "float32"
#     }
#   
#   Returns anomaly_id.
#   """
#
# @app.get("/anomalies/{anomaly_id}", response_model=AnomalyDetailResponse)
# async def get_anomaly(
#   anomaly_id: str,
#   hitl: HITL = Depends(get_hitl)
# ):
#   """
#   Get anomaly details by ID.
#   
#   Returns metadata and optionally feedback history.
#   """
#
# @app.post("/feedback", response_model=FeedbackSubmitResponse, status_code=201)
# async def submit_feedback(
#   req: FeedbackSubmitRequest,
#   hitl: HITL = Depends(get_hitl)
# ):
#   """
#   Submit human feedback on anomaly.
#   
#   Request body:
#     {
#       "anomaly_id": "A1",
#       "user_id": "analyst-1",
#       "label": "TP",
#       "confidence": 0.9,
#       "note": "Clear anomaly"
#     }
#   
#   Returns feedback_id.
#   """
#
# @app.post("/train", response_model=TrainModelResponse, status_code=202)
# async def train_model(
#   req: TrainModelRequest,
#   hitl: HITL = Depends(get_hitl)
# ):
#   """
#   Train new anomaly detection model.
#   
#   This is a long-running operation. In production, consider
#   making this async with Celery/RQ or background tasks.
#   
#   Request body:
#     {
#       "mode": "dense",
#       "schema_id": null,  # auto-detect
#       "params": {
#         "epochs": 100,
#         "lr": 0.001
#       }
#     }
#   
#   Returns model_version.
#   """
#
# @app.get("/models", response_model=list[ModelDetailResponse])
# async def list_models(
#   kind: str | None = Query(None, description="Filter by mode: dense or conv1d"),
#   hitl: HITL = Depends(get_hitl)
# ):
#   """
#   List all trained models.
#   
#   Optional query param:
#     - kind: Filter by model type
#   """
#
# @app.post("/models/live", response_model=SetLiveModelResponse)
# async def set_live_model(
#   req: SetLiveModelRequest,
#   hitl: HITL = Depends(get_hitl)
# ):
#   """
#   Set which model is used for inference.
#   
#   Request body:
#     {
#       "model_version": "AE-2025.11.04-1"
#     }
#   """
#
# @app.get("/models/live", response_model=ModelDetailResponse)
# async def get_live_model(hitl: HITL = Depends(get_hitl)):
#   """
#   Get current live model details.
#   """
#
# @app.post("/predict", response_model=PredictResponse)
# async def predict(
#   req: PredictRequest,
#   hitl: HITL = Depends(get_hitl)
# ):
#   """
#   Predict whether anomaly is true positive.
#   
#   Provide either tensor or anomaly_id (not both).
#   
#   Request body (option 1 - direct tensor):
#     {
#       "tensor": [1.0, 2.0, ...]
#     }
#   
#   Request body (option 2 - from database):
#     {
#       "anomaly_id": "A1"
#     }
#   
#   Returns prediction result with label, score, threshold.
#   """
#
# # ===== ERROR HANDLERS =====
#
# @app.exception_handler(NoLiveModel)
# async def no_live_model_handler(request: Request, exc: NoLiveModel):
#   """Handle NoLiveModel exception."""
#   return JSONResponse(
#     status_code=503,
#     content=ErrorResponse(
#       error="No live model",
#       detail=str(exc),
#       error_type="NoLiveModel"
#     ).model_dump()
#   )
#
# @app.exception_handler(SchemaNotFound)
# async def schema_not_found_handler(request: Request, exc: SchemaNotFound):
#   """Handle SchemaNotFound exception."""
#   return JSONResponse(
#     status_code=404,
#     content=ErrorResponse(
#       error="Schema not found",
#       detail=str(exc),
#       error_type="SchemaNotFound"
#     ).model_dump()
#   )
#
# @app.exception_handler(ShapeMismatch)
# async def shape_mismatch_handler(request: Request, exc: ShapeMismatch):
#   """Handle ShapeMismatch exception."""
#   return JSONResponse(
#     status_code=400,
#     content=ErrorResponse(
#       error="Shape mismatch",
#       detail=str(exc),
#       error_type="ShapeMismatch"
#     ).model_dump()
#   )
#
# @app.exception_handler(UnsupportedShape)
# async def unsupported_shape_handler(request: Request, exc: UnsupportedShape):
#   """Handle UnsupportedShape exception."""
#   return JSONResponse(
#     status_code=400,
#     content=ErrorResponse(
#       error="Unsupported shape",
#       detail=str(exc),
#       error_type="UnsupportedShape"
#     ).model_dump()
#   )
#
# @app.exception_handler(ValidationError)
# async def validation_error_handler(request: Request, exc: ValidationError):
#   """Handle ValidationError exception."""
#   return JSONResponse(
#     status_code=400,
#     content=ErrorResponse(
#       error="Validation error",
#       detail=str(exc),
#       error_type="ValidationError"
#     ).model_dump()
#   )
#
# @app.exception_handler(DBError)
# async def db_error_handler(request: Request, exc: DBError):
#   """Handle DBError exception."""
#   return JSONResponse(
#     status_code=500,
#     content=ErrorResponse(
#       error="Database error",
#       detail=str(exc),
#       error_type="DBError"
#     ).model_dump()
#   )
#
# @app.exception_handler(Exception)
# async def generic_exception_handler(request: Request, exc: Exception):
#   """Handle unexpected exceptions."""
#   logger.error("Unexpected error", exc_info=True)
#   return JSONResponse(
#     status_code=500,
#     content=ErrorResponse(
#       error="Internal server error",
#       detail="An unexpected error occurred",
#       error_type=type(exc).__name__
#     ).model_dump()
#   )
#
# # ===== MIDDLEWARE =====
#
# # Optional: Add CORS middleware
# # from fastapi.middleware.cors import CORSMiddleware
# # app.add_middleware(
# #   CORSMiddleware,
# #   allow_origins=["*"],
# #   allow_methods=["*"],
# #   allow_headers=["*"],
# # )
#
# # Optional: Request logging middleware
# # Log request method, path, duration, status code
#
# # ===== MAIN =====
#
# if __name__ == "__main__":
#   """
#   Run server directly (for development).
#   
#   Production: use uvicorn CLI
#     uvicorn hitl.api.server:app --host 0.0.0.0 --port 8000 --workers 1
#   
#   Note: --workers 1 for MVP to avoid SQLite concurrency issues.
#   """
#   import uvicorn
#   uvicorn.run(app, host="0.0.0.0", port=8000)
#
# Usage:
#   # Start server
#   python -m hitl.api.server
#   
#   # Or with uvicorn
#   uvicorn hitl.api.server:app --reload
#   
#   # Test endpoints
#   curl http://localhost:8000/health
#   curl -X POST http://localhost:8000/anomalies \
#     -H "Content-Type: application/json" \
#     -d '{"anomaly_id":"A1","occurred_at":"2025-11-04T10:00:00Z","source":"unit-1","tensor":[1.0,2.0]}'
