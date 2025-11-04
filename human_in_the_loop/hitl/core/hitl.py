# HITL Orchestrator - Main Public API
#
# This module provides the main HITL class that orchestrates all
# subsystems and exposes the public API for library and CLI usage.
#
# Class to implement:
#
# class HITL:
#   """
#   Human-in-the-Loop anomaly filtering orchestrator.
#   
#   Main entry point for all HITL operations. Coordinates between
#   database, artifacts, training, and inference subsystems.
#   """
#   
#   def __init__(self, config: Config | None = None):
#     """
#     Initialize HITL system.
#     
#     Args:
#       - config: Configuration object. If None, load from environment.
#     
#     Responsibilities:
#       1. Load or create config
#       2. Initialize logger
#       3. Initialize database (SQLite + Repository)
#       4. Initialize schema registry
#       5. Initialize artifacts manager
#       6. Initialize trainer
#       7. Initialize live model (but don't load yet)
#     
#     All subsystems are created and wired together here.
#     """
#     self.config = config or get_env_config()
#     self.logger = get_logger(__name__)
#     
#     # Initialize subsystems
#     # self.db = SQLite(self.config.sqlite_path)
#     # self.repo = Repository(self.db)
#     # self.registry = SchemaRegistry(self.repo)
#     # self.artifacts = Artifacts(self.config.artifacts_dir)
#     # self.trainer = Trainer(self.repo, self.artifacts, self.config, self.logger)
#     # self.live_model = LiveModel(self.repo, self.artifacts, self.config, self.logger)
#   
#   # ===== ANOMALY MANAGEMENT =====
#   
#   def upsert_anomaly(
#     self,
#     anomaly: dict,
#     tensor: np.ndarray | list,
#     dtype: str = "float32"
#   ) -> str:
#     """
#     Insert or update anomaly with feature vector.
#     
#     Args:
#       - anomaly: Dict with metadata
#         Required keys: "anomaly_id", "occurred_at", "source"
#       - tensor: Feature vector as NumPy array or Python list
#       - dtype: Target dtype for storage
#     
#     Returns: anomaly_id
#     
#     Algorithm:
#       1. Extract and validate required fields from anomaly dict
#       2. Convert tensor to NumPy array if needed
#       3. Validate tensor (shape, no NaN/Inf)
#       4. Ensure schema in registry (get schema_id)
#       5. Encode tensor to .npy blob
#       6. Begin transaction:
#          - Upsert anomaly record
#          - Put vector blob
#       7. Log operation
#       8. Return anomaly_id
#     
#     Raises:
#       - ValidationError: if required fields missing or invalid
#       - UnsupportedShape: if tensor shape invalid
#       - DBError: on database errors
#     
#     Example:
#       aid = hitl.upsert_anomaly(
#         anomaly={
#           "anomaly_id": "A1",
#           "occurred_at": "2025-11-04T10:00:00Z",
#           "source": "unit-1"
#         },
#         tensor=np.random.randn(128)
#       )
#     """
#   
#   def get_anomaly(self, anomaly_id: str) -> dict | None:
#     """
#     Retrieve anomaly by ID.
#     
#     Args:
#       - anomaly_id: Anomaly identifier
#     
#     Returns: Anomaly dict with metadata (no tensor)
#     
#     Converts database row to plain dict.
#     """
#   
#   def get_anomaly_with_tensor(self, anomaly_id: str) -> tuple[dict, np.ndarray] | None:
#     """
#     Retrieve anomaly with its feature vector.
#     
#     Args:
#       - anomaly_id: Anomaly identifier
#     
#     Returns: (anomaly_dict, tensor) tuple or None
#     
#     Loads vector from database and decodes.
#     """
#   
#   # ===== FEEDBACK =====
#   
#   def submit_feedback(
#     self,
#     anomaly_id: str,
#     label: str,
#     user_id: str,
#     confidence: float | None = None,
#     note: str | None = None
#   ) -> str:
#     """
#     Submit human feedback on anomaly.
#     
#     Args:
#       - anomaly_id: Anomaly to label
#       - label: Label type (e.g., "TP", "FP", "TN", "FN")
#       - user_id: Who is providing feedback
#       - confidence: Optional confidence score (0.0-1.0)
#       - note: Optional human-readable notes
#     
#     Returns: feedback_id
#     
#     Algorithm:
#       1. Validate anomaly exists
#       2. Validate confidence in range if provided
#       3. Generate feedback_id
#       4. Insert feedback record
#       5. Log operation
#       6. Return feedback_id
#     
#     Raises:
#       - ValidationError: if anomaly_id invalid or confidence out of range
#       - DBError: on database errors
#     
#     Example:
#       fid = hitl.submit_feedback(
#         anomaly_id="A1",
#         label="TP",
#         user_id="analyst-1",
#         confidence=0.9,
#         note="Clear anomaly pattern"
#       )
#     """
#   
#   def get_feedback(self, anomaly_id: str) -> list[dict]:
#     """
#     Get all feedback for an anomaly.
#     
#     Args:
#       - anomaly_id: Anomaly identifier
#     
#     Returns: List of feedback dicts (newest first)
#     """
#   
#   # ===== TRAINING =====
#   
#   def train_model(
#     self,
#     mode: str | None = None,
#     schema_id: str | None = None,
#     params: dict | None = None
#   ) -> str:
#     """
#     Train new anomaly detection model.
#     
#     Args:
#       - mode: "dense" or "conv1d". If None, use config default.
#       - schema_id: Feature schema to train on. If None, auto-detect.
#       - params: Training parameters dict. If None, use defaults.
#     
#     Returns: model_version string
#     
#     Algorithm:
#       1. Determine mode (arg > config)
#       2. Resolve schema_id:
#          - If provided, use it
#          - Else list all schemas
#          - If exactly one, use it
#          - If multiple, raise error (ambiguous)
#          - If none, raise error (no data)
#       3. Validate schema exists and has vectors
#       4. Merge params with defaults
#       5. Delegate to trainer.train_and_publish()
#       6. Log completion
#       7. Return model_version
#     
#     Raises:
#       - SchemaNotFound: if schema_id invalid or ambiguous
#       - ValueError: if no training data available
#       - DBError: on database errors
#     
#     Example:
#       model_version = hitl.train_model(
#         mode="dense",
#         params={"epochs": 100, "lr": 0.001}
#       )
#       print(f"Trained: {model_version}")
#     """
#   
#   def list_models(self, kind: str | None = None) -> list[dict]:
#     """
#     List trained models.
#     
#     Args:
#       - kind: Filter by mode ("dense" or "conv1d")
#     
#     Returns: List of model dicts with metadata
#     """
#   
#   # ===== MODEL MANAGEMENT =====
#   
#   def set_live_model(self, model_version: str) -> None:
#     """
#     Set which model is used for inference.
#     
#     Args:
#       - model_version: Model version to activate
#     
#     Algorithm:
#       1. Validate model_version exists in database
#       2. Validate artifacts exist on filesystem
#       3. Update settings in database
#       4. Clear live model cache (force reload)
#       5. Log activation
#     
#     Raises:
#       - ValueError: if model_version not found
#       - ArtifactMissing: if artifacts not found
#     
#     Example:
#       hitl.set_live_model("AE-2025.11.04-1")
#     """
#   
#   def get_live_model(self) -> str | None:
#     """
#     Get current live model version.
#     
#     Returns: model_version string or None if not set
#     """
#   
#   # ===== INFERENCE =====
#   
#   def filter_predict(
#     self,
#     tensor: np.ndarray | list | None = None,
#     anomaly_id: str | None = None
#   ) -> PredictResult:
#     """
#     Predict whether anomaly is true positive or false positive.
#     
#     Args:
#       - tensor: Feature vector to score (provide this OR anomaly_id)
#       - anomaly_id: Anomaly ID to load and score (provide this OR tensor)
#     
#     Returns: PredictResult dict
#       {
#         "label": 0 or 1,
#         "score": float,
#         "threshold": float,
#         "model_version": str
#       }
#     
#     Algorithm:
#       1. Validate exactly one of tensor/anomaly_id provided
#       2. If anomaly_id:
#          - Load vector from database
#          - Decode to NumPy array
#       3. If tensor is list:
#          - Convert to NumPy array
#       4. Ensure live model is loaded
#       5. Delegate to live_model.predict_tensor()
#       6. Log prediction
#       7. Return result
#     
#     Raises:
#       - ValueError: if neither or both args provided
#       - NoLiveModel: if no live model set
#       - ShapeMismatch: if tensor shape wrong
#     
#     Example:
#       # Direct tensor
#       result = hitl.filter_predict(tensor=np.array([...]))
#       
#       # From database
#       result = hitl.filter_predict(anomaly_id="A1")
#       
#       if result["label"] == 1:
#         print(f"Anomaly confirmed: {result['score']:.4f}")
#     """
#   
#   def batch_predict(self, tensors: list[np.ndarray]) -> list[PredictResult]:
#     """
#     Predict on batch of tensors.
#     
#     Args:
#       - tensors: List of feature vectors
#     
#     Returns: List of PredictResult dicts
#     
#     More efficient than calling filter_predict repeatedly
#     as it keeps model in memory and batches inference.
#     """
#   
#   # ===== UTILITY =====
#   
#   def get_stats(self) -> dict:
#     """
#     Get system statistics.
#     
#     Returns: Dict with:
#       - num_anomalies: Total anomaly count
#       - num_vectors: Total stored vectors
#       - num_schemas: Schema count
#       - num_models: Trained model count
#       - num_feedback: Feedback record count
#       - live_model: Current live model version
#       - schemas: List of schema summaries
#     
#     Useful for dashboards and monitoring.
#     """
#   
#   def health_check(self) -> dict:
#     """
#     Check system health.
#     
#     Returns: Dict with:
#       - status: "ok" or "error"
#       - database: "ok" or error message
#       - artifacts: "ok" or error message
#       - live_model: "loaded" or "not set"
#       - errors: List of error messages if any
#     
#     Useful for API health endpoint.
#     """
#   
#   def close(self) -> None:
#     """
#     Clean shutdown of HITL system.
#     
#     Closes database connections, clears caches.
#     Call this before application exit.
#     """
#
# Helper functions:
#
# def _validate_anomaly_dict(anomaly: dict) -> None:
#   """
#   Validate anomaly metadata dict has required fields.
#   
#   Required: "anomaly_id", "occurred_at", "source"
#   Validate occurred_at is valid ISO timestamp.
#   Raise ValidationError with helpful message if invalid.
#   """
#
# def _merge_train_params(user_params: dict | None, defaults: TrainParams) -> TrainParams:
#   """
#   Merge user-provided params with defaults.
#   
#   User params override defaults.
#   Returns complete TrainParams dict.
#   """
#
# Usage patterns:
#   # Initialize system
#   hitl = HITL()
#   
#   # Ingest anomaly
#   aid = hitl.upsert_anomaly(
#     anomaly={"anomaly_id": "A1", "occurred_at": "...", "source": "unit-1"},
#     tensor=np.random.randn(128)
#   )
#   
#   # Human feedback
#   hitl.submit_feedback("A1", label="TP", user_id="analyst-1")
#   
#   # Train model
#   mv = hitl.train_model(mode="dense")
#   
#   # Activate model
#   hitl.set_live_model(mv)
#   
#   # Predict
#   result = hitl.filter_predict(anomaly_id="A1")
#   print(f"Label: {result['label']}, Score: {result['score']}")
#   
#   # Stats
#   stats = hitl.get_stats()
#   print(f"System has {stats['num_anomalies']} anomalies")
#   
#   # Cleanup
#   hitl.close()
