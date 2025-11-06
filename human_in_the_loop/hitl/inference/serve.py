# Inference Service - Live Model Loading and Prediction
#
# This module manages the live model for inference, handling loading,
# caching, and prediction on new anomalies.
#
# Class to implement:
#
# class LiveModel:
#   """Manages live model for real-time inference."""
#
#   def __init__(
#     self,
#     repository: Repository,
#     artifacts: Artifacts,
#     config: Config,
#     logger
#   ):
#     """
#     Initialize live model manager.
#
#     Args:
#       - repository: Database repository
#       - artifacts: Artifact manager
#       - config: System configuration
#       - logger: Structured logger
#
#     Caching:
#       - Loads model once and caches in memory
#       - Reloads only when live model version changes
#       - Thread-safe access if needed for web server
#     """
#     self.repo = repository
#     self.artifacts = artifacts
#     self.config = config
#     self.logger = logger
#
#     # Cache state
#     self._cached_version: str | None = None
#     self._model: nn.Module | None = None
#     self._threshold: dict | None = None
#     self._config: dict | None = None
#     self._device: torch.device = get_device()
#
#   def load_live(self) -> None:
#     """
#     Load the current live model into memory.
#
#     Algorithm:
#       1. Get live model version from database
#       2. If version == cached version, skip (already loaded)
#       3. Load all artifacts (model, config, threshold - no scaler)
#       4. Build model architecture from config
#       5. Load state dict into model
#       6. Set model to eval mode
#       7. Move to device
#       8. Cache everything
#
#     Raises:
#       - NoLiveModel: if no live model set in database
#       - ArtifactMissing: if artifacts not found
#
#     Call this at startup or after set_live_model()
#     
#     Note: No scaler needed - inference on raw data
#     """
#
#   def predict_tensor(self, arr: np.ndarray) -> PredictResult:
#     """
#     Predict whether tensor is anomalous.
#
#     Args:
#       - arr: NumPy array in STORAGE format (raw data, no preprocessing):
#           - Dense mode: (D,) - feature vector
#           - Conv1d mode: (T, F) - time-series (time-first)
#
#     Returns: PredictResult dict
#       {
#         "label": 0 or 1 (0=normal, 1=anomaly),
#         "score": float (reconstruction error),
#         "threshold": float,
#         "model_version": str
#       }
#
#     Algorithm:
#       1. Ensure model is loaded (call load_live if needed)
#       2. Validate tensor shape matches model input
#       3. For Conv1d: transpose (T, F) -> (F, T) to match model format
#       4. Add batch dimension: (F, T) -> (1, F, T)
#       5. Convert to PyTorch tensor (raw data, no normalization)
#       6. Forward pass through model
#       7. Compute reconstruction error (MSE)
#       8. Compare to threshold
#       9. Return result
#
#     Shape Convention (Conv1d):
#       Storage format: (T, F) - time-first, human-readable
#       Model format: (B, F, T) - channels-first, PyTorch Conv1d requirement
#       See: docs/SHAPE_CONVENTIONS.md for full details
#
#     Raises:
#       - NoLiveModel: if no live model
#       - ShapeMismatch: if tensor shape wrong
#       - ValidationError: if tensor has NaN/Inf
#     
#     TODO IMPLEMENTATION:
#       if self._config["mode"] == "conv1d" and arr.ndim == 2:
#           # Transform from storage format (T, F) to model format (F, T)
#           arr = arr.T  # (T, F) -> (F, T)
#           self.logger.debug(
#               "Transposed input for Conv1d inference",
#               storage_shape="(T, F)",
#               model_shape="(F, T)"
#           )
#       
#       # Add batch dimension
#       arr = arr[np.newaxis, ...]  # (F, T) -> (1, F, T) for conv1d
#                                    # (D,) -> (1, D) for dense
#       
#       # Forward pass on raw data - no normalization
#       x = torch.FloatTensor(arr).to(self._device)
#       x_hat = self._model(x)
#       
#       # Compute reconstruction error
#       mse = ((x - x_hat) ** 2).mean().item()
#       
#       # Compare to threshold
#       label = int(mse >= self._threshold["value"])
#       
#       return {
#           "label": label,
#           "score": mse,
#           "threshold": self._threshold["value"],
#           "model_version": self._cached_version
#       }
#     """
#
#   def reload_if_changed(self) -> bool:
#     """
#     Check if live model changed and reload if so.
#
#     Returns: True if reloaded, False if unchanged
#
#     Algorithm:
#       1. Get current live model version from DB
#       2. Compare to cached version
#       3. If different, call load_live()
#       4. Return whether reload happened
#
#     Useful for long-running servers to pick up model updates.
#     Can be called periodically or before each prediction.
#     """
#
#   def clear_cache(self) -> None:
#     """
#     Clear cached model from memory.
#
#     Sets all cached attributes to None.
#     Forces reload on next predict_tensor call.
#     Useful for memory management or testing.
#     """
#
#   def get_info(self) -> dict:
#     """
#     Get information about loaded model.
#
#     Returns: Dict with model metadata
#       {
#         "model_version": str,
#         "mode": str,
#         "input_shape": tuple,
#         "threshold": float,
#         "loaded": bool
#       }
#
#     Useful for health checks and debugging.
#     """
#
#   def _compute_reconstruction_error(
#     self,
#     x: torch.Tensor,
#     x_hat: torch.Tensor
#   ) -> float:
#     """
#     Compute reconstruction error for single sample.
#
#     Args:
#       - x: Original tensor
#       - x_hat: Reconstructed tensor
#
#     Returns: Scalar MSE value
#
#     If batch dimension present, compute mean over batch.
#     Return single float score.
#     """
#
# Helper functions:
#
# def mse_per_sample(x: torch.Tensor, x_hat: torch.Tensor, mode: str) -> np.ndarray:
#   """
#   Compute MSE per sample for batch predictions.
#
#   Args:
#     - x: Original tensors (B, ...)
#     - x_hat: Reconstructed tensors (B, ...)
#     - mode: "dense" or "conv1d"
#
#   Returns: NumPy array of shape (B,) with error per sample
#
#   Used during training threshold computation.
#   For inference of single sample, just compute scalar MSE.
#
#   Dense mode (B, D):
#     - Squared error over dim 1
#     - Mean to get scalar per sample
#
#   Conv1d mode (B, F, T):
#     - Squared error over dims (1, 2)
#     - Mean to get scalar per sample
#   """
#
# def get_device() -> torch.device:
#   """Get appropriate device (CUDA if available, else CPU)."""
#
# def validate_input_shape(
#   arr: np.ndarray,
#   expected_shape: tuple[int, ...],
#   allow_batch: bool = False
# ) -> np.ndarray:
#   """
#   Validate and optionally add batch dimension.
#
#   Args:
#     - arr: Input array
#     - expected_shape: Expected shape without batch
#     - allow_batch: If True, accept (B, ...) or (...) shapes
#
#   Returns: Array with batch dimension (B, ...)
#
#   Examples:
#     - Input (128,), expected (128,) → add batch → (1, 128)
#     - Input (5, 128), expected (128,) → already batched → (5, 128)
#     - Input (128,), expected (64,) → raise ShapeMismatch
#   """
#
# Usage patterns:
#   live = LiveModel(repo, artifacts, config, logger)
#
#   # Load model at startup
#   live.load_live()
#
#   # Predict on new tensor
#   arr = np.array([1.0, 2.0, 3.0, ...])  # Feature vector
#   result = live.predict_tensor(arr)
#
#   if result["label"] == 1:
#     print(f"Anomaly detected! Score: {result['score']:.4f}, "
#           f"Threshold: {result['threshold']:.4f}")
#
#   # In web server: periodic reload check
#   if live.reload_if_changed():
#     logger.info("Live model updated")
#
#   # Health check
#   info = live.get_info()
#   print(f"Running model: {info['model_version']}")
