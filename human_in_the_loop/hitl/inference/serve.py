# Inference Service - Live Model Loading and Prediction
#
# This module manages the live model for inference, handling loading,
# caching, and prediction on new anomalies.
#
# Class to implement:
#
from typing import Optional

import numpy as np
import torch

from hitl.store.repository import Repository
from hitl.artifacts.manager import Artifacts
from hitl.settings import Config
from hitl.errors import NoLiveModel, ArtifactMissing
from hitl.models.ae import build_model
from hitl.training.trainer import get_device
from hitl.io.serialization import validate_tensor
from hitl.types import PredictResult


class LiveModel:
	"""Manages live model for real-time inference."""

	def __init__(
		self,
		repository: Repository,
		artifacts: Artifacts,
		config: Config,
		logger,
	):
		self.repo = repository
		self.artifacts = artifacts
		self.config = config
		self.logger = logger

		self._cached_version: Optional[str] = None
		self._model: Optional[torch.nn.Module] = None
		self._threshold: Optional[dict] = None
		self._config: Optional[dict] = None
		self._device: torch.device = get_device()

	def load_live(self) -> None:
		"""Load the current live model into memory (if changed)."""
		version = self.repo.get_live_model()
		if not version:
			raise NoLiveModel("No live model set")

		if version == self._cached_version and self._model is not None:
			# already loaded
			return

		# Load artifacts
		self.logger.info(f"Loading live model artifacts: {version}")
		all_art = self.artifacts.load_all(version)
		cfg = all_art["config"]
		state = all_art["model"]
		thresh = all_art["threshold"]

		# Build model from config
		mode = cfg.get("mode")
		input_shape = tuple(cfg.get("input_shape"))

		model = build_model(mode, input_shape)
		model.load_state_dict(state)
		model.to(self._device)
		model.eval()

		self._cached_version = version
		self._model = model
		self._threshold = thresh
		self._config = cfg

		self.logger.info(f"Live model {version} loaded on {self._device}")

	def predict_tensor(self, arr: np.ndarray) -> PredictResult:
		"""Predict whether tensor is anomalous.

		`arr` must be in STORAGE format:
		  - Dense: (D,)
		  - Conv1d: (T, F)
		"""
		if self._model is None or self._cached_version is None:
			# Try to load
			self.load_live()

		validate_tensor(arr)

		cfg = self._config
		if cfg is None:
			raise ArtifactMissing("Model config missing after load")

		mode = cfg["mode"]
		input_shape = tuple(cfg["input_shape"])

		# Convert storage -> model format
		if mode == "conv1d":
			if arr.ndim != 2:
				raise ValueError("Conv1d prediction expects 2D array (T, F)")
			# transpose (T, F) -> (F, T)
			arr_proc = np.transpose(arr, (1, 0))
			tensor = torch.from_numpy(arr_proc).float().unsqueeze(0).to(self._device)
		else:
			# dense
			if arr.ndim != 1:
				raise ValueError("Dense prediction expects 1D vector")
			tensor = torch.from_numpy(arr).float().unsqueeze(0).to(self._device)

		with torch.no_grad():
			recon = self._model(tensor)
			mse = ((tensor - recon) ** 2).mean().item()

		threshold_value = self._threshold["value"]
		label = 1 if mse > threshold_value else 0

		return {
			"label": int(label),
			"score": float(mse),
			"threshold": float(threshold_value),
			"model_version": self._cached_version,
		}
