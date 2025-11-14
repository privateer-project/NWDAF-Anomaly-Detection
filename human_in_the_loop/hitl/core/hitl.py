"""HITL orchestrator implementation.

Provides a high-level class `HITL` that wires together the database,
schema registry, artifacts manager, trainer and live model. This class
is intended as the primary public API for programmatic use.
"""

from typing import Optional, Any

import numpy as np

from hitl.settings import get_env_config, Config
from hitl.utils.logging import get_logger
from hitl.store.sqlite import SQLite
from hitl.store.repository import Repository
from hitl.schemas.registry import SchemaRegistry
from hitl.artifacts.manager import Artifacts
from hitl.training.trainer import Trainer
from hitl.inference.serve import LiveModel
from hitl.io.serialization import tensor_from_list, encode_npy, decode_npy, validate_tensor
from hitl.utils.time import now_iso
from hitl.types import PredictResult
from hitl.errors import ShapeMismatch


class HITL:
	"""Human-in-the-Loop orchestrator.

	High-level interface that coordinates ingestion, feedback, training
	and inference.
	"""

	def __init__(self, config: Optional[Config] = None, logger: Optional[Any] = None):
		self.config = config or get_env_config()
		self.logger = logger or get_logger(__name__)

		# Initialize subsystems
		self.db = SQLite(self.config.sqlite_path)
		self.repo = Repository(self.db)
		self.registry = SchemaRegistry(self.repo)
		self.artifacts = Artifacts(str(self.config.artifacts_dir))
		self.trainer = Trainer(self.repo, self.artifacts, self.config, self.logger)
		self.live_model = LiveModel(self.repo, self.artifacts, self.config, self.logger)

	# ----------------------- Anomalies ---------------------------------
	def upsert_anomaly(self, anomaly: dict, tensor: np.ndarray | list, dtype: str = "float32") -> str:
		"""Insert or update anomaly and store tensor.

		Args:
			anomaly: dict with required keys: anomaly_id, occurred_at, source
			tensor: numpy array or nested list
			dtype: target dtype string

		Returns:
			anomaly_id
		"""
		# Validation
		if "anomaly_id" not in anomaly or "occurred_at" not in anomaly or "source" not in anomaly:
			raise ValueError("anomaly must include 'anomaly_id', 'occurred_at' and 'source'")

		arr = tensor_from_list(tensor) if not isinstance(tensor, np.ndarray) else tensor
		validate_tensor(arr)

		# Determine schema (storage shape is arr.shape)
		sample_shape = tuple(arr.shape)
		schema_info = self.registry.ensure(sample_shape, dtype)
		schema_id = schema_info["schema_id"]

		anomaly_id = anomaly["anomaly_id"]
		occurred_at = anomaly.get("occurred_at", now_iso())
		created_at = anomaly.get("created_at", occurred_at)
		updated_at = anomaly.get("updated_at", occurred_at)

		# Upsert record
		self.repo.upsert_anomaly(
			anomaly_id=anomaly_id,
			occurred_at=occurred_at,
			source=anomaly["source"],
			schema_id=schema_id,
			created_at=created_at,
			updated_at=updated_at,
		)

		# Store vector blob
		blob = encode_npy(arr.astype(dtype))
		self.repo.put_vector(anomaly_id, schema_id, blob, occurred_at)

		return anomaly_id

	def get_anomaly(self, anomaly_id: str) -> dict | None:
		row = self.repo.get_anomaly(anomaly_id)
		return dict(row) if row else None

	def get_anomaly_with_tensor(self, anomaly_id: str) -> tuple[dict, np.ndarray] | None:
		row = self.repo.get_anomaly(anomaly_id)
		if not row:
			return None
		vec = self.repo.get_vector(anomaly_id)
		if not vec:
			return (dict(row), None)
		schema_id, blob = vec
		arr = decode_npy(blob)
		return (dict(row), arr)

	# ----------------------- Feedback ---------------------------------
	def submit_feedback(self, anomaly_id: str, label: str, user_id: str, confidence: float | None = None, note: str | None = None) -> str:
		fid = f"fb-{now_iso()}-{user_id}"
		created_at = now_iso()
		self.repo.insert_feedback(fid, anomaly_id, user_id, label, confidence, note, created_at)
		return fid

	def get_feedback(self, anomaly_id: str) -> list[dict]:
		rows = self.repo.list_feedback(anomaly_id)
		return [dict(r) for r in rows]

	# ----------------------- Training ---------------------------------
	def train_model(self, mode: Optional[str] = None, schema_id: Optional[str] = None, params: Optional[dict] = None) -> str:
		# Resolve mode and schema
		if mode is None:
			mode = self.config.mode

		if schema_id is None:
			schemas = self.registry.list_all()
			if len(schemas) == 0:
				raise ValueError("No schemas available to train on")
			if len(schemas) > 1:
				raise ValueError("Multiple schemas available; provide schema_id")
			schema_id = schemas[0]["schema_id"]

		train_params = params or {}
		# Ensure mode in params
		train_params.setdefault("mode", mode)

		model_version = self.trainer.train_and_publish(schema_id, train_params)
		return model_version

	def set_live_model(self, model_version: str) -> None:
		self.repo.set_live_model(model_version)

	def get_live_model(self) -> str | None:
		return self.repo.get_live_model()

	# ----------------------- Inference --------------------------------
	def filter_predict(self, tensor: np.ndarray | list | None = None, anomaly_id: str | None = None) -> PredictResult:
		if anomaly_id:
			res = self.get_anomaly_with_tensor(anomaly_id)
			if res is None:
				raise ValueError(f"Anomaly not found: {anomaly_id}")
			_, arr = res
		else:
			arr = tensor_from_list(tensor) if not isinstance(tensor, np.ndarray) else tensor

		return self.live_model.predict_tensor(arr)

	def close(self) -> None:
		# No explicit close required for SQLite wrapper
		self.logger.info("HITL shutdown")
