"""
LIME time-series explainer utilities.

This module wraps LIME's LimeTabularExplainer for time-series models. It
provides utilities to flatten 3D inputs for LIME, configure feature names, and
generate local explanations for individual sequences using a PyTorch model.
"""

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import logging
import numpy as np
import torch
from lime import lime_tabular
from app.config import XAI_FEATURE_NAMES


logger = logging.getLogger(__name__)


class LimeInTimeSeries:

    """LIME wrapper for PyTorch time-series models.

    Parameters
    ----------
    model:
        PyTorch model that accepts input tensors of shape
        (batch_size, sequence_length, num_features) and outputs tensors with the
        same shape (e.g., autoencoder reconstruction). The class computes a
        per-instance MSE as the scalar to explain.
    dataset:
        Background dataset (3D torch.Tensor) used to fit the LIME explainer.
        Shape: (num_instances, sequence_length, num_features).
    output:
        Optional opaque object with model output context from the caller.
    mode:
        LIME mode. Either "regression" or "classification". Default: "regression".
    sequence_length, num_features:
        Optional explicit shape parameters; inferred from `dataset` if omitted.
    feature_columns:
        Optional iterable of base feature names (length must equal `num_features`).
        If omitted, a default list of 8 feature names is used.
    discretize_continuous:
        Whether to discretize continuous features in LIME. Default: False.
    kernel_width:
        Optional LIME kernel width.
    background_size:
        Optional cap on number of background instances used to fit explainer.
    device:
        Optional torch device. Defaults to model's device or CPU.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        dataset: torch.Tensor,
        output: Optional[Any] = None,
        mode: str = "regression",
        sequence_length: Optional[int] = None,
        num_features: Optional[int] = None,
        feature_columns: Optional[Sequence[str]] = None,
        discretize_continuous: bool = False,
        kernel_width: Optional[float] = None,
        background_size: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.model: torch.nn.Module = model
        self.model.eval()

        self.initial_dataset: torch.Tensor = self._validate_and_prepare_dataset(dataset)
        inferred_seq_len, inferred_num_features = self._infer_time_series_shape(self.initial_dataset)

        self.sequence_length: int = sequence_length or inferred_seq_len
        self.num_features: int = num_features or inferred_num_features

        if mode not in {"regression", "classification"}:
            raise ValueError("mode must be 'regression' or 'classification'")
        self.mode: str = mode

        if feature_columns is None:
            feature_columns = XAI_FEATURE_NAMES
        if len(feature_columns) != self.num_features:
            raise ValueError(
                "feature_columns length must equal num_features: "
                f"expected {self.num_features}, got {len(feature_columns)}"
            )

        self.feature_names: List[str] = self._generate_feature_names(feature_columns)

        if device is None:
            try:
                device = next(self.model.parameters()).device  # type: ignore[assignment]
            except StopIteration:
                device = torch.device("cpu")
        self.device: torch.device = device

        self.explainer: Optional[lime_tabular.LimeTabularExplainer] = None

        self._init_lime(
            discretize_continuous=discretize_continuous,
            kernel_width=kernel_width,
            background_size=background_size,
        )

        self.output: Optional[Any] = output

    def _validate_and_prepare_dataset(self, dataset: torch.Tensor) -> torch.Tensor:
        if not isinstance(dataset, torch.Tensor):
            raise TypeError("dataset must be a torch.Tensor")
        if dataset.ndim != 3:
            raise ValueError(
                f"dataset must be 3D (batch, seq_len, features), got shape {tuple(dataset.shape)}"
            )
        if dataset.numel() == 0:
            raise ValueError("dataset is empty")
        return dataset.detach().to(torch.float32).cpu()

    def _infer_time_series_shape(self, data: torch.Tensor) -> Tuple[int, int]:
        _, sequence_length, num_features = data.shape
        return int(sequence_length), int(num_features)

    def _reshape_to_tabular(self, tensor_3d: torch.Tensor) -> np.ndarray:
        if tensor_3d.ndim != 3:
            raise ValueError(
                f"Expected 3D tensor (batch, seq_len, features), got shape {tuple(tensor_3d.shape)}"
            )
        n, l, f = tensor_3d.shape
        return tensor_3d.cpu().numpy().reshape(n, l * f)

    def _generate_feature_names(self, feature_columns: Sequence[str]) -> List[str]:
        
        feature_columns = [
            "dl_bitrate",
            "dl_retx",
            "dl_tx",
            "ul_bitrate",
            "ul_mcs",
            "ul_retx",
            "ul_tx",
            "turbo_decoder_avg"
        ]

        # Generate time-based feature names, e.g., dl_bitrate_0, dl_bitrate_1, ..., dl_bitrate_11
        feature_names = []
        for i in range(12):
            feature_names += [f"{item}_{i}" for item in feature_columns]
        return feature_names    
        # names: List[str] = []
        # Define base feature names
        # for t in range(self.sequence_length):
        #     names.extend([f"{col}_{t}" for col in feature_columns])
        # return names

    def _init_lime(
        self,
        discretize_continuous: bool,
        kernel_width: Optional[float],
        background_size: Optional[int],
    ) -> None:
        background = self.initial_dataset
        if background_size is not None and background_size < background.shape[0]:
            logger.info(
                "Subsampling background for LIME: using %d of %d instances",
                background_size,
                background.shape[0],
            )
            background = background[:background_size]

        training_data = self._reshape_to_tabular(background)
        self.explainer = lime_tabular.LimeTabularExplainer(
            training_data=training_data,
            feature_names=self.feature_names,
            mode=self.mode,
            discretize_continuous=discretize_continuous,
            kernel_width=kernel_width,
        )

    def _model_predict_fn(self):
        def predict(x_flat: np.ndarray) -> np.ndarray:
            if x_flat.ndim != 2:
                raise ValueError(f"Expected 2D array, got shape {x_flat.shape}")
            batch_size = x_flat.shape[0]
            x_tensor = torch.from_numpy(x_flat.astype(np.float32, copy=False))
            x_tensor = x_tensor.reshape(batch_size, self.sequence_length, self.num_features)
            x_tensor = x_tensor.to(self.device)
            with torch.no_grad():
                output = self.model(x_tensor)
            if output.shape != x_tensor.shape:
                raise ValueError(
                    "Model output shape must match input shape to compute per-instance MSE. "
                    f"Got input {tuple(x_tensor.shape)}, output {tuple(output.shape)}"
                )
            mse_per_instance = ((x_tensor - output) ** 2).mean(axis=1).mean(axis=1)
            return mse_per_instance.detach().cpu().numpy()

        return predict

    def lime_values_from_instance(self, instance: torch.Tensor, num_features: int = 8) -> Dict[str, Any]:
        """Compute LIME explanation for one instance or a small batch.

        Parameters
        ----------
        instance:
            2D or 3D tensor. Accepts (sequence_length, num_features) or
            (1, sequence_length, num_features). If a batch with N>1 is provided,
            the first instance is explained.
        num_features:
            Number of features to include in the local explanation.

        Returns
        -------
        Dict[str, Any]
            A dictionary with keys: `lime_values`, `x`, `explanation`,
            `feature_names`, `explainer`, `model_output`.
        """
        if self.explainer is None:
            raise RuntimeError("Explainer is not initialized")

        if not isinstance(instance, torch.Tensor):
            raise TypeError("instance must be a torch.Tensor")

        if instance.ndim == 3:
            if instance.shape[0] > 1:
                logger.info("Batch provided with size %d; using the first instance for LIME", instance.shape[0])
            instance = instance[0]
        elif instance.ndim != 2:
            raise ValueError(
                f"instance must be 2D or 3D tensor, got shape {tuple(instance.shape)}"
            )

        if instance.shape != (self.sequence_length, self.num_features):
            raise ValueError(
                "instance shape does not match (sequence_length, num_features): "
                f"expected ({self.sequence_length}, {self.num_features}), got {tuple(instance.shape)}"
            )

        instance_flat: np.ndarray = instance.detach().to(torch.float32).cpu().numpy().reshape(-1)

        explanation = self.explainer.explain_instance(
            data_row=instance_flat,
            predict_fn=self._model_predict_fn(),
            num_features=num_features,
        )

        lime_values = None
        # try:
        #     # `local_exp` is a dict: key is class index (classification) or 0 (regression)
        #     local_exp = getattr(explanation, "local_exp", None)
        #     if isinstance(local_exp, dict) and len(local_exp) > 0:
        #         first_key = sorted(local_exp.keys())[0]
        #         lime_values = local_exp[first_key]
        #     elif hasattr(explanation, "as_map"):
        #         maps = explanation.as_map()
        #         if isinstance(maps, dict) and len(maps) > 0:
        #             first_key = sorted(maps.keys())[0]
        #             lime_values = maps[first_key]
        # except Exception as exc:  # noqa: BLE001
        #     logger.exception("Failed to parse LIME explanation: %s", exc)

        return {
            "lime_values": explanation,
            "x": instance,
            "explanation": explanation,
            "feature_names": self.feature_names,
            "explainer": self.explainer,
            "model_output": self.output,
        }