"""
SHAP time-series explainer utilities.

This module provides a high-level wrapper around SHAP's KernelExplainer for
time-series models implemented in PyTorch. It handles input validation,
reshaping between 3D tensors and 2D matrices expected by KernelExplainer,
and exposes a clean API to compute SHAP values for given instances.
"""

from typing import Any, Callable, Dict, Optional, Tuple

import logging
import numpy as np
import shap
import torch


logger = logging.getLogger(__name__)


class ShapTimeSeries:
    """SHAP KernelExplainer wrapper for PyTorch time-series models.

    The class flattens 3D time-series inputs with shape
    (batch_size, sequence_length, num_features) to 2D matrices required by
    SHAP's KernelExplainer and reverses the reshape inside the prediction
    function used by SHAP during value estimation.

    Parameters
    ----------
    model:
        A PyTorch model. The model is expected to accept input tensors of
        shape (batch_size, sequence_length, num_features) and return tensors
        with the same shape (e.g., autoencoder reconstruction). The class
        computes a per-instance MSE between input and output for explanation.
    dataset:
        A 3D `torch.Tensor` background dataset used by KernelExplainer to
        estimate feature contributions. Shape must be
        (num_instances, sequence_length, num_features).
    sequence_length:
        Optional explicit sequence length. If omitted, inferred from `dataset`.
    num_features:
        Optional explicit feature count. If omitted, inferred from `dataset`.
    background_size:
        Optional cap on the number of background instances for the explainer.
        If provided and lower than `dataset` size, a subset is used for speed.
    device:
        Optional torch device. If omitted, uses the model's device if it can be
        inferred, otherwise CPU.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        dataset: torch.Tensor,
        sequence_length: Optional[int] = None,
        num_features: Optional[int] = None,
        background_size: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.model: torch.nn.Module = model
        self.model.eval()

        self.initial_dataset: torch.Tensor = self._validate_and_prepare_dataset(dataset)
        inferred_seq_len, inferred_num_features = self._infer_time_series_shape(self.initial_dataset)

        self.sequence_length: int = sequence_length or inferred_seq_len
        self.num_features: int = num_features or inferred_num_features

        if device is None:
            try:
                # Attempt to infer device from model parameters
                device = next(self.model.parameters()).device  # type: ignore[assignment]
            except StopIteration:
                device = torch.device("cpu")
        self.device: torch.device = device

        self.explainer: Optional[shap.KernelExplainer] = None
        self._init_explainer(background_size=background_size)

    def _validate_and_prepare_dataset(self, dataset: torch.Tensor) -> torch.Tensor:
        """Validate dataset type and dimensionality and detach to CPU.

        Returns a detached, float32 tensor on CPU.
        """
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
        """Infer sequence length and number of features from a 3D tensor."""
        _, sequence_length, num_features = data.shape
        return int(sequence_length), int(num_features)

    def _reshape_to_kernel(self, tensor_3d: torch.Tensor) -> np.ndarray:
        """Flatten (N, L, F) tensor to (N, L*F) NumPy array for KernelExplainer."""
        if tensor_3d.ndim != 3:
            raise ValueError(
                f"Expected 3D tensor (batch, seq_len, features), got shape {tuple(tensor_3d.shape)}"
            )
        n, l, f = tensor_3d.shape
        return tensor_3d.cpu().numpy().reshape(n, l * f)

    def _init_explainer(self, background_size: Optional[int] = None) -> None:
        """Initialize SHAP KernelExplainer with optional background subsampling."""
        background = self.initial_dataset
        if background_size is not None and background_size < background.shape[0]:
            logger.info(
                "Subsampling background for SHAP: using %d of %d instances",
                background_size,
                background.shape[0],
            )
            background = background[:background_size]

        background_np = self._reshape_to_kernel(background)
        self.explainer = shap.KernelExplainer(self._model_predict_fn(), background_np)

    def _model_predict_fn(self) -> Callable[[np.ndarray], np.ndarray]:
        """Wrap model forward pass into a function compatible with KernelExplainer.

        The returned function expects a 2D NumPy array with shape (N, L*F),
        reshapes it to (N, L, F), performs a forward pass, and returns a 1D
        NumPy array of per-instance scalar scores computed as the mean squared
        error (MSE) between input and model output over sequence_length and
        num_features dimensions.
        """

        def predict(x_flat: np.ndarray) -> np.ndarray:
            if x_flat.ndim != 2:
                raise ValueError(f"Expected 2D array for KernelExplainer, got shape {x_flat.shape}")

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

            mse_per_instance = ((x_tensor - output) ** 2).mean(dim=(1, 2))
            return mse_per_instance.detach().cpu().numpy()

        return predict

    def shap_values_from_instance(self, instance: torch.Tensor) -> Dict[str, Any]:
        """Compute SHAP values for one or more instances.

        Parameters
        ----------
        instance:
            A 3D tensor of shape (batch, sequence_length, num_features). If a
            single instance is provided as 2D (sequence_length, num_features),
            a batch dimension will be added automatically.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing SHAP values array under key `shap_values`,
            the original input tensor under key `x`, and the `explainer` used.
        """
        if self.explainer is None:
            raise RuntimeError("Explainer is not initialized")

        if not isinstance(instance, torch.Tensor):
            raise TypeError("instance must be a torch.Tensor")

        if instance.ndim == 2:
            instance = instance.unsqueeze(0)
        elif instance.ndim != 3:
            raise ValueError(
                f"instance must be 2D or 3D tensor, got shape {tuple(instance.shape)}"
            )

        if (instance.shape[1], instance.shape[2]) != (self.sequence_length, self.num_features):
            raise ValueError(
                "instance shape does not match expected (sequence_length, num_features): "
                f"expected ({self.sequence_length}, {self.num_features}), got "
                f"({instance.shape[1]}, {instance.shape[2]})"
            )

        input_kernel_explainer = self._reshape_to_kernel(instance.detach().to(torch.float32).cpu())
        shap_values = self.explainer.shap_values(input_kernel_explainer)

        if isinstance(shap_values, list):
            logger.info("KernelExplainer returned list, using shap_values[0]")
            shap_values = shap_values[0]

        return {
            "shap_values": shap_values,
            "x": instance,
            "explainer": self.explainer,
        }