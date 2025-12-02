"""
Autoencoder Model Architectures

This module defines PyTorch autoencoder architectures for anomaly detection.
Two modes: DenseAE for 1D vectors, Conv1dAE for 2D time-series.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal

class HITLError(Exception):
    """
    Base exception for all HITL-specific errors.

    All custom exceptions in the HITL system inherit from this base class,
    making it easy to catch all HITL-related errors.

    Example:
        >>> try:
        ...     raise HITLError("Something went wrong")
        ... except HITLError as e:
        ...     print(f"HITL error: {e}")
        HITL error: Something went wrong
    """

    pass


class UnsupportedShape(HITLError):
    """
    Raised when tensor has invalid shape (not 1D or 2D).

    The HITL system only supports:
        * 1D vectors for "dense" models (shape: (n,))
        * 2D tensors for "conv1d" models (shape: (features, timesteps))

    Example:
        >>> raise UnsupportedShape("Expected 1D or 2D, got shape (2, 3, 4)")
        Traceback (most recent call last):
        ...
        hitl.errors.UnsupportedShape: Expected 1D or 2D, got shape (2, 3, 4)
    """

    pass



class DenseAE(nn.Module):
    """Dense/Fully-connected autoencoder for 1D feature vectors."""

    def __init__(
        self, input_dim: int, latent_dim: int = 32, hidden_dims: list[int] | None = None
    ):
        """
        Initialize dense autoencoder.

        Args:
            input_dim: Size of input vector (D)
            latent_dim: Size of latent/bottleneck dimension
            hidden_dims: List of hidden layer sizes for encoder
                        If None, use default: [input_dim//2, input_dim//4]

        Architecture:
            Encoder: Linear layers with ReLU activations
                input_dim → hidden[0] → hidden[1] → ... → latent_dim

            Decoder: Mirror of encoder
                latent_dim → hidden[-1] → hidden[-2] → ... → input_dim

        Example with input_dim=128, latent_dim=32:
            Encoder: 128 → 64 → 32 (latent)
            Decoder: 32 → 64 → 128 (reconstruction)

        Layers use:
            - nn.Linear for transformations
            - nn.ReLU for activations (except final layer)
            - No activation on final decoder layer (reconstruction)
        """
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Default hidden dims if not provided
        if hidden_dims is None:
            hidden_dims = [input_dim // 2, input_dim // 4, input_dim // 8]

        self.hidden_dims = hidden_dims

        # Build encoder
        encoder_layers = []
        in_dim = input_dim

        for hidden_dim in hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, hidden_dim))
            encoder_layers.append(nn.ReLU())
            in_dim = hidden_dim

        # Final encoder layer to latent
        encoder_layers.append(nn.Linear(in_dim, latent_dim))

        self.encoder = nn.Sequential(*encoder_layers)

        # Build decoder (mirror of encoder)
        decoder_layers = []

        # Start from latent
        decoder_layers.append(nn.Linear(latent_dim, hidden_dims[-1]))
        decoder_layers.append(nn.ReLU())

        # Reverse through hidden dims
        for i in range(len(hidden_dims) - 1, 0, -1):
            decoder_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i - 1]))
            decoder_layers.append(nn.ReLU())

        # Final layer back to input_dim (no activation)
        decoder_layers.append(nn.Linear(hidden_dims[0], input_dim))

        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through autoencoder.

        Args:
            x: Input tensor of shape (B, D) where B=batch, D=input_dim

        Returns:
            Reconstructed tensor of shape (B, D)

        Algorithm:
            1. Pass through encoder: latent = encoder(x)
            2. Pass through decoder: x_hat = decoder(latent)
            3. Return x_hat
        """
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Get latent representation."""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstruct from latent."""
        return self.decoder(z)


def build_model(
    mode: Literal["dense", "conv1d"],
    input_shape: tuple[int, ...],
    latent_dim: int = 16,
    **kwargs,
) -> nn.Module:
    """Factory function to build appropriate autoencoder based on mode.
    
    Args:
        mode: "dense" for 1D vectors, "conv1d" for 2D time-series
        input_shape: Shape tuple from STORAGE format (not model format)
            - Dense: (D,) where D = number of features
            - Conv1d: (T, F) where T = timesteps, F = features
        latent_dim: Size of latent/bottleneck dimension
        **kwargs: Additional model-specific parameters
    
    Returns:
        Initialized autoencoder model (DenseAE or Conv1dAE)
    
    Shape Convention Note:
        input_shape represents the STORAGE format (per sample).
        For Conv1d: Storage is (T, F) but model expects (B, F, T).
        Data transformation happens in trainer.py, not here.
        This function extracts T and F from input_shape and passes them
        to Conv1dAE constructor in the correct order.
    
    Raises:
        UnsupportedShape: If input_shape doesn't match expected dimensionality
        ValueError: If mode is invalid
    """
    if mode == "dense":
        if len(input_shape) != 1:
            raise UnsupportedShape(
                f"Dense mode expects 1D shape (D,), got {input_shape}"
            )

        input_dim = input_shape[0]
        hidden_dims = kwargs.get("hidden_dims")

        return DenseAE(
            input_dim=input_dim, latent_dim=latent_dim, hidden_dims=hidden_dims
        )

    else:
        raise ValueError(f"Invalid mode: {mode}. Must be 'dense' or 'conv1d'")


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def init_weights(model: nn.Module) -> None:
    """Initialize model weights with appropriate strategy."""
    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, (nn.Conv1d, nn.ConvTranspose1d)):
            nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)


def model_summary(model: nn.Module, input_shape: tuple[int, ...]) -> str:
    """Generate human-readable model summary."""
    lines = []
    lines.append("=" * 80)
    lines.append(f"Model: {model.__class__.__name__}")
    lines.append("=" * 80)
    lines.append(f"{'Layer (type)':<30} {'Output Shape':<25} {'Param #':<15}")
    lines.append("=" * 80)

    total_params = 0

    # Create dummy input
    if len(input_shape) == 1:
        dummy_input = torch.randn(1, input_shape[0])
    elif len(input_shape) == 2:
        dummy_input = torch.randn(1, input_shape[1], input_shape[0])
    else:
        return "Unsupported input shape"

    # Register hooks
    layer_info = []

    def hook_fn(module, input, output):
        class_name = module.__class__.__name__
        params = sum(p.numel() for p in module.parameters())

        if isinstance(output, torch.Tensor):
            output_shape = str(tuple(output.shape))
        else:
            output_shape = "multiple"

        layer_info.append((class_name, output_shape, params))

    hooks = []
    for module in model.modules():
        if not isinstance(module, nn.Sequential) and module != model:
            hooks.append(module.register_forward_hook(hook_fn))

    # Forward pass
    with torch.no_grad():
        model(dummy_input)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Build summary
    for layer_name, output_shape, params in layer_info:
        lines.append(f"{layer_name:<30} {output_shape:<25} {params:<15,}")
        total_params += params

    lines.append("=" * 80)
    lines.append(f"Total params: {total_params:,}")
    lines.append(f"Trainable params: {total_params:,}")
    lines.append("Non-trainable params: 0")
    lines.append("=" * 80)

    return "\n".join(lines)
