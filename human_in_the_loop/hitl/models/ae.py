"""
Autoencoder Model Architectures

This module defines PyTorch autoencoder architectures for anomaly detection.
Two modes: DenseAE for 1D vectors, Conv1dAE for 2D time-series.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal

from ..errors import UnsupportedShape


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
            hidden_dims = [input_dim // 2, input_dim // 4]

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


class Conv1dAE(nn.Module):
    """Convolutional autoencoder for 2D time-series (T, F) tensors."""

    def __init__(
        self,
        num_features: int,
        seq_len: int,
        latent_dim: int = 32,
        num_filters: list[int] | None = None,
    ):
        """
        Initialize 1D convolutional autoencoder.

        Args:
            num_features: Number of features/channels (F)
            seq_len: Sequence length (T)
            latent_dim: Size of latent representation
            num_filters: List of filter counts for conv layers
                        If None, use default: [16, 32, 64]

        Architecture:
            Input format: (B, F, T) where B=batch, F=features, T=time
            PyTorch Conv1d expects channels-first format.

            Encoder: Conv1d layers with ReLU
                (B, F, T) → Conv1d(F, 16, k=3, s=2, p=1) → ReLU
                          → Conv1d(16, 32, k=3, s=2, p=1) → ReLU
                          → Conv1d(32, 64, k=3, s=2, p=1) → ReLU
                          → Flatten → Linear → latent_dim

            Decoder: Mirror with ConvTranspose1d
                latent_dim → Linear → Unflatten
                           → ConvTranspose1d(64, 32, k=3, s=2, p=1, output_padding=1) → ReLU
                           → ConvTranspose1d(32, 16, k=3, s=2, p=1, output_padding=1) → ReLU
                           → ConvTranspose1d(16, F, k=3, s=2, p=1, output_padding=1)

        Note: Adjust output_padding to match input seq_len exactly.
        """
        super().__init__()

        self.num_features = num_features
        self.seq_len = seq_len
        self.latent_dim = latent_dim

        # Default filter counts if not provided
        if num_filters is None:
            num_filters = [16, 32, 64]

        self.num_filters = num_filters

        # Build encoder
        encoder_layers = []
        in_channels = num_features

        for out_channels in num_filters:
            encoder_layers.append(
                nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
            )
            encoder_layers.append(nn.ReLU())
            in_channels = out_channels

        self.encoder_conv = nn.Sequential(*encoder_layers)

        # Calculate the flattened dimension after convolutions
        flattened_len = seq_len
        for _ in num_filters:
            flattened_len = (flattened_len + 2 * 1 - 3) // 2 + 1

        self.flattened_dim = num_filters[-1] * flattened_len
        self.flattened_len = flattened_len

        # Linear layer to latent
        self.encoder_fc = nn.Linear(self.flattened_dim, latent_dim)

        # Build decoder
        self.decoder_fc = nn.Linear(latent_dim, self.flattened_dim)

        # Deconv layers
        decoder_layers = []

        for i in range(len(num_filters) - 1, 0, -1):
            decoder_layers.append(
                nn.ConvTranspose1d(
                    num_filters[i],
                    num_filters[i - 1],
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                )
            )
            decoder_layers.append(nn.ReLU())

        # Final deconv layer
        decoder_layers.append(
            nn.ConvTranspose1d(
                num_filters[0],
                num_features,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            )
        )

        self.decoder_conv = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through convolutional autoencoder."""
        z = self.encode(x)
        x_hat = self.decode(z)

        # Trim or pad to match original seq_len
        if x_hat.size(2) != self.seq_len:
            if x_hat.size(2) > self.seq_len:
                x_hat = x_hat[:, :, : self.seq_len]
            else:
                padding = self.seq_len - x_hat.size(2)
                x_hat = F.pad(x_hat, (0, padding))

        return x_hat

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Get latent representation."""
        h = self.encoder_conv(x)
        h = h.view(h.size(0), -1)
        z = self.encoder_fc(h)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstruct from latent."""
        h = self.decoder_fc(z)
        h = h.view(h.size(0), self.num_filters[-1], self.flattened_len)
        x_hat = self.decoder_conv(h)
        return x_hat


def build_model(
    mode: Literal["dense", "conv1d"],
    input_shape: tuple[int, ...],
    latent_dim: int = 32,
    **kwargs,
) -> nn.Module:
    """Factory function to build appropriate autoencoder based on mode."""
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

    elif mode == "conv1d":
        if len(input_shape) != 2:
            raise UnsupportedShape(
                f"Conv1d mode expects 2D shape (T, F), got {input_shape}"
            )

        seq_len, num_features = input_shape
        num_filters = kwargs.get("num_filters")

        return Conv1dAE(
            num_features=num_features,
            seq_len=seq_len,
            latent_dim=latent_dim,
            num_filters=num_filters,
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
