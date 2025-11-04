# Autoencoder Model Architectures
#
# This module defines PyTorch autoencoder architectures for anomaly detection.
# Two modes: DenseAE for 1D vectors, Conv1dAE for 2D time-series.
#
# Classes to implement:
#
# class DenseAE(nn.Module):
#   """Dense/Fully-connected autoencoder for 1D feature vectors."""
#   
#   def __init__(self, input_dim: int, latent_dim: int = 32, hidden_dims: list[int] | None = None):
#     """
#     Initialize dense autoencoder.
#     
#     Args:
#       - input_dim: Size of input vector (D)
#       - latent_dim: Size of latent/bottleneck dimension
#       - hidden_dims: List of hidden layer sizes for encoder
#                      If None, use default: [input_dim//2, input_dim//4]
#     
#     Architecture:
#       Encoder: Linear layers with ReLU activations
#         input_dim → hidden[0] → hidden[1] → ... → latent_dim
#       
#       Decoder: Mirror of encoder
#         latent_dim → hidden[-1] → hidden[-2] → ... → input_dim
#       
#     Example with input_dim=128, latent_dim=32:
#       Encoder: 128 → 64 → 32 (latent)
#       Decoder: 32 → 64 → 128 (reconstruction)
#     
#     Layers should use:
#       - nn.Linear for transformations
#       - nn.ReLU for activations (except final layer)
#       - No activation on final decoder layer (reconstruction)
#     """
#     super().__init__()
#     # TODO: Build encoder layers
#     # TODO: Build decoder layers
#   
#   def forward(self, x: torch.Tensor) -> torch.Tensor:
#     """
#     Forward pass through autoencoder.
#     
#     Args:
#       - x: Input tensor of shape (B, D) where B=batch, D=input_dim
#     
#     Returns: Reconstructed tensor of shape (B, D)
#     
#     Algorithm:
#       1. Pass through encoder: latent = encoder(x)
#       2. Pass through decoder: x_hat = decoder(latent)
#       3. Return x_hat
#     """
#   
#   def encode(self, x: torch.Tensor) -> torch.Tensor:
#     """Get latent representation."""
#     # Just encoder forward pass
#   
#   def decode(self, z: torch.Tensor) -> torch.Tensor:
#     """Reconstruct from latent."""
#     # Just decoder forward pass
#
#
# class Conv1dAE(nn.Module):
#   """Convolutional autoencoder for 2D time-series (T, F) tensors."""
#   
#   def __init__(
#     self,
#     num_features: int,
#     seq_len: int,
#     latent_dim: int = 32,
#     num_filters: list[int] | None = None
#   ):
#     """
#     Initialize 1D convolutional autoencoder.
#     
#     Args:
#       - num_features: Number of features/channels (F)
#       - seq_len: Sequence length (T)
#       - latent_dim: Size of latent representation
#       - num_filters: List of filter counts for conv layers
#                      If None, use default: [16, 32, 64]
#     
#     Architecture:
#       Input format: (B, F, T) where B=batch, F=features, T=time
#       PyTorch Conv1d expects channels-first format.
#       
#       Encoder: Conv1d layers with ReLU
#         (B, F, T) → Conv1d(F, 16, k=3, s=2, p=1) → ReLU
#                   → Conv1d(16, 32, k=3, s=2, p=1) → ReLU
#                   → Conv1d(32, 64, k=3, s=2, p=1) → ReLU
#                   → Flatten → Linear → latent_dim
#       
#       Decoder: Mirror with ConvTranspose1d
#         latent_dim → Linear → Unflatten
#                    → ConvTranspose1d(64, 32, k=3, s=2, p=1, output_padding=1) → ReLU
#                    → ConvTranspose1d(32, 16, k=3, s=2, p=1, output_padding=1) → ReLU
#                    → ConvTranspose1d(16, F, k=3, s=2, p=1, output_padding=1)
#       
#     Note: Adjust output_padding to match input seq_len exactly.
#     May need to calculate intermediate dimensions based on seq_len.
#     """
#     super().__init__()
#     # TODO: Build encoder conv layers
#     # TODO: Build decoder deconv layers
#     # TODO: Calculate flattened dimension for latent connection
#   
#   def forward(self, x: torch.Tensor) -> torch.Tensor:
#     """
#     Forward pass through convolutional autoencoder.
#     
#     Args:
#       - x: Input tensor of shape (B, F, T)
#     
#     Returns: Reconstructed tensor of shape (B, F, T)
#     
#     Algorithm:
#       1. Pass through conv encoder
#       2. Flatten and project to latent
#       3. Project from latent and unflatten
#       4. Pass through deconv decoder
#       5. Return reconstruction
#     """
#   
#   def encode(self, x: torch.Tensor) -> torch.Tensor:
#     """Get latent representation."""
#   
#   def decode(self, z: torch.Tensor) -> torch.Tensor:
#     """Reconstruct from latent."""
#
#
# Factory function:
#
# def build_model(
#   mode: Literal["dense", "conv1d"],
#   input_shape: tuple[int, ...],
#   latent_dim: int = 32,
#   **kwargs
# ) -> nn.Module:
#   """
#   Factory function to build appropriate autoencoder based on mode.
#   
#   Args:
#     - mode: "dense" for 1D or "conv1d" for 2D
#     - input_shape: Tensor dimensions
#       For dense: (D,) where D is feature count
#       For conv1d: (T, F) where T is time steps, F is features
#     - latent_dim: Size of latent bottleneck
#     - **kwargs: Additional model-specific params
#   
#   Returns: Initialized nn.Module (DenseAE or Conv1dAE)
#   
#   Raises:
#     - ValueError: if mode invalid
#     - UnsupportedShape: if input_shape doesn't match mode
#   
#   Example:
#     # For 1D vector of 128 features
#     model = build_model("dense", (128,), latent_dim=32)
#     
#     # For 2D time-series (10 timesteps, 8 features)
#     model = build_model("conv1d", (10, 8), latent_dim=32)
#   """
#
#
# Helper functions:
#
# def count_parameters(model: nn.Module) -> int:
#   """
#   Count trainable parameters in model.
#   
#   Returns: Total number of trainable parameters
#   Useful for logging model complexity.
#   """
#
# def init_weights(model: nn.Module) -> None:
#   """
#   Initialize model weights with appropriate strategy.
#   
#   For Linear layers: Xavier/Glorot initialization
#   For Conv layers: Kaiming/He initialization
#   Biases: zeros
#   
#   Call after model construction:
#     model = build_model(...)
#     init_weights(model)
#   """
#
# def model_summary(model: nn.Module, input_shape: tuple[int, ...]) -> str:
#   """
#   Generate human-readable model summary.
#   
#   Args:
#     - model: PyTorch model
#     - input_shape: Input dimensions (excluding batch)
#   
#   Returns: String with layer info, shapes, param counts
#   Similar to Keras model.summary()
#   """
#
# Usage patterns:
#   # Build and initialize model
#   model = build_model("dense", (128,), latent_dim=32)
#   init_weights(model)
#   
#   # Check model
#   num_params = count_parameters(model)
#   print(f"Model has {num_params:,} parameters")
#   
#   # Forward pass
#   x = torch.randn(32, 128)  # batch of 32
#   x_hat = model(x)
#   
#   # Compute reconstruction loss
#   loss = F.mse_loss(x_hat, x)
