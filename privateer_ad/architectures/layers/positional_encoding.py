import math
import torch
from torch import nn


class PositionalEncoding(nn.Module):
    """
    Implement sinusoidal positional encoding for transformer architectures.

    This module adds position-dependent sinusoidal patterns to input embeddings,
    enabling the model to understand sequential ordering without recurrent connections.
    The encoding uses alternating sine and cosine functions with different frequencies
    to create unique positional signatures for each position in the sequence.

    The mathematical formulation follows the original Transformer paper:
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Args:
        d_model (int): Dimension of the model embeddings
        max_seq_length (int): Maximum sequence length to pre-compute encodings for
        dropout (float): Dropout probability applied after adding positional encoding
    """

    def __init__(self, d_model: int, max_seq_length: int = 100, dropout: float = 0.1):
        """
        Initialize positional encoding with pre-computed sinusoidal patterns.

        Args:
            d_model: Model embedding dimension, must be positive integer
            max_seq_length: Maximum sequence length for pre-computation
            dropout: Dropout rate for regularization after position addition
        """
        super(PositionalEncoding, self).__init__()

        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}")
        if max_seq_length <= 0:
            raise ValueError(f"max_seq_length must be positive, got {max_seq_length}")
        if not 0. <= dropout <= 1.:
            raise ValueError(f"dropout must be between 0 and 1, got {dropout}")

        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)

        # Create division term for frequency scaling
        # This implements: 10000^(-2i/d_model) = exp(-2i * ln(10000) / d_model)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) *
            (-math.log(10000.0) / d_model)
        )

        # Apply sine to even indices (0, 2, 4, ...)
        pe[:, 0::2] = torch.sin(position * div_term)

        # Apply cosine to odd indices (1, 3, 5, ...)
        # Handle case where d_model is odd
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)

        # Add batch dimension: [1, max_seq_length, d_model]
        pe = pe.unsqueeze(0)

        # Register as buffer so it moves with the model but isn't a parameter
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input embeddings.

        Args:
            x: Input tensor of shape [batch_size, seq_length, d_model]

        Returns:
            Tensor with positional encoding added, same shape as input

        Raises:
            RuntimeError: If sequence length exceeds pre-computed maximum
            ValueError: If input doesn't have expected 3D shape
        """
        if x.dim() != 3:
            raise ValueError(f"Expected 3D input [batch, seq, features], got {x.dim()}D")

        batch_size, seq_length, feature_dim = x.shape

        if seq_length > self.pe.size(1):
            raise RuntimeError(
                f"Input sequence length {seq_length} exceeds maximum "
                f"pre-computed length {self.pe.size(1)}. "
                f"Increase max_seq_length during initialization."
            )

        if feature_dim != self.d_model:
            raise ValueError(
                f"Input feature dimension {feature_dim} doesn't match "
                f"model dimension {self.d_model}"
            )

        # Add positional encoding to input
        # pe shape: [1, max_seq_length, d_model]
        # x shape: [batch_size, seq_length, d_model]
        x = x + self.pe[:, :seq_length, :]

        return self.dropout(x)

    def extend_length(self, new_max_length: int) -> None:
        """
        Extend the maximum sequence length by recomputing positional encodings.

        This method allows dynamic extension of the positional encoding buffer
        when longer sequences are encountered during runtime.

        Args:
            new_max_length: New maximum sequence length, must be greater than current

        Raises:
            ValueError: If new_max_length is not greater than current maximum
        """
        current_max = self.pe.size(1)
        if new_max_length <= current_max:
            raise ValueError(
                f"New max length {new_max_length} must be greater than "
                f"current max length {current_max}"
            )

        # Recompute positional encoding with new length
        pe = torch.zeros(new_max_length, self.d_model, device=self.pe.device)
        position = torch.arange(0, new_max_length, dtype=torch.float, device=self.pe.device).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float, device=self.pe.device) *
            (-math.log(10000.0) / self.d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)

        if self.d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)

        # Update the registered buffer
        self.register_buffer('pe', pe)


# Example usage and testing
if __name__ == "__main__":
    # Test basic functionality
    d_model = 512
    max_seq_len = 100
    batch_size = 2
    seq_len = 50

    pos_enc = PositionalEncoding(d_model=d_model, max_seq_length=max_seq_len)

    # Create sample input
    x = torch.randn(batch_size, seq_len, d_model)

    # Apply positional encoding
    output = pos_enc(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Positional encoding buffer shape: {pos_enc.pe.shape}")

    # Test with odd d_model
    pos_enc_odd = PositionalEncoding(d_model=513, max_seq_length=100)
    x_odd = torch.randn(2, 50, 513)
    output_odd = pos_enc_odd(x_odd)
    print(f"Odd d_model output shape: {output_odd.shape}")

    # Test extension
    pos_enc.extend_length(200)
    print(f"Extended PE buffer shape: {pos_enc.pe.shape}")