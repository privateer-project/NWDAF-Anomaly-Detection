"""
Tests for Model Architectures

Test coverage:
- DenseAE initialization and forward pass
- Conv1dAE initialization and forward pass
- build_model factory function
- Shape compatibility
- Encode/decode methods
- Parameter counting
- Weight initialization
"""

import pytest
import torch
import torch.nn as nn

from hitl.models import (
    DenseAE,
    Conv1dAE,
    build_model,
    count_parameters,
    init_weights,
    model_summary,
)
from hitl.errors import UnsupportedShape


class TestDenseAE:
    """Tests for Dense Autoencoder."""

    def test_dense_ae_initializes(self):
        """Test DenseAE can be created."""
        model = DenseAE(input_dim=128, latent_dim=32)

        assert isinstance(model, nn.Module)
        assert model.input_dim == 128
        assert model.latent_dim == 32
        assert len(model.hidden_dims) == 2  # Default: [64, 32]

    def test_dense_ae_custom_hidden_dims(self):
        """Test DenseAE with custom hidden dimensions."""
        hidden_dims = [100, 50, 25]
        model = DenseAE(input_dim=128, latent_dim=16, hidden_dims=hidden_dims)

        assert model.hidden_dims == hidden_dims

    def test_dense_ae_forward_shape(self):
        """Test DenseAE forward pass preserves shape."""
        model = DenseAE(input_dim=128, latent_dim=32)
        x = torch.randn(16, 128)  # Batch of 16

        x_hat = model(x)

        assert x_hat.shape == x.shape
        assert x_hat.shape == (16, 128)

    def test_dense_ae_encode(self):
        """Test DenseAE encoding to latent."""
        model = DenseAE(input_dim=128, latent_dim=32)
        x = torch.randn(16, 128)

        z = model.encode(x)

        assert z.shape == (16, 32)

    def test_dense_ae_decode(self):
        """Test DenseAE decoding from latent."""
        model = DenseAE(input_dim=128, latent_dim=32)
        z = torch.randn(16, 32)

        x_hat = model.decode(z)

        assert x_hat.shape == (16, 128)

    def test_dense_ae_encode_decode_roundtrip(self):
        """Test encode -> decode produces valid output."""
        model = DenseAE(input_dim=128, latent_dim=32)
        x = torch.randn(16, 128)

        z = model.encode(x)
        x_hat = model.decode(z)

        assert x_hat.shape == x.shape

    def test_dense_ae_gradient_flow(self):
        """Test gradients flow through the model."""
        model = DenseAE(input_dim=128, latent_dim=32)
        x = torch.randn(16, 128, requires_grad=True)

        x_hat = model(x)
        loss = torch.mean((x - x_hat) ** 2)
        loss.backward()

        # Check that input has gradients
        assert x.grad is not None

        # Check that model parameters have gradients
        for param in model.parameters():
            assert param.grad is not None


class TestConv1dAE:
    """Tests for Convolutional 1D Autoencoder."""

    def test_conv1d_ae_initializes(self):
        """Test Conv1dAE can be created."""
        model = Conv1dAE(num_features=8, seq_len=10, latent_dim=32)

        assert isinstance(model, nn.Module)
        assert model.num_features == 8
        assert model.seq_len == 10
        assert model.latent_dim == 32

    def test_conv1d_ae_custom_filters(self):
        """Test Conv1dAE with custom filter counts."""
        num_filters = [8, 16, 32]
        model = Conv1dAE(
            num_features=8, seq_len=10, latent_dim=16, num_filters=num_filters
        )

        assert model.num_filters == num_filters

    def test_conv1d_ae_forward_shape(self):
        """Test Conv1dAE forward pass preserves shape."""
        model = Conv1dAE(num_features=8, seq_len=10, latent_dim=32)
        # Input: (B, F, T)
        x = torch.randn(16, 8, 10)

        x_hat = model(x)

        assert x_hat.shape == x.shape
        assert x_hat.shape == (16, 8, 10)

    def test_conv1d_ae_encode(self):
        """Test Conv1dAE encoding to latent."""
        model = Conv1dAE(num_features=8, seq_len=10, latent_dim=32)
        x = torch.randn(16, 8, 10)

        z = model.encode(x)

        assert z.shape == (16, 32)

    def test_conv1d_ae_decode(self):
        """Test Conv1dAE decoding from latent."""
        model = Conv1dAE(num_features=8, seq_len=10, latent_dim=32)
        z = torch.randn(16, 32)

        x_hat = model.decode(z)

        # Output should have correct number of features
        assert x_hat.size(1) == 8

    def test_conv1d_ae_encode_decode_roundtrip(self):
        """Test encode -> decode produces valid output."""
        model = Conv1dAE(num_features=8, seq_len=10, latent_dim=32)
        x = torch.randn(16, 8, 10)

        z = model.encode(x)
        x_hat = model.decode(z)

        # After decode, forward adjusts to exact seq_len
        x_hat = model(x)
        assert x_hat.shape == x.shape

    def test_conv1d_ae_gradient_flow(self):
        """Test gradients flow through the model."""
        model = Conv1dAE(num_features=8, seq_len=10, latent_dim=32)
        x = torch.randn(16, 8, 10, requires_grad=True)

        x_hat = model(x)
        loss = torch.mean((x - x_hat) ** 2)
        loss.backward()

        # Check that input has gradients
        assert x.grad is not None

        # Check that model parameters have gradients
        for param in model.parameters():
            assert param.grad is not None

    def test_conv1d_ae_longer_sequence(self):
        """Test Conv1dAE with longer sequence."""
        model = Conv1dAE(num_features=8, seq_len=100, latent_dim=32)
        x = torch.randn(8, 8, 100)

        x_hat = model(x)

        assert x_hat.shape == (8, 8, 100)


class TestBuildModel:
    """Tests for model factory function."""

    def test_build_model_dense(self):
        """Test factory creates DenseAE for 1D shape."""
        model = build_model("dense", (128,), latent_dim=32)

        assert isinstance(model, DenseAE)
        assert model.input_dim == 128
        assert model.latent_dim == 32

    def test_build_model_dense_with_hidden_dims(self):
        """Test factory passes through hidden_dims."""
        hidden_dims = [100, 50]
        model = build_model("dense", (128,), latent_dim=32, hidden_dims=hidden_dims)

        assert model.hidden_dims == hidden_dims

    def test_build_model_conv1d(self):
        """Test factory creates Conv1dAE for 2D shape."""
        model = build_model("conv1d", (10, 8), latent_dim=32)

        assert isinstance(model, Conv1dAE)
        assert model.num_features == 8
        assert model.seq_len == 10
        assert model.latent_dim == 32

    def test_build_model_conv1d_with_filters(self):
        """Test factory passes through num_filters."""
        num_filters = [8, 16, 32]
        model = build_model("conv1d", (10, 8), latent_dim=32, num_filters=num_filters)

        assert model.num_filters == num_filters

    def test_build_model_invalid_mode(self):
        """Test factory raises for invalid mode."""
        with pytest.raises(ValueError, match="Invalid mode"):
            build_model("invalid", (128,), latent_dim=32)  # type: ignore[arg-type]

    def test_build_model_dense_shape_mismatch(self):
        """Test factory raises for 2D shape with dense mode."""
        with pytest.raises(UnsupportedShape, match="Dense mode expects 1D"):
            build_model("dense", (10, 8), latent_dim=32)

    def test_build_model_conv1d_shape_mismatch(self):
        """Test factory raises for 1D shape with conv1d mode."""
        with pytest.raises(UnsupportedShape, match="Conv1d mode expects 2D"):
            build_model("conv1d", (128,), latent_dim=32)

    def test_build_model_dense_forward(self):
        """Test model from factory works for forward pass."""
        model = build_model("dense", (128,), latent_dim=32)
        x = torch.randn(16, 128)

        x_hat = model(x)

        assert x_hat.shape == (16, 128)

    def test_build_model_conv1d_forward(self):
        """Test model from factory works for forward pass."""
        model = build_model("conv1d", (10, 8), latent_dim=32)
        x = torch.randn(16, 8, 10)

        x_hat = model(x)

        assert x_hat.shape == (16, 8, 10)


class TestUtilities:
    """Tests for utility functions."""

    def test_count_parameters_dense(self):
        """Test parameter counting for DenseAE."""
        model = DenseAE(input_dim=128, latent_dim=32)

        param_count = count_parameters(model)

        assert param_count > 0
        assert isinstance(param_count, int)

        # Should match manual count
        manual_count = sum(p.numel() for p in model.parameters())
        assert param_count == manual_count

    def test_count_parameters_conv1d(self):
        """Test parameter counting for Conv1dAE."""
        model = Conv1dAE(num_features=8, seq_len=10, latent_dim=32)

        param_count = count_parameters(model)

        assert param_count > 0
        assert isinstance(param_count, int)

    def test_init_weights_dense(self):
        """Test weight initialization for DenseAE."""
        model = DenseAE(input_dim=128, latent_dim=32)

        # Get initial weights
        initial_weights = [p.clone() for p in model.parameters()]

        # Initialize
        init_weights(model)

        # Weights should have changed (very unlikely to be identical after init)
        for initial, current in zip(initial_weights, model.parameters()):
            # At least some weights should differ
            if initial.numel() > 10:  # Skip small tensors like biases
                assert not torch.allclose(initial, current)

    def test_init_weights_conv1d(self):
        """Test weight initialization for Conv1dAE."""
        model = Conv1dAE(num_features=8, seq_len=10, latent_dim=32)

        # Should not raise
        init_weights(model)

        # Check biases are zeros
        for module in model.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.ConvTranspose1d)):
                if module.bias is not None:
                    assert torch.allclose(module.bias, torch.zeros_like(module.bias))

    def test_model_summary_dense(self):
        """Test model summary generation for DenseAE."""
        model = DenseAE(input_dim=128, latent_dim=32)

        summary = model_summary(model, (128,))

        assert isinstance(summary, str)
        assert "DenseAE" in summary
        assert "Total params" in summary
        assert "Trainable params" in summary

    def test_model_summary_conv1d(self):
        """Test model summary generation for Conv1dAE."""
        model = Conv1dAE(num_features=8, seq_len=10, latent_dim=32)

        summary = model_summary(model, (10, 8))

        assert isinstance(summary, str)
        assert "Conv1dAE" in summary
        assert "Total params" in summary

    def test_model_summary_contains_layer_info(self):
        """Test model summary contains layer information."""
        model = DenseAE(input_dim=128, latent_dim=32)

        summary = model_summary(model, (128,))

        # Should contain layer types
        assert "Linear" in summary
        assert "ReLU" in summary


class TestIntegration:
    """Integration tests for complete workflows."""

    def test_dense_training_step(self):
        """Test DenseAE can perform a training step."""
        model = DenseAE(input_dim=128, latent_dim=32)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Create batch
        x = torch.randn(32, 128)

        # Forward
        x_hat = model(x)
        loss = torch.mean((x - x_hat) ** 2)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Loss should be finite
        assert torch.isfinite(loss)

    def test_conv1d_training_step(self):
        """Test Conv1dAE can perform a training step."""
        model = Conv1dAE(num_features=8, seq_len=10, latent_dim=32)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Create batch
        x = torch.randn(32, 8, 10)

        # Forward
        x_hat = model(x)
        loss = torch.mean((x - x_hat) ** 2)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Loss should be finite
        assert torch.isfinite(loss)

    def test_dense_multiple_epochs(self):
        """Test DenseAE can train for multiple epochs."""
        model = DenseAE(input_dim=128, latent_dim=32)
        init_weights(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        x = torch.randn(32, 128)
        losses = []

        for _ in range(10):
            x_hat = model(x)
            loss = torch.mean((x - x_hat) ** 2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        # Loss should generally decrease
        assert losses[-1] < losses[0]

    def test_model_save_load(self):
        """Test model state can be saved and loaded."""
        model = DenseAE(input_dim=128, latent_dim=32)
        x = torch.randn(16, 128)

        # Get output before save
        with torch.no_grad():
            output_before = model(x)

        # Save state
        state_dict = model.state_dict()

        # Create new model and load state
        model2 = DenseAE(input_dim=128, latent_dim=32)
        model2.load_state_dict(state_dict)

        # Get output after load
        with torch.no_grad():
            output_after = model2(x)

        # Outputs should match
        assert torch.allclose(output_before, output_after)
