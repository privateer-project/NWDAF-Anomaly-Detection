#!/usr/bin/env python3
"""
Model Adapter - Loads and interfaces with Attention Autoencoder models
Part of the PRIVATEER Security Analytics framework
"""

import os
import sys
import logging
import importlib.util
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn as nn

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('model-adapter')


class ModelAdapter:
    """Adapter for loading and using different model architectures"""

    def __init__(self, model_dir: str = "/app/models"):
        self.model_dir = Path(model_dir)
        self.attention_ae_path = self.model_dir / "attention_ae.py"
        self.model_path = self.model_dir / "latest.pt"

    def load_model(self, config: Dict[str, Any]):
        """
        Load the appropriate model architecture based on configuration
        """
        # Check if we have attention_ae.py
        if self.attention_ae_path.exists():
            logger.info(f"Loading Attention Autoencoder from {self.attention_ae_path}")
            return self._load_attention_autoencoder(config)
        else:
            logger.warning(f"attention_ae.py not found at {self.attention_ae_path}. Using fallback model.")
            return self._load_fallback_model(config)

    def _load_attention_autoencoder(self, config: Dict[str, Any]):
        """
        Dynamically import and load the Attention Autoencoder model
        """
        try:
            # Dynamically import the attention_ae module
            spec = importlib.util.spec_from_file_location("attention_ae", self.attention_ae_path)
            attention_ae_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(attention_ae_module)

            # Extract model classes from the module
            AttentionAutoencoder = getattr(attention_ae_module, "AttentionAutoencoder", None)
            AttentionAutoencoderConfig = getattr(attention_ae_module, "AttentionAutoencoderConfig", None)

            if AttentionAutoencoder is None:
                raise ImportError("AttentionAutoencoder class not found in attention_ae.py")

            # Create model configuration
            model_config = None
            if AttentionAutoencoderConfig is not None:
                model_config = AttentionAutoencoderConfig(
                    input_size=len(config.get("FEATURE_LIST", [])) or 8,
                    seq_len=int(config.get("SEQUENCE_LENGTH", "120")),
                    hidden_dim=int(config.get("HIDDEN_SIZE", "64")),
                    num_layers=int(config.get("NUM_LAYERS", "2")),
                    num_heads=int(config.get("NUM_HEADS", "4")),
                    dropout=float(config.get("DROPOUT", "0.1"))
                )

            # Create the model
            model = AttentionAutoencoder(config=model_config)

            # Load weights if available
            if self.model_path.exists():
                try:
                    state_dict = torch.load(self.model_path)
                    model.load_state_dict(state_dict)
                    logger.info(f"Loaded model weights from {self.model_path}")
                except Exception as e:
                    logger.error(f"Error loading model weights: {e}")
            else:
                logger.warning(f"No model weights found at {self.model_path}")

            return model

        except Exception as e:
            logger.error(f"Error loading Attention Autoencoder: {e}")
            return self._load_fallback_model(config)

    def _load_fallback_model(self, config: Dict[str, Any]):
        """
        Load a fallback LSTM Autoencoder model
        """

        class LSTMAutoencoder(nn.Module):
            """
            LSTM Autoencoder model for anomaly detection.
            """

            def __init__(self, input_size, hidden_size, num_layers=1):
                super(LSTMAutoencoder, self).__init__()

                # Encoder
                self.encoder = nn.LSTM(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    batch_first=True
                )

                # Decoder
                self.decoder = nn.LSTM(
                    input_size=hidden_size,
                    hidden_size=input_size,
                    num_layers=num_layers,
                    batch_first=True
                )

            def forward(self, x):
                # Encode
                _, (hidden, _) = self.encoder(x)

                # Reshape for decoder
                hidden_state = hidden[-1].unsqueeze(0).repeat(x.size(1), 1).unsqueeze(0)
                cell_state = torch.zeros_like(hidden_state)

                # Initialize decoder input as zeros
                decoder_input = torch.zeros(
                    (x.size(0), x.size(1), hidden.size(2)),
                    device=x.device
                )

                # Decode
                outputs, _ = self.decoder(decoder_input, (hidden_state, cell_state))

                return outputs

        # Create fallback model
        model = LSTMAutoencoder(
            input_size=len(config.get("FEATURE_LIST", [])) or 8,
            hidden_size=int(config.get("HIDDEN_SIZE", "64")),
            num_layers=int(config.get("NUM_LAYERS", "2"))
        )

        # Load weights if available and compatible
        if self.model_path.exists():
            try:
                state_dict = torch.load(self.model_path)
                model.load_state_dict(state_dict)
                logger.info(f"Loaded model weights into fallback model from {self.model_path}")
            except Exception as e:
                logger.error(f"Error loading model weights into fallback model: {e}")

        logger.info("Created fallback LSTM Autoencoder model")
        return model
