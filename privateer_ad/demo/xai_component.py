#!/usr/bin/env python3
"""
XAI Component - Provides explainable insights for anomaly detection models
Part of the PRIVATEER Security Analytics framework
"""

import os
import sys
import json
import time
import argparse
import logging
import threading
import importlib.util
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
from kafka import KafkaConsumer, KafkaProducer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('xai-component')


class XAIComponent:
    """
    Provides explainable insights for anomaly detection models.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize Kafka consumer to listen for XAI explanation requests
        self.consumer = KafkaConsumer(
            config.get("KAFKA_XAI_REQUEST_TOPIC", "xai-request"),
            bootstrap_servers=config.get("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092"),
            value_deserializer=lambda v: json.loads(v.decode('utf-8')),
            auto_offset_reset='latest',
            group_id='xai-component'
        )

        # Initialize Kafka producer to publish XAI explanations
        self.producer = KafkaProducer(
            bootstrap_servers=config.get("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092"),
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )

        # Load the model
        self.model = self._load_model()
        self.model.to(self.device)
        self.model.eval()

        # Initialize processing thread
        self.thread = None
        self.running = False

        # Feature list for the model
        self.feature_list = config.get("FEATURE_LIST", [
            'dl_bitrate', 'ul_bitrate', 'dl_retx', 'ul_tx',
            'dl_tx', 'ul_retx', 'bearer_0_dl_total_bytes',
            'bearer_0_ul_total_bytes'
        ])

        logger.info("XAI Component initialized")

    def _load_model(self) -> nn.Module:
        """Load the appropriate ML model"""
        model_dir = Path(self.config.get("MODEL_REGISTRY_PATH", "/app/models"))
        model_path = model_dir / f"{self.config.get('MODEL_CURRENT_VERSION', 'latest')}.pt"
        attention_ae_path = model_dir / "attention_ae.py"

        # Try to load Attention Autoencoder if it exists
        if attention_ae_path.exists():
            try:
                logger.info(f"Loading Attention Autoencoder from {attention_ae_path}")

                # Dynamically import the module
                spec = importlib.util.spec_from_file_location("attention_ae", attention_ae_path)
                attention_ae_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(attention_ae_module)

                # Get model classes
                AttentionAutoencoder = getattr(attention_ae_module, "AttentionAutoencoder", None)
                AttentionAutoencoderConfig = getattr(attention_ae_module, "AttentionAutoencoderConfig", None)

                if not AttentionAutoencoder:
                    raise ImportError("AttentionAutoencoder class not found")

                # Create model configuration
                model_config = None
                if AttentionAutoencoderConfig:
                    model_config = AttentionAutoencoderConfig(
                        input_size=len(self.config.get("FEATURE_LIST", [])) or 8,
                        seq_len=int(self.config.get("SEQUENCE_LENGTH", "120")),
                        hidden_dim=int(self.config.get("HIDDEN_SIZE", "64")),
                        num_layers=int(self.config.get("NUM_LAYERS", "2")),
                        num_heads=int(self.config.get("NUM_HEADS", "4")),
                        dropout=float(self.config.get("DROPOUT", "0.1"))
                    )

                # Create model
                model = AttentionAutoencoder(config=model_config)

                # Load weights if available
                if model_path.exists():
                    model.load_state_dict(torch.load(model_path))
                    logger.info(f"Loaded model weights from {model_path}")

                return model

            except Exception as e:
                logger.error(f"Error loading Attention Autoencoder: {e}")
                # Fall back to LSTM Autoencoder

        # Fallback to LSTM Autoencoder
        logger.info("Using fallback LSTM Autoencoder model")

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
            input_size=len(self.config.get("FEATURE_LIST", [])) or 8,
            hidden_size=int(self.config.get("HIDDEN_SIZE", "64")),
            num_layers=int(self.config.get("NUM_LAYERS", "2"))
        )

        # Load weights if available and compatible
        if model_path.exists():
            try:
                model.load_state_dict(torch.load(model_path))
                logger.info(f"Loaded model weights into fallback model from {model_path}")
            except Exception as e:
                logger.error(f"Error loading model weights into fallback model: {e}")

        return model

    def _calculate_feature_importance(self, inputs: torch.Tensor) -> Dict[str, float]:
        """
        Calculate the importance of each feature using perturbation analysis.
        This is a simple implementation - perturbing each feature and measuring impact.
        """
        feature_importances = {}

        # Make sure we're in evaluation mode
        self.model.eval()

        # Get baseline reconstruction error
        with torch.no_grad():
            baseline_output = self.model(inputs)
            baseline_error = torch.mean(torch.pow(inputs - baseline_output, 2)).item()

        # For each feature, perturb it and see how much error changes
        for i, feature_name in enumerate(self.feature_list):
            # Create a perturbed copy
            perturbed_input = inputs.clone()

            # Perturb by adding noise to this feature only
            perturbed_input[:, :, i] = perturbed_input[:, :, i] * 1.1  # 10% perturbation

            # Calculate new error
            with torch.no_grad():
                perturbed_output = self.model(perturbed_input)
                perturbed_error = torch.mean(torch.pow(perturbed_input - perturbed_output, 2)).item()

            # Importance is the difference in error
            importance = (perturbed_error - baseline_error) / baseline_error
            feature_importances[feature_name] = float(importance)

        # Normalize to sum to 1.0
        total = sum(abs(v) for v in feature_importances.values())
        if total > 0:
            feature_importances = {k: abs(v) / total for k, v in feature_importances.items()}

        return feature_importances

    def _calculate_attention_weights(self, inputs: torch.Tensor) -> List[float]:
        """
        For Attention-based models, try to extract attention weights
        This is model-specific and depends on the attention mechanism implementation
        """
        # Check if model has attention layers that expose weights
        if hasattr(self.model, 'transformer_encoder') and hasattr(self.model.transformer_encoder, 'layers'):
            try:
                # This implementation is specific to the attention mechanism
                # in the transformer encoder architecture
                with torch.no_grad():
                    # Forward pass to get attention weights
                    _ = self.model(inputs)

                    # Extract attention weights from the last layer
                    # This is implementation-specific and may need adjustment
                    last_layer = self.model.transformer_encoder.layers[-1]
                    if hasattr(last_layer, 'self_attn'):
                        # Some implementations store attention weights
                        if hasattr(last_layer.self_attn, 'attn_weights'):
                            attn_weights = last_layer.self_attn.attn_weights

                            # Average across heads and batch dimension
                            avg_weights = attn_weights.mean(dim=(0, 1)).cpu().numpy().tolist()
                            return avg_weights
            except Exception as e:
                logger.error(f"Error extracting attention weights: {e}")

        # Fallback: analyze time contribution through reconstruction error
        return self._analyze_time_contribution(inputs)

    def _analyze_time_contribution(self, inputs: torch.Tensor) -> List[float]:
        """
        Analyze which time periods contribute most to the anomaly score.
        """
        time_contributions = []

        with torch.no_grad():
            output = self.model(inputs)

            # Calculate error at each time step
            error_per_timestep = torch.mean(torch.pow(inputs - output, 2), dim=2)

            # Convert to list of floats
            time_contributions = error_per_timestep.squeeze().cpu().tolist()

        return time_contributions

    def _generate_explanation(self, data: Dict) -> Dict:
        """
        Generate an explanation for a given sequence of data.
        """
        # Extract the input sequence
        sequence_data = data.get("sequence", [])
        if not sequence_data:
            return {"error": "No sequence data provided"}

        # Convert to tensor
        try:
            sequence = []
            for data_point in sequence_data:
                features = [float(data_point.get(feat, 0.0)) for feat in self.feature_list]
                sequence.append(features)

            inputs = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(self.device)

            # Calculate feature importance
            feature_importance = self._calculate_feature_importance(inputs)

            # Get time contribution - try attention weights first, fall back to reconstruction error
            time_contribution = self._calculate_attention_weights(inputs)

            # Get top contributing features
            sorted_features = sorted(
                feature_importance.items(),
                key=lambda x: x[1],
                reverse=True
            )

            # Generate explanation text
            if data.get("is_anomaly", False):
                explanation_text = f"Anomaly detected with score {data.get('anomaly_score', 0):.4f}. "
                explanation_text += f"The main contributing feature was {sorted_features[0][0]} (importance: {sorted_features[0][1]:.2f}). "

                # Add information about time pattern
                max_time_idx = np.argmax(time_contribution)
                explanation_text += f"The anomaly was most evident at time index {max_time_idx}."
            else:
                explanation_text = "No anomaly detected. "
                explanation_text += f"The data shows normal patterns in all features."

            # Add information about model architecture
            model_type = "Attention Autoencoder" if hasattr(self.model, 'transformer_encoder') else "LSTM Autoencoder"
            explanation_text += f" Analysis performed using {model_type} model."

            return {
                "explanation_text": explanation_text,
                "feature_importance": feature_importance,
                "time_contribution": time_contribution,
                "top_features": [name for name, _ in sorted_features[:3]],
                "model_type": model_type,
                "request_id": data.get("request_id", "unknown")
            }

        except Exception as e:
            logger.error(f"Error generating explanation: {e}")
            return {"error": f"Failed to generate explanation: {str(e)}"}

    def run(self):
        """Main processing loop for XAI component"""
        self.running = True
        logger.info("Starting XAI Component")

        try:
            for message in self.consumer:
                if not self.running:
                    break

                # Get explanation request
                request = message.value
                logger.info(f"Received explanation request: {request.get('request_id', 'unknown')}")

                # Generate explanation
                explanation = self._generate_explanation(request)

                # Publish explanation
                self.producer.send(
                    self.config.get("KAFKA_XAI_RESPONSE_TOPIC", "xai-response"),
                    value=explanation
                )

                logger.info(f"Published explanation for request: {request.get('request_id', 'unknown')}")

        except KeyboardInterrupt:
            logger.info("Stopping XAI Component")
            self.running = False
        except Exception as e:
            logger.error(f"Error in XAI Component: {e}")
            self.running = False


def main():
    parser = argparse.ArgumentParser(description='XAI Component')
    parser.add_argument('--config', type=str, default='/app/config.json',
                        help='Path to configuration file')
    args = parser.parse_args()

    # Load configuration
    try:
        with open(args.config, 'r') as f:
            config = json.load(f)
        logger.info(f"Loaded configuration from {args.config}")
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        # Use default configuration
        config = {
            "KAFKA_BOOTSTRAP_SERVERS": os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092"),
            "KAFKA_XAI_REQUEST_TOPIC": os.environ.get("KAFKA_XAI_REQUEST_TOPIC", "xai-request"),
            "KAFKA_XAI_RESPONSE_TOPIC": os.environ.get("KAFKA_XAI_RESPONSE_TOPIC", "xai-response"),
            "MODEL_REGISTRY_PATH": os.environ.get("MODEL_REGISTRY_PATH", "/app/models"),
            "MODEL_CURRENT_VERSION": os.environ.get("MODEL_CURRENT_VERSION", "latest"),
            "HIDDEN_SIZE": int(os.environ.get("HIDDEN_SIZE", "64")),
            "NUM_LAYERS": int(os.environ.get("NUM_LAYERS", "2")),
            "NUM_HEADS": int(os.environ.get("NUM_HEADS", "4")),
            "SEQUENCE_LENGTH": int(os.environ.get("SEQUENCE_LENGTH", "120")),
            "DROPOUT": float(os.environ.get("DROPOUT", "0.1")),
            "FEATURE_LIST": [
                'dl_bitrate', 'ul_bitrate', 'dl_retx', 'ul_tx',
                'dl_tx', 'ul_retx', 'bearer_0_dl_total_bytes',
                'bearer_0_ul_total_bytes'
            ]
        }
        logger.info("Using default configuration")

    # Create and run XAI component
    xai = XAIComponent(config)
    xai.run()


if __name__ == "__main__":
    main()
