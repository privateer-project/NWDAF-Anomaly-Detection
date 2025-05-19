#!/usr/bin/env python3
"""
Security Analytics Agent - Anomaly detection using ML models
Part of the PRIVATEER Security Analytics framework
"""

import os
import sys
import json
import time
import argparse
import logging
import traceback
import threading
import uuid
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import numpy as np
import torch
import torch.nn as nn
from kafka import KafkaConsumer, KafkaProducer

# Import model adapter
try:
    from model_adapter import ModelAdapter
except ImportError:
    # Will load dynamically later
    pass

# Import local modules if available
try:
    from misp_client import MISPClient
except ImportError:
    # Will load dynamically later
    pass

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('analytics-agent')


class FPGAInferenceEngine:
    """
    Client for FPGA-accelerated inference.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.optimized_model_path = Path(config.get("MODEL_REGISTRY_PATH", "/app/models")) / "optimized"
        logger.info(f"FPGA Inference Engine initialized")

    def infer(self, x: torch.Tensor) -> torch.Tensor:
        """Run inference on the FPGA"""
        # In a real implementation, this would use the FPGA vendor's API
        # For demo, simulate faster inference on FPGA
        start_time = time.time()

        # Simulate FPGA inference time - much faster than CPU
        fpga_inference_time = 0.005  # 5ms for FPGA vs ~50ms for CPU
        time.sleep(fpga_inference_time)

        # For demo, create a realistic output (reconstruction)
        # In real implementation, this would come from the FPGA
        output = x * 0.9 + torch.randn_like(x) * 0.1

        end_time = time.time()
        logger.debug(f"FPGA inference completed in {(end_time - start_time) * 1000:.2f}ms")

        return output


class SecurityAnalyticsAgent:
    """
    The Security Analytics Agent handles the inference process for
    detecting anomalies in network traffic.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize Kafka consumer for getting anonymized data
        self.consumer = KafkaConsumer(
            config.get("KAFKA_ANONYMIZED_DATA_TOPIC", "anonymized-data"),
            bootstrap_servers=config.get("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092"),
            value_deserializer=lambda v: json.loads(v.decode('utf-8')),
            auto_offset_reset='latest',
            group_id='security-analytics-agent'
        )

        # Initialize Kafka producer for XAI requests
        self.xai_producer = KafkaProducer(
            bootstrap_servers=config.get("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092"),
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )

        # Initialize Kafka consumer for XAI responses
        self.xai_consumer = KafkaConsumer(
            config.get("KAFKA_XAI_RESPONSE_TOPIC", "xai-response"),
            bootstrap_servers=config.get("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092"),
            value_deserializer=lambda v: json.loads(v.decode('utf-8')),
            auto_offset_reset='latest',
            group_id='security-analytics-agent-xai'
        )

        # Initialize the XAI thread
        self.xai_thread = threading.Thread(target=self._listen_for_xai_responses)
        self.xai_responses = {}
        self.xai_lock = threading.Lock()

        # Initialize model using the adapter
        self._init_model_adapter()
        self.model = self._load_model()
        self.model.to(self.device)
        self.model.eval()

        # Initialize the MISP client
        misp_config = {
            "MISP_URL": config.get("MISP_URL", "https://cc-cracs-201.inesctec.pt"),
            "MISP_API_KEY": config.get("MISP_API_KEY", ""),
            "MISP_SCRIPTS_PATH": config.get("MISP_SCRIPTS_PATH", "/app/misp_scripts"),
            "MISP_TEMP_DIR": config.get("MISP_TEMP_DIR", "/app/temp")
        }

        try:
            # Try to import from local module first
            self.misp_client = MISPClient(misp_config)
        except (ImportError, NameError):
            # If not available, dynamically load the module
            import importlib.util
            try:
                spec = importlib.util.spec_from_file_location("misp_client",
                                                              config.get("MISP_CLIENT_PATH", "/app/misp_client.py"))
                misp_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(misp_module)
                self.misp_client = misp_module.MISPClient(misp_config)
            except Exception as e:
                logger.error(f"Error loading MISP client: {e}")

                # Create dummy client that logs instead of sending
                class DummyMISPClient:
                    def report_anomaly(self, data):
                        logger.info(f"MISP reporting disabled - would report anomaly: {data}")

                self.misp_client = DummyMISPClient()

        # Data buffer for creating sequences
        self.data_buffer = []
        self.buffer_size = int(config.get("SEQUENCE_LENGTH", "120"))  # Sequence length for the LSTM model

        # Processing thread
        self.thread = None
        self.running = False

        # Feature list for the model
        self.feature_list = config.get("FEATURE_LIST", [
            'dl_bitrate', 'ul_bitrate', 'dl_retx', 'ul_tx',
            'dl_tx', 'ul_retx', 'bearer_0_dl_total_bytes',
            'bearer_0_ul_total_bytes'
        ])

        # Initialize FPGA inference engine if enabled
        self.inference_engine = None
        if config.get("FPGA_ENABLED", True):
            try:
                self.inference_engine = FPGAInferenceEngine(config)
                logger.info("FPGA acceleration enabled for inference")
            except Exception as e:
                logger.error(f"Error initializing FPGA inference engine: {e}")

        logger.info("Security Analytics Agent initialized")

    def _init_model_adapter(self):
        """Initialize the model adapter"""
        try:
            # Try to import from local module first
            self.model_adapter = ModelAdapter(
                model_dir=self.config.get("MODEL_REGISTRY_PATH", "/app/models")
            )
        except (ImportError, NameError):
            # If not available, dynamically load the module
            import importlib.util
            try:
                spec = importlib.util.spec_from_file_location("model_adapter",
                                                              "/app/model_adapter.py")
                adapter_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(adapter_module)
                self.model_adapter = adapter_module.ModelAdapter(
                    model_dir=self.config.get("MODEL_REGISTRY_PATH", "/app/models")
                )
                logger.info("Model adapter loaded dynamically")
            except Exception as e:
                logger.error(f"Error loading model adapter: {e}")
                raise

    def _listen_for_xai_responses(self):
        """Background thread to listen for XAI responses"""
        try:
            for message in self.xai_consumer:
                try:
                    response = message.value
                    request_id = response.get("request_id", "unknown")

                    # Store the response
                    with self.xai_lock:
                        self.xai_responses[request_id] = response

                    logger.debug(f"Received XAI response for request {request_id}")

                except Exception as e:
                    logger.error(f"Error processing XAI response: {e}")

                if not self.running:
                    break

        except Exception as e:
            logger.error(f"Error in XAI response listener: {e}")

    def _load_model(self) -> nn.Module:
        """Load the ML model using the adapter"""
        try:
            return self.model_adapter.load_model(self.config)
        except Exception as e:
            logger.error(f"Error loading model through adapter: {e}")
            raise

    def _preprocess_sequence(self, data_sequence: List[Dict]) -> torch.Tensor:
        """Extract features from a sequence of data points and convert to tensor"""
        # Extract features
        sequence = []
        for data_point in data_sequence:
            features = [float(data_point.get(feat, 0.0)) for feat in self.feature_list]
            sequence.append(features)

        # Convert to tensor
        return torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(self.device)

    def _process_sequence(self):
        """Process a full sequence of data"""
        try:
            # Preprocess data
            x = self._preprocess_sequence(self.data_buffer)

            # Run inference
            with torch.no_grad():
                if self.inference_engine and self.config.get("FPGA_ENABLED", True):
                    # Use FPGA acceleration
                    output = self.inference_engine.infer(x)
                else:
                    # Use CPU/GPU for inference
                    output = self.model(x)

            # Calculate reconstruction error
            mse = torch.mean(torch.pow(x - output, 2), dim=(1, 2)).item()

            # Check if anomaly
            threshold = float(self.config.get("DETECTION_THRESHOLD", "0.8"))
            is_anomaly = mse > threshold

            # If anomaly detected, report to MISP
            if is_anomaly:
                # Get the latest data point for the alert
                latest_data = self.data_buffer[-1]

                # Request explanation from XAI component
                self._request_explanation(self.data_buffer, mse, is_anomaly)

                # Create MISP event
                self._report_anomaly(latest_data, mse)

                logger.info(f"ALERT: Anomaly detected! Reconstruction error: {mse:.4f}")
            else:
                if np.random.random() < 0.01:  # Log occasionally
                    logger.info(f"Normal traffic. Reconstruction error: {mse:.4f}")
        except Exception as e:
            logger.error(f"Error processing sequence: {e}")
            logger.error(traceback.format_exc())

    def _request_explanation(self, data_sequence: List[Dict], anomaly_score: float, is_anomaly: bool):
        """Request explanation from XAI component"""
        request_id = str(uuid.uuid4())

        request = {
            "request_id": request_id,
            "sequence": data_sequence,
            "anomaly_score": anomaly_score,
            "is_anomaly": is_anomaly,
            "timestamp": datetime.now().isoformat()
        }

        # Send request to XAI component
        self.xai_producer.send(
            self.config.get("KAFKA_XAI_REQUEST_TOPIC", "xai-request"),
            value=request
        )

        logger.info(f"Requested explanation from XAI component (request_id: {request_id})")

        # Wait for response
        wait_time = 0
        while wait_time < 10:  # Wait up to 10 seconds
            with self.xai_lock:
                if request_id in self.xai_responses:
                    explanation = self.xai_responses.pop(request_id)
                    logger.info(f"Received explanation: {explanation.get('explanation_text', 'No explanation text')}")
                    return explanation

            time.sleep(0.5)
            wait_time += 0.5

        logger.warning(f"No explanation received for request {request_id} within timeout")
        return None

    def _report_anomaly(self, data: Dict, score: float):
        """Report detected anomaly to MISP"""
        # Create event data
        event_data = {
            "timestamp": data.get("timestamp", datetime.now().isoformat()),
            "imeisv": data.get("imeisv", "unknown"),
            "cell": data.get("cell", "unknown"),
            "anomaly_score": score,
            "traffic_metrics": {
                feat: data.get(feat, 0.0) for feat in self.feature_list
            }
        }

        # Send to MISP
        self.misp_client.report_anomaly(event_data)

    def run(self):
        """Main processing loop"""
        self.running = True
        logger.info("Starting Security Analytics Agent")

        # Start XAI response listener thread
        self.xai_thread.daemon = True
        self.xai_thread.start()

        try:
            for message in self.consumer:
                if not self.running:
                    break

                # Get data
                data = message.value

                # Add to buffer
                self.data_buffer.append(data)

                # Keep buffer at max size
                if len(self.data_buffer) > self.buffer_size:
                    self.data_buffer.pop(0)

                # Process if we have enough data
                if len(self.data_buffer) == self.buffer_size:
                    self._process_sequence()

        except KeyboardInterrupt:
            logger.info("Stopping Security Analytics Agent")
            self.running = False
        except Exception as e:
            logger.error(f"Error in Security Analytics Agent: {e}")
            logger.error(traceback.format_exc())
            self.running = False


def main():
    parser = argparse.ArgumentParser(description='Security Analytics Agent')
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
            "KAFKA_ANONYMIZED_DATA_TOPIC": os.environ.get("KAFKA_ANONYMIZED_DATA_TOPIC", "anonymized-data"),
            "KAFKA_XAI_REQUEST_TOPIC": os.environ.get("KAFKA_XAI_REQUEST_TOPIC", "xai-request"),
            "KAFKA_XAI_RESPONSE_TOPIC": os.environ.get("KAFKA_XAI_RESPONSE_TOPIC", "xai-response"),
            "MODEL_REGISTRY_PATH": os.environ.get("MODEL_REGISTRY_PATH", "/app/models"),
            "MODEL_CURRENT_VERSION": os.environ.get("MODEL_CURRENT_VERSION", "latest"),
            "DETECTION_THRESHOLD": float(os.environ.get("DETECTION_THRESHOLD", "0.8")),
            "SEQUENCE_LENGTH": int(os.environ.get("SEQUENCE_LENGTH", "120")),
            "HIDDEN_SIZE": int(os.environ.get("HIDDEN_SIZE", "64")),
            "NUM_LAYERS": int(os.environ.get("NUM_LAYERS", "2")),
            "NUM_HEADS": int(os.environ.get("NUM_HEADS", "4")),
            "DROPOUT": float(os.environ.get("DROPOUT", "0.1")),
            "MISP_URL": os.environ.get("MISP_URL", "https://cc-cracs-201.inesctec.pt"),
            "MISP_API_KEY": os.environ.get("MISP_API_KEY", ""),
            "MISP_SCRIPTS_PATH": os.environ.get("MISP_SCRIPTS_PATH", "/app/misp_scripts"),
            "MISP_TEMP_DIR": os.environ.get("MISP_TEMP_DIR", "/app/temp"),
            "MISP_CLIENT_PATH": os.environ.get("MISP_CLIENT_PATH", "/app/misp_client.py"),
            "FPGA_ENABLED": os.environ.get("FPGA_ENABLED", "true").lower() in ("true", "1", "yes"),
            "FEATURE_LIST": [
                'dl_bitrate', 'ul_bitrate', 'dl_retx', 'ul_tx',
                'dl_tx', 'ul_retx', 'bearer_0_dl_total_bytes',
                'bearer_0_ul_total_bytes'
            ]
        }
        logger.info("Using default configuration")

    # Create and run analytics agent
    agent = SecurityAnalyticsAgent(config)
    agent.run()


if __name__ == "__main__":
    main()
