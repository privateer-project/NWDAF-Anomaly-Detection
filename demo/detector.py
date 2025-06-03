"""
Anomaly Detector Service - Detects anomalies using PRIVATEER model
"""
import os
import sys
import json
import time
import logging

import torch
from datetime import datetime

import requests
from kafka import KafkaConsumer, KafkaProducer
from kafka.errors import NoBrokersAvailable

from privateer_ad.config import MLFlowConfig, MetadataConfig, ModelConfig
from privateer_ad.utils import load_champion_model


# Force stdout to be unbuffered for Docker logging
sys.stdout.reconfigure(line_buffering=True)


def wait_for_service(url, service_name, max_retries=30, delay=5):
    """Wait for a service to be available"""
    for i in range(max_retries):
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                logging.info(f"✅ {service_name} is available at {url}")
                return True
        except Exception as e:
            logging.warning(f"⏳ Waiting for {service_name}... attempt {i+1}/{max_retries} - {e}")
            time.sleep(delay)
    
    logging.error(f"❌ Failed to connect to {service_name} after {max_retries} attempts")
    return False


def wait_for_kafka(bootstrap_servers, max_retries=30):
    """Wait for Kafka to be available"""
    for i in range(max_retries):
        try:
            # Test connection
            test_producer = KafkaProducer(
                bootstrap_servers=bootstrap_servers,
                request_timeout_ms=5000,
                retries=1
            )
            test_producer.close()
            logging.info(f"✅ Connected to Kafka at {bootstrap_servers}")
            return True
        except NoBrokersAvailable:
            logging.warning(f"⏳ Waiting for Kafka... attempt {i+1}/{max_retries}")
            time.sleep(2)
        except Exception as e:
            logging.warning(f"⏳ Kafka connection error: {e}")
            time.sleep(2)

    logging.error(f"❌ Failed to connect to Kafka after {max_retries} attempts")
    return False


class Detector:
    def __init__(self):
        """Initialize the detector with proper error handling"""
        logging.info("🔧 Initializing PRIVATEER Anomaly Detector...")
        
        # Configure MLflow
        self.mlflow_config = MLFlowConfig()
        # Use environment variable for MLflow tracking URI
        logging.info(f"📊 MLflow URI: {self.mlflow_config.tracking_uri}")
        
        # Configure metadata
        self.metadata = MetadataConfig()
        self.feature_list = self.metadata.get_input_features()
        logging.info(f"📝 Input features: {self.feature_list}")
        
        # Model configuration
        self.model_config = ModelConfig()

        logging.info(f"🤖 Model name: {self.model_config.model_type}")
        
        # Initialize model, threshold, and loss function
        self.model = None
        self.threshold = None
        self.loss_fn = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f"💻 Using device: {self.device}")

    def load_model(self):
        """Load model with retry logic"""
        max_retries = 5
        for attempt in range(max_retries):
            try:
                logging.info(f"🔄 Loading model... attempt {attempt + 1}/{max_retries}")
                self.model, self.threshold, self.loss_fn = load_champion_model(
                    tracking_uri=self.mlflow_config.tracking_uri,
                    model_name=self.model_config.model_type
                )
                self.model.to(self.device)
                self.model.eval()
                logging.info(f"✅ Model loaded successfully!")
                logging.info(f"📏 Threshold: {self.threshold}")
            except Exception as e:
                logging.error(f"❌ Failed to load model (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(10)  # Wait before retry
                else:
                    raise e

    def detect(self, data):
        """Detect anomaly in the data"""
        try:
            # Extract features
            features = []
            for feature in self.feature_list:
                value = data.get(feature, 0)
                if value is None or value == '' or (isinstance(value, str) and value.lower() == 'nan'):
                    value = 0
                try:
                    features.append(float(value))
                except (ValueError, TypeError):
                    logging.warning(f"⚠️ Invalid value for feature {feature}: {value}, using 0")
                    features.append(0.0)
            
            # Convert to tensor with proper shape for sequence model
            # The model expects input shape: [batch_size, seq_len, features]
            # For single prediction, we'll create a sequence of length 1
            features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
            
            # Run inference
            with torch.no_grad():
                output = self.model(features_tensor)
                reconstruction_error = self.loss_fn(features_tensor, output).mean().item()
            
            # Check if anomaly
            is_anomaly = reconstruction_error > self.threshold
            
            return {
                'is_anomaly': bool(is_anomaly),
                'reconstruction_error': float(reconstruction_error),
                'threshold': float(self.threshold),
                'timestamp': datetime.now().isoformat(),
                'device_id': data.get('imeisv', 'unknown'),
                'true_label': data.get('attack', None),
                'features_used': len(features),
                'model_name': self.model_config.model_type
            }
        except Exception as e:
            logging.error(f"❌ Error in anomaly detection: {e}")
            return {
                'is_anomaly': False,
                'reconstruction_error': 0.0,
                'threshold': self.threshold or 0.5,
                'timestamp': datetime.now().isoformat(),
                'device_id': data.get('imeisv', 'unknown'),
                'true_label': data.get('attack', None),
                'error': str(e),
                'model_name': self.model_config.model_type
            }


def main():
    logging.info("🛡️ Starting PRIVATEER Anomaly Detection Service...")
    
    # Configuration from environment
    bootstrap_servers = os.environ.get('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092')
    input_topic = os.environ.get('INPUT_TOPIC', 'anonymized-data')
    alert_topic = os.environ.get('ALERT_TOPIC', 'anomaly-alerts')
    mlflow_uri = os.environ.get('PRIVATEER_MLFLOW_TRACKING_URI', 'http://localhost:5001')
    
    logging.info(f"📥 Input topic: {input_topic}")
    logging.info(f"📤 Alert topic: {alert_topic}")
    logging.info(f"🔌 Kafka servers: {bootstrap_servers}")
    logging.info(f"📊 MLflow URI: {mlflow_uri}")
    
    # Wait for MLflow to be available
    mlflow_health_url = mlflow_uri.replace(':5001', ':5050') + '/health'
    if not wait_for_service(mlflow_health_url, "MLflow", max_retries=20, delay=10):
        logging.error("❌ MLflow is not available. Exiting...")
        sys.exit(1)
    
    # Wait for Kafka to be available
    if not wait_for_kafka(bootstrap_servers):
        logging.error("❌ Kafka is not available. Exiting...")
        sys.exit(1)
    
    # Initialize detector and load model
    try:
        detector = Detector()
        detector.load_model()
    except Exception as e:
        logging.error(f"❌ Failed to initialize detector: {e}")
        sys.exit(1)
    
    # Set up Kafka consumer and producer
    try:
        logging.info("🔧 Setting up Kafka consumer and producer...")
        
        consumer = KafkaConsumer(
            input_topic,
            bootstrap_servers=bootstrap_servers,
            value_deserializer=lambda v: json.loads(v.decode('utf-8')),
            auto_offset_reset='latest',
            group_id='detector-group',
            enable_auto_commit=True,
            consumer_timeout_ms=5000
        )
        
        producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        
        logging.info("✅ Kafka consumer and producer created successfully")
        
    except Exception as e:
        logging.error(f"❌ Failed to create Kafka clients: {e}")
        sys.exit(1)
    
    # Statistics tracking
    stats = {
        'total': 0,
        'anomalies': 0,
        'true_positives': 0,
        'false_positives': 0,
        'true_negatives': 0,
        'false_negatives': 0
    }
    
    last_log_time = time.time()
    logging.info("🚀 Starting anomaly detection...")
    logging.info("⏳ Waiting for messages...")
    
    try:
        while True:
            # Poll for messages
            message_batch = consumer.poll(timeout_ms=1000)
            
            if not message_batch:
                # No messages received, log periodically
                current_time = time.time()
                if current_time - last_log_time > 30:  # Log every 30 seconds
                    logging.info(f"💤 Still waiting for messages... (processed {stats['total']} so far)")
                    last_log_time = current_time
                continue
            
            # Process messages
            for topic_partition, messages in message_batch.items():
                for message in messages:
                    try:
                        data = message.value
                        result = detector.detect(data)
                        
                        stats['total'] += 1
                        
                        if result['is_anomaly']:
                            stats['anomalies'] += 1
                            
                            # Send alert
                            alert = {
                                'alert_id': f"alert-{stats['anomalies']}",
                                'device_id': result['device_id'],
                                'score': result['reconstruction_error'],
                                'threshold': result['threshold'],
                                'timestamp': result['timestamp'],
                                'model_name': result['model_name'],
                                'data': data
                            }
                            producer.send(alert_topic, value=alert)
                            
                            # Update accuracy statistics
                            true_label = result.get('true_label')
                            if true_label == 1:  # Actually malicious
                                stats['true_positives'] += 1
                            elif true_label == 0:  # Actually benign
                                stats['false_positives'] += 1
                                
                            logging.info(f"🚨 Anomaly detected: Device {result['device_id']}, "
                                       f"Score: {result['reconstruction_error']:.4f}, "
                                       f"True label: {true_label}")
                        else:
                            # Track true negatives and false negatives
                            true_label = result.get('true_label')
                            if true_label == 0:  # Actually benign, correctly identified
                                stats['true_negatives'] += 1
                            elif true_label == 1:  # Actually malicious, missed
                                stats['false_negatives'] += 1
                        
                        # Log detailed stats periodically
                        if stats['total'] % 100 == 0:
                            total_anomalies = stats['anomalies']
                            tpr = (stats['true_positives'] / (stats['true_positives'] + stats['false_negatives'])) * 100 if (stats['true_positives'] + stats['false_negatives']) > 0 else 0
                            fpr = (stats['false_positives'] / (stats['false_positives'] + stats['true_negatives'])) * 100 if (stats['false_positives'] + stats['true_negatives']) > 0 else 0
                            
                            logging.info(f"📊 Processed: {stats['total']}, "
                                       f"Anomalies: {total_anomalies}, "
                                       f"TPR: {tpr:.1f}%, "
                                       f"FPR: {fpr:.1f}%")
                        
                        # Log first few messages for debugging
                        if stats['total'] <= 5:
                            logging.info(f"📝 Sample detection {stats['total']}: "
                                       f"device={result['device_id']}, "
                                       f"anomaly={result['is_anomaly']}, "
                                       f"score={result['reconstruction_error']:.4f}")
                            
                    except Exception as e:
                        logging.error(f"❌ Error processing message: {e}")
                        continue
    
    except KeyboardInterrupt:
        logging.info("🛑 Stopping detector...")
    except Exception as e:
        logging.error(f"❌ Unexpected error: {e}")
    finally:
        logging.info("🧹 Cleaning up...")
        consumer.close()
        producer.close()
        logging.info(f"✅ Detector stopped. Final stats: {stats}")


if __name__ == '__main__':
    main()