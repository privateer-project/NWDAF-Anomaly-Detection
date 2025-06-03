"""
Anomaly Detector Service - Detects anomalies using PRIVATEER model
"""
import os
import json
import numpy as np
import torch
from kafka import KafkaConsumer, KafkaProducer
from datetime import datetime

from privateer_ad.config import MLFlowConfig, MetadataConfig, TrainingConfig
from privateer_ad.utils import load_champion_model


class Detector:
    def __init__(self):
        self.threshold = float(os.environ.get('THRESHOLD', '0.061'))
        self.model = self._load_model()
        self.metadata = MetadataConfig()
        self.feature_list = self.metadata.get_input_features()
        self.mlflow_config = MLFlowConfig()
        self.loss_fn = getattr(torch.nn, TrainingConfig().loss_fn)(reduction='none')

    def _load_model(self):
        """Load pre-trained model or use simple threshold-based detection"""
        try:
            # In production, load from MLFlow
            # For demo, use simple autoencoder simulation
            model = load_champion_model(tracking_uri=self.mlflow_config.tracking_uri)
            return model
        except Exception as e:
            raise f"Model did not load: {e}"

    def detect(self, data):
        """Detect anomaly in the data"""
        # Extract features
        features = []
        for feature in self.feature_list:
            value = data.get(feature, 0)
            if value is None:
                value = 0
            features.append(float(value))
        input = np.array(features)
        # Calculate reconstruction error
        output = self.model(input)
        reconstruction_error = self.loss_fn(input, output)  # reconstruction loss
        reconstruction_error = torch.mean(input=reconstruction_error, dim=(1, 2))
        # Check if anomaly
        is_anomaly = reconstruction_error > self.threshold

        return {
            'is_anomaly': is_anomaly,
            'reconstruction_error': float(reconstruction_error),
            'threshold': self.threshold,
            'timestamp': datetime.now().isoformat(),
            'device_id': data.get('imeisv', 'unknown'),
            'true_label': data.get('attack', None)
        }

def main():
    bootstrap_servers = os.environ.get('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092')
    input_topic = os.environ.get('INPUT_TOPIC', 'anonymized-data')
    alert_topic = os.environ.get('ALERT_TOPIC', 'anomaly-alerts')

    print(f"Starting anomaly detector...")
    print(f"Input: {input_topic}")
    print(f"Alerts: {alert_topic}")

    consumer = KafkaConsumer(
        input_topic,
        bootstrap_servers=bootstrap_servers,
        value_deserializer=lambda v: json.loads(v.decode('utf-8')),
        auto_offset_reset='latest',
        group_id='detector-group'
    )

    producer = KafkaProducer(
        bootstrap_servers=bootstrap_servers,
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )

    detector = Detector()
    stats = {
        'total': 0,
        'anomalies': 0,
        'true_positives': 0,
        'false_positives': 0
    }

    try:
        for message in consumer:
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
                    'data': data
                }
                producer.send(alert_topic, value=alert)

                # Check accuracy
                if result['true_label'] == 1:
                    stats['true_positives'] += 1
                else:
                    stats['false_positives'] += 1

                print(f"🚨 Anomaly detected: Device {result['device_id']}, Score: {result['reconstruction_error']:.4f}")

            # Print stats periodically
            if stats['total'] % 100 == 0:
                tpr = (stats['true_positives'] / stats['anomalies'] * 100) if stats['anomalies'] > 0 else 0
                print(f"📊 Processed: {stats['total']}, Anomalies: {stats['anomalies']}, TPR: {tpr:.1f}%")

    except KeyboardInterrupt:
        print("Stopping detector...")
    except Exception as e:
        print(f"Error: {e}")

    consumer.close()
    producer.close()

if __name__ == '__main__':
    main()