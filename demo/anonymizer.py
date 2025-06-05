# demo/anonymizer.py
"""
Anonymization and Preprocessing Service
- Applies privacy transformations
- Uses DataProcessor for preprocessing
"""
import os
import json
import pandas as pd

from kafka import KafkaConsumer, KafkaProducer
from collections import defaultdict
from datetime import datetime
import sys

sys.path.append('/app')

from privateer_ad.etl import DataProcessor
from privateer_ad.config import DataConfig


class AnonymizerPreprocessor:
    def __init__(self):
        self.device_buffers = defaultdict(list)

        # Initialize DataProcessor
        self.data_config = DataConfig()
        self.data_config.seq_len = 12
        self.data_processor = DataProcessor(self.data_config)

        # Load scaler to ensure it's ready
        self.data_processor.load_scaler()

    def anonymize(self, data):
        """Apply anonymization to sensitive fields"""
        anonymized = data.copy()

        # Hash device identifier
        if 'imeisv' in anonymized:
            original_imeisv = anonymized['imeisv']
            anonymized['_original_device_id'] = original_imeisv  # Keep for buffering
            anonymized['imeisv'] = f"anon-{hash(anonymized['imeisv']) % 10000}"

        # Mask IP addresses
        for field in ['ip', 'bearer_0_ip', 'bearer_1_ip']:
            if field in anonymized and anonymized[field]:
                anonymized[field] = "xxx.xxx.xxx.xxx"

        # Hash other identifiers
        for field in ['amf_ue_id', '5g_tmsi']:
            if field in anonymized and anonymized[field]:
                anonymized[field] = f"anon-{hash(str(anonymized[field])) % 10000}"

        return anonymized

    def process_sample(self, data):
        """Process incoming sample and return sequences when ready"""
        device_id = data.get('_original_device_id', data.get('imeisv', 'unknown'))

        # Add to device buffer
        self.device_buffers[device_id].append(data)

        # Keep only last seq_len samples
        if len(self.device_buffers[device_id]) > self.data_config.seq_len:
            self.device_buffers[device_id] = self.device_buffers[device_id][-self.data_config.seq_len:]

        # Check if we have enough samples
        if len(self.device_buffers[device_id]) == self.data_config.seq_len:
            try:
                # Create DataFrame from buffered samples
                df = pd.DataFrame(self.device_buffers[device_id])

                # Ensure _time column is datetime
                if '_time' in df.columns:
                    df['_time'] = pd.to_datetime(df['_time'])
                elif '_timestamp' in df.columns:
                    # Use our added timestamp if original _time is missing
                    df['_time'] = pd.to_datetime(df['_timestamp'])
                else:
                    # Create timestamps if missing
                    df['_time'] = pd.date_range(
                        end=datetime.now(),
                        periods=len(df),
                        freq='1S'
                    )

                # Apply DataProcessor's cleaning and preprocessing
                df_cleaned = self.data_processor.clean_data(df)
                df_processed = self.data_processor.preprocess_data(df_cleaned, only_benign=False)

                # Extract processed features in correct order
                feature_matrix = df_processed[self.data_processor.input_features].values

                # Create tensor format [1, seq_len, features]
                tensor = feature_matrix.reshape(1, self.data_config.seq_len, len(self.data_processor.input_features))

                # Get metadata from the last sample
                last_sample = self.device_buffers[device_id][-1]

                return {
                    'device_id': last_sample['imeisv'],  # Use anonymized ID
                    'tensor': tensor.tolist(),  # Convert to list for JSON
                    'timestamp': last_sample.get('_timestamp', datetime.now().isoformat()),
                    'metadata': {
                        'attack': int(last_sample.get('attack', 0)),
                        'malicious': int(last_sample.get('malicious', 0)),
                        'cell': last_sample.get('cell', 'unknown'),
                        'attack_number': int(last_sample.get('attack_number', 0))
                    }
                }

            except Exception as e:
                print(f"Error processing sequence for device {device_id}: {e}")
                # Clear buffer on error to start fresh
                self.device_buffers[device_id] = []
                return None

        return None


def main():
    bootstrap_servers = os.environ.get('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092')
    input_topic = os.environ.get('INPUT_TOPIC', 'raw-network-data')
    output_topic = os.environ.get('OUTPUT_TOPIC', 'preprocessed-data')

    print("Starting anonymizer with DataProcessor preprocessing...")

    consumer = KafkaConsumer(
        input_topic,
        bootstrap_servers=bootstrap_servers,
        value_deserializer=lambda v: json.loads(v.decode('utf-8')),
        auto_offset_reset='earliest'  # Start from beginning
    )

    producer = KafkaProducer(
        bootstrap_servers=bootstrap_servers,
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )

    processor = AnonymizerPreprocessor()
    processed_count = 0

    print(f"Input features: {processor.data_processor.input_features}")
    print(f"Sequence length: {processor.data_config.seq_len}")

    for message in consumer:
        try:
            raw_data = message.value

            # Anonymize first
            anonymized_data = processor.anonymize(raw_data)

            # Process and check if sequence is ready
            result = processor.process_sample(anonymized_data)

            if result:
                # Send preprocessed tensor to detector
                producer.send(output_topic, value=result)
                processed_count += 1

                if processed_count % 10 == 0:
                    print(f"Processed {processed_count} sequences")

                # Log first few for debugging
                if processed_count <= 3:
                    print(f"Sent tensor for device {result['device_id']}")
                    print(f"  Shape: [1, {len(result['tensor'][0])}, {len(result['tensor'][0][0])}]")
                    print(f"  Metadata: {result['metadata']}")

        except Exception as e:
            print(f"Error processing message: {e}")
            import traceback
            traceback.print_exc()
            continue

    consumer.close()
    producer.close()


if __name__ == '__main__':
    main()