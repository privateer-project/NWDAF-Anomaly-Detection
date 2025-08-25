"""
Anonymization and Preprocessing Service
- Applies privacy transformations
- Uses DataProcessor for preprocessing
"""

import os
import json
import sys
import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import privkit as pk

from kafka import KafkaConsumer, KafkaProducer

sys.path.append('/app')
from privateer_ad.etl import DataProcessor


def default_serializer(data: dict) -> bytes:
    try:
        # Convert numpy types and serialize to JSON
        return json.dumps(data, cls=NumpyEncoder).encode('utf-8')
    except Exception as e:
        logging.error(f"Serialization error: {e}")
        raise


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, (pd.Timestamp, datetime)):
            return obj.isoformat()
        elif isinstance(obj, (pd.Timedelta, timedelta)):
            return obj.total_seconds()
        return super().default(obj)


class AnonymizerPreprocessor:
    def __init__(self):
        """Initialize the anonymizer with timestamp-based batching"""
        self.data_processor = DataProcessor()
        self.data_config = self.data_processor.data_config

        # Timestamp-based buffering for aggregation
        self.buffer = []  # timestamp -> list of device samples
        self.max_timestamp_age = 30  # seconds - maximum age before processing incomplete batches

    def anonymize(self, data, epsilon: float = 0.01, sensitivity: float = 1):
        """Apply anonymization to sensitive fields

        :param float epsilon: privacy parameter for the Laplace mechanism. Default value = 0.01.
        :param float sensitivity: sensitivity parameter. Default value = 1.
        """
        anonymized = data.copy()

        # Hash device identifier
        if 'imeisv' in anonymized:
            original_imeisv = anonymized['imeisv']

            anonymized['_original_device_id'] = original_imeisv  # Keep for buffering

            anonymized['imeisv'] = anonymized['imeisv'].apply(lambda x: f"{pk.Hash.get_obfuscated_data(x) % 10000}")

        # Mask IP addresses
        for field in ['ip', 'bearer_0_ip', 'bearer_1_ip']:
            if field in anonymized:
                anonymized[field] = "xxx.xxx.xxx.xxx"

        # Hash other identifiers
        for field in ['amf_ue_id', '5g_tmsi']:
            if field in anonymized:
                anonymized[field] = anonymized[field].apply(lambda x: f"{pk.Hash.get_obfuscated_data(x) % 10000}")

        # Obfuscate sensitive features with Laplace
        for field in ['dl_bitrate', 'ul_bitrate']:
            if field in anonymized:
                results = anonymized[field].apply(
                    pk.Laplace(epsilon=epsilon, sensitivity=sensitivity).get_obfuscated_point)
                obf_data = pd.DataFrame(results.tolist(), index=anonymized.index)
                anonymized[[f"{field}", f"{constants.QUALITY_LOSS}_{field}"]] = obf_data

        return anonymized

    def process_sample(self, data):
        """Process incoming sample using timestamp-based batching approach"""
        # Extract timestamp - normalize to the same second for batching
        df = pd.DataFrame.from_dict(data)
        df = self.data_processor.clean_data(df)
        df['_time'] = pd.to_datetime(df['_time'])
        df = df.sort_values(by=['_time']).reset_index(drop=True)
        df = self.data_processor.aggregate_by_time(df)
        df = self.data_processor.scale_data(df)

        aggregated_point = self._process_timestamp_batch(df['_time'].values, df)

        if aggregated_point is not None:
            self.buffer.append(aggregated_point)

            # Keep only last seq_len aggregated points
            if len(self.buffer) >= self.data_config.seq_len:
                self.buffer = self.buffer[-self.data_config.seq_len:]
                seq = self._build_sequence()
                return seq
        return None

    def _process_timestamp_batch(self, timestamp_key, df):
        """Process a complete timestamp batch across devices"""
        try:
            if len(df) > 0:
                aggregated_point = {
                    'timestamp': timestamp_key,
                    'features': df[self.data_processor.input_features].to_dict(orient='list'),
                    'metadata': {
                        'attack': int(df.get('attack', pd.Series([0])).iloc[0]),
                        'malicious': int(df.get('malicious', pd.Series([0])).iloc[0]),
                        'cell': df.get('cell', pd.Series(['unknown'])).iloc[0],
                        'attack_number': int(df.get('attack_number', pd.Series([0])).iloc[0])
                    }
                }
                return aggregated_point
            return None

        except Exception as e:
            print(f"Error processing timestamp batch {timestamp_key}: {e}")
            # Remove failed batch
            self.buffer.remove(df)
            return None

    def _build_sequence(self):
        """Build a complete sequence from aggregated timestamp points"""
        # Extract features in temporal sequence order
        features = {}
        for feature in self.data_processor.input_features:
            features[feature] = [point['features'][feature] for point in self.buffer]

        # Get metadata from the last (most recent) point
        last_point = self.buffer[-1]
        return {
            'device_id': 'network_aggregated',  # This represents the entire network
            'features': features,
            'timestamp': last_point['timestamp'],
            'metadata': last_point['metadata']
        }

def main():
    bootstrap_servers = os.environ.get('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092')
    input_topic = os.environ.get('INPUT_TOPIC', 'raw-network-data')
    output_topic = os.environ.get('OUTPUT_TOPIC', 'preprocessed-data')

    print("Starting anonymizer with DataProcessor preprocessing...")
    print('bootstrap_servers')
    consumer = KafkaConsumer(
        input_topic,
        bootstrap_servers=bootstrap_servers,
        value_deserializer=lambda v: json.loads(v.decode('utf-8')),
        auto_offset_reset='latest'
    )

    producer = KafkaProducer(
        bootstrap_servers=bootstrap_servers,
        value_serializer=default_serializer
    )

    processor = AnonymizerPreprocessor()
    processed_count = 0

    print(f"Input features: {processor.data_processor.input_features}")
    print(f"Sequence length: {processor.data_processor.data_config.seq_len}")

    for message in consumer:
        try:
            raw_data = message.value

            anonymized_data = processor.anonymize(raw_data)

            # Process and check if sequence is ready
            sequence = processor.process_sample(anonymized_data)
            if sequence:  # We get ONE network sequence when ready
                producer.send(output_topic, value=sequence)
                processed_count += 1
        except Exception as e:
            print(f"Error processing message: {e}")
            import traceback
            traceback.print_exc()
            exit()

    consumer.close()
    producer.close()


if __name__ == '__main__':
    main()