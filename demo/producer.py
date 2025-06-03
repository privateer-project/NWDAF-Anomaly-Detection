"""
Data Producer - Reads dataset and publishes to Kafka
"""
import os
import json
import time
import pandas as pd
from kafka import KafkaProducer
from datetime import datetime

def main():
    bootstrap_servers = os.environ.get('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092')
    topic = os.environ.get('KAFKA_TOPIC', 'raw-network-data')
    data_path = os.environ.get('DATA_PATH', '/app/data/amari_ue_data_merged_with_attack_number.csv')
    interval = float(os.environ.get('INTERVAL', '0.5'))

    print(f"Starting data producer...")
    print(f"Kafka: {bootstrap_servers}")
    print(f"Topic: {topic}")
    print(f"Data: {data_path}")

    # Initialize Kafka producer
    producer = KafkaProducer(
        bootstrap_servers=bootstrap_servers,
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )

    # Load dataset
    try:
        df = pd.read_csv(data_path)
        print(f"Loaded {len(df)} records")
    except Exception as e:
        print(f"Error loading data: {e}")
        # Use synthetic data as fallback
        df = None

    index = 0

    while True:
        try:
            if df is not None:
                # Use real data
                if index >= len(df):
                    index = 0
                    print("Restarting dataset from beginning")

                record = df.iloc[index].to_dict()
                # Convert NaN to None for JSON serialization
                record = {k: (None if pd.isna(v) else v) for k, v in record.items()}
                record['timestamp'] = datetime.now().isoformat()
                index += 1
            else:
                # Synthetic data
                record = {
                    'imeisv': f'86099604{index % 100000000:08d}',
                    'dl_bitrate': 5000 + index % 1000,
                    'ul_bitrate': 2000 + index % 500,
                    'dl_retx': index % 10,
                    'ul_tx': 30 + index % 20,
                    'dl_tx': 30 + index % 20,
                    'ul_retx': index % 5,
                    'bearer_0_dl_total_bytes': 50000 + index * 100,
                    'bearer_0_ul_total_bytes': 20000 + index * 50,
                    'attack': 1 if index % 20 == 0 else 0,
                    'timestamp': datetime.now().isoformat()
                }
                index += 1

            # Send to Kafka
            producer.send(topic, value=record)

            if index % 100 == 0:
                print(f"Published {index} records")

            time.sleep(interval)

        except KeyboardInterrupt:
            print("Stopping producer...")
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(1)

    producer.close()

if __name__ == '__main__':
    main()