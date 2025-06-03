"""
Data Producer - Reads dataset and publishes to Kafka
"""
import os
import json
import time
import logging

from datetime import datetime

import pandas as pd

from kafka import KafkaProducer

from privateer_ad.config.settings import PathConfig

def main():
    bootstrap_servers = os.environ.get('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092')
    topic = os.environ.get('KAFKA_TOPIC', 'raw-network-data')
    data_path = PathConfig().raw_dataset
    interval = float(os.environ.get('INTERVAL', '0.5'))

    logging.info(f"Starting data producer...")
    logging.info(f"Kafka: {bootstrap_servers}")
    logging.info(f"Topic: {topic}")
    logging.info(f"Data: {data_path}")

    # Initialize Kafka producer
    producer = KafkaProducer(
        bootstrap_servers=bootstrap_servers,
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )

    # Load dataset
    try:
        df = pd.read_csv(data_path)
        logging.info(f"Loaded {len(df)} records")
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        # Use synthetic data as fallback
        df = None

    index = 0

    while True:
        try:
            # Use real data
            if index >= len(df):
                index = 0
                print("Restarting dataset from beginning")

            record = df.iloc[index].to_dict()
            # Convert NaN to None for JSON serialization
            record = {k: (None if pd.isna(v) else v) for k, v in record.items()}
            record['timestamp'] = datetime.now().isoformat()
            index += 1
            # Send to Kafka
            producer.send(topic, value=record)

            if index % 100 == 0:
                logging.info(f"Published {index} records")

            time.sleep(interval)

        except KeyboardInterrupt:
            logging.warning("Stopping producer...")
            break
        except Exception as e:
            logging.error(f"Error: {e}")
            time.sleep(1)

    producer.close()

if __name__ == '__main__':
    main()