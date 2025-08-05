# demo/producer.py
"""
Data Producer - Reads dataset and sends benign/attack data in alternating intervals
Creates a demo sequence with 1-minute periods of different traffic types
"""
import os
import json
import time
import pandas as pd
import numpy as np
from kafka import KafkaProducer
from datetime import datetime, timedelta

def main():
    bootstrap_servers = os.environ.get('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092')
    topic = os.environ.get('KAFKA_TOPIC', 'raw-network-data')
    csv_path = os.environ.get('CSV_PATH', '/app/data/raw/amari_ue_data_merged_with_attack_number.csv')
    demo_mode = os.environ.get('DEMO_MODE', 'true').lower() == 'true'

    print(f"🚀 Starting data producer in {'DEMO' if demo_mode else 'NORMAL'} mode")
    print(f"📡 Kafka: {bootstrap_servers}")
    print(f"📤 Topic: {topic}")
    print(f"📁 Data: {csv_path}")

    # Initialize Kafka producer with custom encoder
    producer = KafkaProducer(
        bootstrap_servers=bootstrap_servers,
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )

    # Load dataset
    try:
        print("📂 Loading dataset...")
        df = pd.read_csv(csv_path, parse_dates=['_time'])
        print(f"✅ Loaded {len(df)} records")
        print(f"📅 Time range: {df['_time'].min()} to {df['_time'].max()}")

        # Count attack types
        attack_count = len(df[df['attack'] == 1])
        benign_count = len(df[df['attack'] == 0])
        print(f"🟢 Benign samples: {benign_count}")
        print(f"🔴 Attack samples: {attack_count}")

    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return

    # Main loop
    cycle_count = 0

    while True:
        cycle_count += 1
        print(f"\n🔄 Starting cycle #{cycle_count}")
        # Normal mode - send all data in order
        records_to_send = df.copy().sort_values('_time').reset_index(drop=True)
        counter = 0
        for time_stamp, sample in records_to_send.groupby('_time'):
            counter += len(sample)
            sample['_time'] = sample['_time'].map(lambda x: x.isoformat())
            producer.send(topic, value=sample.to_dict('list'))
            time.sleep(0.1)
            # Progress updates
            if counter % 50 == 0:
                print(f"📤 Sent {counter} records")

        # Complete last window
        print(f"\n🎉 Cycle #{cycle_count} completed! Total sent: {len(records_to_send)} records")
        print("⏳ Waiting 5 seconds before next cycle...")
        time.sleep(5)
    producer.close()


if __name__ == '__main__':
    main()