"""
Data Producer - Reads dataset and publishes to Kafka with demo attack windows
"""
import os
import json
import time
import logging
from datetime import datetime

import pandas as pd
from kafka import KafkaProducer

from privateer_ad.config.settings import PathConfig
from privateer_ad.config.metadata import MetadataRegistry

def create_demo_sequence(df):
    """Create a 5-minute demo sequence with attack periods"""

    # Define demo sequence: benign -> attack1 -> benign -> attack2 -> benign -> attack3
    demo_windows = [
        # Window 1: Benign period (1 minute)
        {
            'name': 'benign_1',
            'duration': 60,  # seconds
            'filter': lambda df: (df['attack'] == 0) & (df['_time'] < '2024-08-18 07:00:00'),
            'sample_size': 60
        },
        # Window 2: SYN Flood Attack (1 minute)
        {
            'name': 'syn_flood',
            'duration': 60,
            'filter': lambda df: (df['attack'] == 1) &
                                (df['_time'] >= '2024-08-18 07:00:00') &
                                (df['_time'] <= '2024-08-18 08:00:00'),
            'sample_size': 60
        },
        # Window 3: Benign period (1 minute)
        {
            'name': 'benign_2',
            'duration': 60,
            'filter': lambda df: (df['attack'] == 0) &
                                (df['_time'] >= '2024-08-18 09:00:00') &
                                (df['_time'] <= '2024-08-18 10:00:00'),
            'sample_size': 60
        },
        # Window 4: ICMP Flood Attack (1 minute)
        {
            'name': 'icmp_flood',
            'duration': 60,
            'filter': lambda df: (df['attack'] == 1) &
                                (df['_time'] >= '2024-08-19 07:00:00') &
                                (df['_time'] <= '2024-08-19 09:41:00'),
            'sample_size': 60
        },
        # Window 5: Final benign period (1 minute)
        {
            'name': 'benign_3',
            'duration': 60,
            'filter': lambda df: (df['attack'] == 0) &
                                (df['_time'] >= '2024-08-19 10:00:00') &
                                (df['_time'] <= '2024-08-19 11:00:00'),
            'sample_size': 60
        }
    ]

    demo_data = []

    for window in demo_windows:
        logging.info(f"📊 Preparing {window['name']} window...")

        # Filter data for this window
        window_df = df[window['filter'](df)].copy()

        if len(window_df) == 0:
            logging.warning(f"⚠️ No data found for {window['name']}, using fallback")
            # Fallback to any benign/attack data
            if 'benign' in window['name']:
                window_df = df[df['attack'] == 0].copy()
            else:
                window_df = df[df['attack'] == 1].copy()

        # Sample data for this window
        if len(window_df) > window['sample_size']:
            window_samples = window_df.sample(n=window['sample_size'], random_state=42)
        else:
            # Repeat samples if not enough data
            repeats = (window['sample_size'] // len(window_df)) + 1
            window_samples = pd.concat([window_df] * repeats).iloc[:window['sample_size']]

        # Add window metadata
        window_samples = window_samples.copy()
        window_samples['demo_window'] = window['name']
        window_samples['demo_duration'] = window['duration']

        demo_data.append(window_samples)

        logging.info(f"✅ {window['name']}: {len(window_samples)} samples, "
                    f"attack rate: {(window_samples['attack'].sum() / len(window_samples) * 100):.1f}%")

    # Combine all windows
    full_demo = pd.concat(demo_data, ignore_index=True)

    logging.info(f"🎬 Demo sequence created: {len(full_demo)} total samples")
    logging.info(f"📈 Overall attack rate: {(full_demo['attack'].sum() / len(full_demo) * 100):.1f}%")

    return full_demo

def main():
    bootstrap_servers = os.environ.get('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092')
    topic = os.environ.get('KAFKA_TOPIC', 'raw-network-data')
    data_path = PathConfig().raw_dataset
    interval = float(os.environ.get('INTERVAL', '1.0'))  # 1 second default for demo
    demo_mode = os.environ.get('DEMO_MODE', 'true').lower() == 'true'

    logging.basicConfig(level=logging.INFO)
    logging.info(f"🚀 Starting PRIVATEER demo data producer...")
    logging.info(f"📡 Kafka: {bootstrap_servers}")
    logging.info(f"📋 Topic: {topic}")
    logging.info(f"📁 Data: {data_path}")
    logging.info(f"🎬 Demo mode: {demo_mode}")
    logging.info(f"⏱️ Interval: {interval}s")

    # Initialize Kafka producer
    producer = KafkaProducer(
        bootstrap_servers=bootstrap_servers,
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )

    # Load dataset
    try:
        df = pd.read_csv(data_path, parse_dates=['_time'])
        logging.info(f"📊 Loaded {len(df)} records from dataset")

        # Ensure attack column exists and is properly set
        if 'attack' not in df.columns:
            df['attack'] = 0

        # Update attack labels based on metadata
        metadata = MetadataRegistry()
        for attack_num, attack_info in metadata.ATTACKS.items():
            start_time = pd.to_datetime(attack_info.start)
            stop_time = pd.to_datetime(attack_info.stop)
            attack_mask = (df['_time'] >= start_time) & (df['_time'] <= stop_time)
            df.loc[attack_mask, 'attack'] = 1
            df.loc[attack_mask, 'attack_number'] = attack_num

        logging.info(f"🎯 Attack samples: {df['attack'].sum()}")
        logging.info(f"😇 Benign samples: {len(df) - df['attack'].sum()}")

    except Exception as e:
        logging.error(f"❌ Error loading data: {e}")
        return

    # Create demo sequence if in demo mode
    if demo_mode:
        df = create_demo_sequence(df)

        # Calculate timing for 5-minute demo
        total_samples = len(df)
        demo_duration = 5 * 60  # 5 minutes in seconds
        interval = demo_duration / total_samples
        logging.info(f"🎬 Demo: {total_samples} samples over {demo_duration/60:.1f} minutes")
        logging.info(f"⏱️ Adjusted interval: {interval:.2f}s per sample")

    index = 0
    total_sent = 0
    start_time = time.time()

    try:
        while True:
            # Restart from beginning when reaching end
            if index >= len(df):
                index = 0
                if demo_mode:
                    logging.info("🔄 Demo sequence completed, restarting...")
                    time.sleep(5)  # 5 second pause between demo cycles

            record = df.iloc[index].to_dict()

            # Convert timestamps and NaN values for JSON serialization
            for key, value in record.items():
                if pd.isna(value):
                    record[key] = None
                elif isinstance(value, pd.Timestamp):
                    record[key] = value.isoformat()
                elif isinstance(value, (pd.Int64Dtype, pd.Float64Dtype)):
                    record[key] = float(value) if not pd.isna(value) else None

            # Add producer metadata
            record['producer_timestamp'] = datetime.now().isoformat()
            record['demo_index'] = index
            record['demo_mode'] = demo_mode

            # Send to Kafka
            producer.send(topic, value=record)
            total_sent += 1
            index += 1

            # Logging
            if total_sent % 50 == 0:
                elapsed = time.time() - start_time
                rate = total_sent / elapsed
                attack_status = "🚨 ATTACK" if record.get('attack', 0) == 1 else "😇 benign"

                logging.info(f"📤 Sent {total_sent} records ({rate:.1f}/s) - "
                           f"Current: {attack_status} - "
                           f"Device: {record.get('imeisv', 'unknown')} - "
                           f"Window: {record.get('demo_window', 'N/A')}")

            # Special logging for attack transitions in demo mode
            if demo_mode and index > 0:
                prev_attack = df.iloc[index-1]['attack'] if index > 0 else 0
                curr_attack = record.get('attack', 0)
                if prev_attack != curr_attack:
                    if curr_attack == 1:
                        logging.warning(f"🚨 ATTACK STARTED: {record.get('demo_window', 'unknown')} - Sample {index}")
                    else:
                        logging.info(f"😇 ATTACK ENDED: Returning to benign - Sample {index}")

            time.sleep(interval)

    except KeyboardInterrupt:
        logging.info("🛑 Stopping producer...")
    # except Exception as e:
    #     logging.error(f"❌ Error: {e}")
    finally:
        producer.close()
        logging.info(f"✅ Producer stopped. Total sent: {total_sent}")

if __name__ == '__main__':
    main()