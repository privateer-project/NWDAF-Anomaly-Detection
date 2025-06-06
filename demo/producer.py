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


class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if pd.isna(obj):
            return None
        if hasattr(obj, 'isoformat'):
            return obj.isoformat()
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        if isinstance(obj, (np.floating, float)):
            return float(obj)
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        return super().default(obj)


def create_demo_sequence(df):
    """Create a demonstration sequence with alternating benign/attack periods"""

    # Define demo windows (1 minute each)
    demo_windows = [
        # Window 1: Benign period (1 minute)
        {
            'name': 'benign_1',
            'duration': 60,  # seconds
            'filter': lambda df: (df['attack'] == 0) & (df['_time'] < '2024-08-18 07:00:00'),
            'sample_size': 60  # samples to send in this window
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
        # Window 5: Benign period (1 minute)
        {
            'name': 'benign_3',
            'duration': 60,
            'filter': lambda df: (df['attack'] == 0) &
                                 (df['_time'] >= '2024-08-19 10:00:00') &
                                 (df['_time'] <= '2024-08-19 11:00:00'),
            'sample_size': 60
        },
        # Window 6: UDP Fragmentation Attack (1 minute)
        {
            'name': 'udp_fragmentation',
            'duration': 60,
            'filter': lambda df: (df['attack'] == 1) &
                                 (df['_time'] >= '2024-08-19 17:00:00') &
                                 (df['_time'] <= '2024-08-19 18:00:00'),
            'sample_size': 60
        },
        # Window 7: Benign period (1 minute)
        {
            'name': 'benign_4',
            'duration': 60,
            'filter': lambda df: (df['attack'] == 0) &
                                 (df['_time'] >= '2024-08-20 10:00:00') &
                                 (df['_time'] <= '2024-08-20 11:00:00'),
            'sample_size': 60
        },
        # Window 8: DNS Flood Attack (1 minute)
        {
            'name': 'dns_flood',
            'duration': 60,
            'filter': lambda df: (df['attack'] == 1) &
                                 (df['_time'] >= '2024-08-21 12:00:00') &
                                 (df['_time'] <= '2024-08-21 13:00:00'),
            'sample_size': 60
        },
        # Window 9: Benign period (1 minute)
        {
            'name': 'benign_5',
            'duration': 60,
            'filter': lambda df: (df['attack'] == 0) &
                                 (df['_time'] >= '2024-08-21 14:00:00') &
                                 (df['_time'] <= '2024-08-21 15:00:00'),
            'sample_size': 60
        },
        # Window 10: GTP-U Flood Attack (1 minute)
        {
            'name': 'gtp_u_flood',
            'duration': 60,
            'filter': lambda df: (df['attack'] == 1) &
                                 (df['_time'] >= '2024-08-21 17:00:00') &
                                 (df['_time'] <= '2024-08-21 18:00:00'),
            'sample_size': 60
        }
    ]

    demo_data = []
    current_time = datetime.now()

    for i, window in enumerate(demo_windows):
        print(f"📊 Preparing {window['name']} window...")

        # Filter data for this window
        window_df = df[window['filter'](df)].copy()

        if len(window_df) == 0:
            print(f"⚠️ No data found for {window['name']}, using fallback")
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

        # Create synthetic temporal progression for demo
        window_samples = window_samples.copy()
        start_time = current_time + timedelta(seconds=i * window['duration'])

        # Create evenly spaced timestamps for this window
        timestamps = pd.date_range(
            start=start_time,
            periods=len(window_samples),
            freq=f"{window['duration'] / len(window_samples)}s"
        )

        window_samples['_demo_time'] = timestamps
        window_samples['_demo_window'] = window['name']
        window_samples['_demo_duration'] = window['duration']

        demo_data.append(window_samples)

        print(f"✅ {window['name']}: {len(window_samples)} samples, "
              f"attack rate: {(window_samples['attack'].sum() / len(window_samples) * 100):.1f}%")

    # Combine all windows
    full_demo = pd.concat(demo_data, ignore_index=True)

    print(f"\n🎬 Demo sequence created: {len(full_demo)} total samples")
    print(f"📈 Overall attack rate: {(full_demo['attack'].sum() / len(full_demo) * 100):.1f}%")
    print(f"⏱️ Total demo duration: {len(demo_windows)} minutes")

    return full_demo


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
        value_serializer=lambda v: json.dumps(v, cls=CustomJSONEncoder).encode('utf-8')
    )

    # Load dataset
    try:
        print("📂 Loading dataset...")
        df = pd.read_csv(csv_path, parse_dates=['_time'])
        print(f"✅ Loaded {len(df)} records")
        print(f"📅 Time range: {df['_time'].min()} to {df['_time'].max()}")

        # Count attack types
        attack_count = df[df['attack'] == 1].shape[0]
        benign_count = df[df['attack'] == 0].shape[0]
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

        if demo_mode:
            # Create demo sequence with alternating windows
            demo_df = create_demo_sequence(df)
            records_to_send = demo_df
        else:
            # Normal mode - send all data in order
            records_to_send = df.copy()
            records_to_send = records_to_send.sort_values('_time').reset_index(drop=True)

        # Send records
        current_window = None
        window_start_time = time.time()
        samples_in_window = 0

        for idx, row in records_to_send.iterrows():
            # Convert row to dict
            record = {}
            for k, v in row.items():
                if pd.isna(v):
                    record[k] = None
                elif k == '_time':
                    record[k] = v.isoformat() if hasattr(v, 'isoformat') else str(v)
                elif hasattr(v, 'isoformat'):
                    record[k] = v.isoformat()
                elif isinstance(v, (bool, np.bool_)):
                    record[k] = bool(v)
                elif isinstance(v, (np.integer, int)):
                    record[k] = int(v)
                elif isinstance(v, (np.floating, float)):
                    record[k] = float(v)
                else:
                    record[k] = v

            # Add processing timestamp
            record['_timestamp'] = datetime.now().isoformat()
            record['_cycle'] = cycle_count

            # Track demo windows
            if demo_mode and '_demo_window' in row:
                if current_window != row['_demo_window']:
                    if current_window:
                        elapsed = time.time() - window_start_time
                        print(f"✅ Completed {current_window} window "
                              f"({samples_in_window} samples in {elapsed:.1f}s)")

                    current_window = row['_demo_window']
                    window_start_time = time.time()
                    samples_in_window = 0

                    # Announce new window
                    attack_type = "🔴 ATTACK" if 'benign' not in current_window else "🟢 BENIGN"
                    print(f"\n{attack_type} Starting {current_window} window...")

                samples_in_window += 1

            # Send to Kafka
            producer.send(topic, value=record)

            # Calculate delay for demo mode
            if demo_mode:
                # Calculate time per sample to fit window duration
                window_duration = row.get('_demo_duration', 60)
                window_sample_count = 60  # samples per window
                delay = window_duration / window_sample_count
            else:
                # Normal mode - small fixed delay
                delay = 0.1

            time.sleep(delay)

            # Progress updates
            if (idx + 1) % 50 == 0:
                if demo_mode and current_window:
                    print(f"📤 Sent {idx + 1} records (current: {current_window})")
                else:
                    print(f"📤 Sent {idx + 1} records")

        # Complete last window
        if demo_mode and current_window:
            elapsed = time.time() - window_start_time
            print(f"✅ Completed {current_window} window "
                  f"({samples_in_window} samples in {elapsed:.1f}s)")

        print(f"\n🎉 Cycle #{cycle_count} completed! Total sent: {len(records_to_send)} records")
        print("⏳ Waiting 5 seconds before next cycle...")
        time.sleep(5)

    producer.close()


if __name__ == '__main__':
    main()