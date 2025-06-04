"""
Data Producer - Reads dataset and publishes to Kafka with proper temporal ordering
"""
import os
import json
import time
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from kafka import KafkaProducer
from privateer_ad.config.settings import PathConfig

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
    """Create a 5-minute demo sequence with attack periods"""
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
    current_time = datetime.now()

    for i, window in enumerate(demo_windows):
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

        # Create synthetic temporal progression for demo
        window_samples = window_samples.copy()
        start_time = current_time + timedelta(seconds=i * window['duration'])

        # Create evenly spaced timestamps for this window
        timestamps = pd.date_range(
            start=start_time,
            periods=len(window_samples),
            freq=f"{window['duration']/len(window_samples)}S"
        )

        window_samples['_time'] = timestamps
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
    interval = float(os.environ.get('INTERVAL', '0.5'))
    demo_mode = os.environ.get('DEMO_MODE', 'true').lower() == 'true'
    time_acceleration = float(os.environ.get('TIME_ACCELERATION', '1.0'))

    logging.info(f"Starting data producer...")
    logging.info(f"Kafka: {bootstrap_servers}")
    logging.info(f"Topic: {topic}")
    logging.info(f"Data: {data_path}")
    logging.info(f"Time acceleration: {time_acceleration}x")
    logging.info(f"🎬 Demo mode: {demo_mode}")

    # Initialize Kafka producer with custom encoder
    producer = KafkaProducer(
        bootstrap_servers=bootstrap_servers,
        value_serializer=lambda v: json.dumps(v, cls=CustomJSONEncoder).encode('utf-8')
    )

    # Load and process dataset
    try:
        df = pd.read_csv(data_path, parse_dates=['_time'])

        if demo_mode:
            df = create_demo_sequence(df)
        else:
            # Sort by time to ensure temporal ordering
            df = df.sort_values('_time').reset_index(drop=True)

        logging.info(f"Loaded {len(df)} records")
        logging.info(f"Time range: {df['_time'].min()} to {df['_time'].max()}")
        logging.info(f"Devices: {df['imeisv'].unique()}")

    except Exception as e:
        logging.error(f"Error loading data: {e}")
        return

    # Calculate time deltas for realistic playback
    df['time_delta'] = df['_time'].diff().dt.total_seconds().fillna(0)

    # Cap max delays to prevent long waits
    max_delay = 10 if demo_mode else 60
    df['time_delta'] = df['time_delta'].clip(upper=max_delay)

    index = 0
    records_sent = 0
    start_time = datetime.now()

    logging.info("🚀 Starting temporal playback...")

    while True:
        try:
            # Restart from beginning when done
            if index >= len(df):
                if demo_mode:
                    logging.info("🔄 Demo sequence completed, restarting...")
                else:
                    logging.info("🔄 Dataset completed, restarting...")
                index = 0
                start_time = datetime.now()

            row = df.iloc[index]
            record = {}

            # Convert all fields safely
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

            # Add metadata
            record['original_time'] = record['_time']
            record['processing_timestamp'] = datetime.now().isoformat()
            if demo_mode and 'demo_window' in row:
                record['demo_phase'] = row['demo_window']

            # Send to Kafka
            producer.send(topic, value=record)
            records_sent += 1

            if records_sent % 50 == 0:
                current_phase = record.get('demo_phase', 'normal')
                logging.info(f"📤 Published {records_sent} records (phase: {current_phase})")

            # Calculate delay
            if index > 0:
                time_gap = df.iloc[index]['time_delta']
                delay = max(interval, time_gap / time_acceleration)
            else:
                delay = interval

            time.sleep(delay)
            index += 1

        except KeyboardInterrupt:
            logging.warning("⏹️ Stopping producer...")
            break
        except Exception as e:
            logging.error(f"❌ Error: {e}")
            time.sleep(1)

    producer.close()

if __name__ == '__main__':
    main()