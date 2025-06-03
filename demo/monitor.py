"""
Alert Monitor - Displays anomaly alerts
"""
import os
import json
from kafka import KafkaConsumer
from datetime import datetime

def main():
    bootstrap_servers = os.environ.get('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092')
    alert_topic = os.environ.get('ALERT_TOPIC', 'anomaly-alerts')

    print(f"Starting alert monitor...")
    print(f"Listening to: {alert_topic}")
    print("-" * 80)

    consumer = KafkaConsumer(
        alert_topic,
        bootstrap_servers=bootstrap_servers,
        value_deserializer=lambda v: json.loads(v.decode('utf-8')),
        auto_offset_reset='latest',
        group_id='monitor-group'
    )

    alert_count = 0

    try:
        for message in consumer:
            alert = message.value
            alert_count += 1

            timestamp = datetime.fromisoformat(alert['timestamp'])
            print(f"\n🚨 ALERT #{alert_count} - {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"   Device: {alert['device_id']}")
            print(f"   Score: {alert['score']:.4f} (threshold: {alert['threshold']:.4f})")

            # Show key metrics from data
            data = alert.get('data', {})
            print(f"   Metrics:")
            print(f"     - DL Bitrate: {data.get('dl_bitrate', 'N/A')}")
            print(f"     - UL Bitrate: {data.get('ul_bitrate', 'N/A')}")
            print(f"     - DL Retx: {data.get('dl_retx', 'N/A')}")
            print(f"     - True Label: {'Attack' if data.get('attack') == 1 else 'Benign'}")
            print("-" * 80)

    except KeyboardInterrupt:
        print("\nStopping monitor...")
    except Exception as e:
        print(f"Error: {e}")

    consumer.close()
    print(f"\nTotal alerts received: {alert_count}")

if __name__ == '__main__':
    main()
