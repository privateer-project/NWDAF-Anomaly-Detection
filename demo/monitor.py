# demo/monitor.py
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
        value_deserializer=lambda v: json.loads(v.decode('utf-8'))
    )
    
    alert_count = 0
    
    for message in consumer:
        alert = message.value
        alert_count += 1
        
        timestamp = datetime.fromisoformat(alert['timestamp'])
        print(f"\n🚨 ALERT #{alert_count} - {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   Device: {alert['device_id']}")
        print(f"   Error: {alert['reconstruction_error']:.4f} (threshold: {alert['threshold']:.4f})")
        print(f"   Anomalies in window: {alert['anomaly_count']}")
        print(f"   Attack Type: {'ATTACK' if alert['metadata']['attack'] == 1 else 'Normal'}")
        print("-" * 80)
    
    consumer.close()

if __name__ == '__main__':
    main()