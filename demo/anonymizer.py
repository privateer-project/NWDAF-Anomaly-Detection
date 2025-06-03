"""
Anonymization Service - Applies privacy-preserving transformations
"""
import os
import sys
import json
import time
from kafka import KafkaConsumer, KafkaProducer
from kafka.errors import NoBrokersAvailable

class Anonymizer:
    def anonymize(self, data):
        """Apply anonymization to sensitive fields"""
        anonymized = data.copy()

        # Hash device identifier
        if 'imeisv' in anonymized:
            anonymized['imeisv'] = f"anon-{hash(anonymized['imeisv']) % 10000}"

        # Mask IP addresses
        if 'ip' in anonymized:
            ip_parts = anonymized['ip'].split('.')
            if len(ip_parts) >= 2:
                anonymized['ip'] = f"{ip_parts[0]}.{ip_parts[1]}.xxx.xxx"

        # Hash other identifiers
        for field in ['amf_ue_id', 'bearer_0_ip', 'bearer_1_ip', '5g_tmsi']:
            if field in anonymized and anonymized[field]:
                anonymized[field] = f"anon-{hash(str(anonymized[field])) % 10000}"

        anonymized['_anonymized'] = True
        return anonymized

def wait_for_kafka(bootstrap_servers, max_retries=30):
    """Wait for Kafka to be available"""
    for i in range(max_retries):
        try:
            # Test connection
            test_producer = KafkaProducer(bootstrap_servers=bootstrap_servers)
            test_producer.close()
            print(f"✅ Connected to Kafka at {bootstrap_servers}")
            return True
        except NoBrokersAvailable:
            print(f"⏳ Waiting for Kafka... attempt {i+1}/{max_retries}")
            time.sleep(2)
        except Exception as e:
            print(f"❌ Kafka connection error: {e}")
            time.sleep(2)

    print(f"❌ Failed to connect to Kafka after {max_retries} attempts")
    return False

def main():
    # Force stdout to be unbuffered for Docker logging
    sys.stdout.reconfigure(line_buffering=True)

    bootstrap_servers = os.environ.get('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092')
    input_topic = os.environ.get('INPUT_TOPIC', 'raw-network-data')
    output_topic = os.environ.get('OUTPUT_TOPIC', 'anonymized-data')

    print("🔐 Starting PRIVATEER Anonymization Service...")
    print(f"📥 Input topic: {input_topic}")
    print(f"📤 Output topic: {output_topic}")
    print(f"🔌 Kafka servers: {bootstrap_servers}")

    # Wait for Kafka to be available
    if not wait_for_kafka(bootstrap_servers):
        sys.exit(1)

    # Create consumer and producer
    try:
        print("🔧 Setting up Kafka consumer and producer...")

        consumer = KafkaConsumer(
            input_topic,
            bootstrap_servers=bootstrap_servers,
            value_deserializer=lambda v: json.loads(v.decode('utf-8')),
            auto_offset_reset='earliest',  # Changed from 'latest' to process existing messages
            group_id='anonymizer-group',
            enable_auto_commit=True,
            consumer_timeout_ms=5000  # Add timeout to avoid hanging
        )

        producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )

        print("✅ Consumer and producer created successfully")

    except Exception as e:
        print(f"❌ Failed to create Kafka client: {e}")
        sys.exit(1)

    anonymizer = Anonymizer()
    count = 0
    last_log_time = time.time()

    print("🚀 Starting message processing...")
    print("⏳ Waiting for messages...")

    try:
        while True:
            # Poll for messages
            message_batch = consumer.poll(timeout_ms=1000)

            if not message_batch:
                # No messages received, log periodically to show it's alive
                current_time = time.time()
                if current_time - last_log_time > 30:  # Log every 30 seconds
                    print(f"💤 Still waiting for messages... (processed {count} so far)")
                    last_log_time = current_time
                continue

            # Process messages
            for topic_partition, messages in message_batch.items():
                for message in messages:
                    try:
                        raw_data = message.value
                        anonymized_data = anonymizer.anonymize(raw_data)

                        producer.send(output_topic, value=anonymized_data)

                        count += 1
                        if count % 50 == 0:  # Log more frequently
                            print(f"🔐 Anonymized {count} records")

                        # Log first few messages for debugging
                        if count <= 5:
                            print(f"📝 Sample anonymized record {count}: device={anonymized_data.get('imeisv', 'N/A')}")

                    except Exception as e:
                        print(f"❌ Error processing message: {e}")
                        continue

    except KeyboardInterrupt:
        print("🛑 Stopping anonymizer...")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
    finally:
        print("🧹 Cleaning up...")
        consumer.close()
        producer.close()
        print(f"✅ Anonymizer stopped. Total processed: {count}")

if __name__ == '__main__':
    main()