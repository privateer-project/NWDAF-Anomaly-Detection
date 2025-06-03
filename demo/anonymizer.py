"""
Anonymization Service - Applies privacy-preserving transformations
"""
import os
import json
from kafka import KafkaConsumer, KafkaProducer

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
            anonymized['ip'] = f"{ip_parts[0]}.{ip_parts[1]}.xxx.xxx"

        # Hash other identifiers
        for field in ['amf_ue_id', 'bearer_0_ip', 'bearer_1_ip', '5g_tmsi']:
            if field in anonymized and anonymized[field]:
                anonymized[field] = f"anon-{hash(str(anonymized[field])) % 10000}"

        anonymized['_anonymized'] = True
        return anonymized

def main():
    bootstrap_servers = os.environ.get('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092')
    input_topic = os.environ.get('INPUT_TOPIC', 'raw-network-data')
    output_topic = os.environ.get('OUTPUT_TOPIC', 'anonymized-data')

    print(f"Starting anonymization service...")
    print(f"Input: {input_topic}")
    print(f"Output: {output_topic}")

    consumer = KafkaConsumer(
        input_topic,
        bootstrap_servers=bootstrap_servers,
        value_deserializer=lambda v: json.loads(v.decode('utf-8')),
        auto_offset_reset='latest',
        group_id='anonymizer-group'
    )

    producer = KafkaProducer(
        bootstrap_servers=bootstrap_servers,
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )

    anonymizer = Anonymizer()
    count = 0

    try:
        for message in consumer:
            raw_data = message.value
            anonymized_data = anonymizer.anonymize(raw_data)

            producer.send(output_topic, value=anonymized_data)

            count += 1
            if count % 100 == 0:
                print(f"Anonymized {count} records")

    except KeyboardInterrupt:
        print("Stopping anonymizer...")
    except Exception as e:
        print(f"Error: {e}")

    consumer.close()
    producer.close()

if __name__ == '__main__':
    main()