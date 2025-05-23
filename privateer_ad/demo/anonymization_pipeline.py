"""
Anonymization Pipeline - Processes raw data and adds privacy-preserving features
"""

import os
import sys
import json
import time
import argparse
import logging
from typing import Dict, Any
from kafka import KafkaProducer, KafkaConsumer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('anonymization')


class AnonymizationPipeline:
    """
    Pipeline for anonymizing sensitive data before sending it to the analytics agent.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.consumer = KafkaConsumer(
            self.config["KAFKA_RAW_DATA_TOPIC"],
            bootstrap_servers=self.config["KAFKA_BOOTSTRAP_SERVERS"],
            value_deserializer=lambda v: json.loads(v.decode('utf-8')),
            auto_offset_reset='latest',
            group_id='anonymization-pipeline'
        )
        self.producer = KafkaProducer(
            bootstrap_servers=self.config["KAFKA_BOOTSTRAP_SERVERS"],
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        self.running = False

    def _anonymize(self, data: Dict) -> Dict:
        """
        Apply anonymization to sensitive data fields.
        In a real implementation, this would apply sophisticated privacy-preserving mechanisms.
        """
        # Create a copy to avoid modifying the original
        anonymized = data.copy()

        # Apply anonymization techniques to sensitive fields
        if 'imeisv' in anonymized:
            # Hash the device identifier
            anonymized['imeisv'] = f"anon-{hash(anonymized['imeisv']) % 10000}"

        if 'ip' in anonymized:
            # Mask the IP address
            ip_parts = anonymized['ip'].split('.')
            anonymized['ip'] = f"{ip_parts[0]}.{ip_parts[1]}.xxx.xxx"

        if 'amf_ue_id' in anonymized:
            # Hash the AMF UE ID
            anonymized['amf_ue_id'] = f"anon-{hash(anonymized['amf_ue_id']) % 10000}"

        if 'bearer_0_ip' in anonymized:
            # Mask the bearer IP
            ip_parts = anonymized['bearer_0_ip'].split('.')
            anonymized['bearer_0_ip'] = f"{ip_parts[0]}.{ip_parts[1]}.xxx.xxx"

        # Add processing flag
        anonymized['_anonymized'] = True

        return anonymized

    def run(self):
        """Main processing loop"""
        self.running = True
        logger.info("Starting anonymization pipeline")

        try:
            for message in self.consumer:
                if not self.running:
                    break

                # Get the raw data
                raw_data = message.value

                # Apply anonymization
                anonymized_data = self._anonymize(raw_data)

                # Publish anonymized data
                self.producer.send(
                    self.config["KAFKA_ANONYMIZED_DATA_TOPIC"],
                    value=anonymized_data
                )

                # Occasionally log progress
                if hash(str(raw_data)) % 100 == 0:
                    logger.info("Processed and anonymized data record")

        except KeyboardInterrupt:
            logger.info("Stopping anonymization pipeline")
            self.running = False
        except Exception as e:
            logger.error(f"Error in anonymization pipeline: {e}")
            self.running = False


def main():
    parser = argparse.ArgumentParser(description='Anonymization Pipeline')
    parser.add_argument('--config', type=str, default='/app/config.json',
                        help='Path to configuration file')
    args = parser.parse_args()

    # Load configuration
    try:
        with open(args.config, 'r') as f:
            config = json.load(f)
        logger.info(f"Loaded configuration from {args.config}")
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        # Use default configuration
        config = {
            "KAFKA_BOOTSTRAP_SERVERS": os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092"),
            "KAFKA_RAW_DATA_TOPIC": os.environ.get("KAFKA_RAW_DATA_TOPIC", "nwdaf-raw-data"),
            "KAFKA_ANONYMIZED_DATA_TOPIC": os.environ.get("KAFKA_ANONYMIZED_DATA_TOPIC", "anonymized-data")
        }
        logger.info("Using default configuration")

    # Create and run anonymization pipeline
    pipeline = AnonymizationPipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()
