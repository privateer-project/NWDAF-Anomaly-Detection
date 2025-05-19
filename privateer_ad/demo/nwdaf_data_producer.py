#!/usr/bin/env python3
"""
NWDAF Data Producer - Reads training dataset and publishes to Kafka
Part of the PRIVATEER Security Analytics framework
"""

import os
import sys
import json
import time
import argparse
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any
from kafka import KafkaProducer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('data-producer')


class NWDAFDataProducer:
    """
    Loads training data and publishes it to a Kafka topic to simulate
    the NWDAF data producer in a real system.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.producer = KafkaProducer(
            bootstrap_servers=config["KAFKA_BOOTSTRAP_SERVERS"],
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        self.running = False
        self.dataset_path = config["TRAINING_DATASET_PATH"]

        # Load dataset
        try:
            self.dataset = pd.read_csv(self.dataset_path)
            logger.info(f"Loaded dataset with {len(self.dataset)} records from {self.dataset_path}")

            # Convert timestamps to datetime if present
            if '_time' in self.dataset.columns:
                self.dataset['_time'] = pd.to_datetime(self.dataset['_time'])

            # Add attack label if not present (for simulation only)
            if 'attack' not in self.dataset.columns and 'attack_number' in self.dataset.columns:
                self.dataset['attack'] = (self.dataset['attack_number'] > 0).astype(int)

        except Exception as e:
            logger.error(f"Could not load dataset: {e}")
            logger.info("Using synthetic data instead")
            self.dataset = None

    def _convert_np_type(self, value):
        """Convert numpy types to Python native types for JSON serialization"""
        if isinstance(value, (np.integer, np.int64, np.int32)):
            return int(value)
        elif isinstance(value, (np.floating, np.float64, np.float32)):
            return float(value)
        elif isinstance(value, (np.ndarray,)):
            return value.tolist()
        elif isinstance(value, pd.Timestamp):
            return value.isoformat()
        elif isinstance(value, np.bool_):
            return bool(value)
        else:
            return value

    def _generate_synthetic_sample(self) -> Dict:
        """Generate a simulated network telemetry sample"""
        # This is a simplified example based on the NCSRD-DS-5GDDoS dataset structure
        # Used as fallback if real dataset is not available

        # Randomly decide if this will be an attack or normal traffic
        is_attack = np.random.random() < 0.05  # 5% chance of attack

        # Generate traffic metrics with different patterns for normal vs attack
        dl_bitrate = np.random.normal(5000, 1000) * (3 if is_attack else 1)
        ul_bitrate = np.random.normal(2000, 500) * (5 if is_attack else 1)
        dl_retx = np.random.poisson(2) * (4 if is_attack else 1)
        ul_tx = np.random.poisson(30) * (2 if is_attack else 1)

        return {
            "timestamp": datetime.now().isoformat(),
            "imeisv": f"86099604{np.random.randint(10000000, 99999999)}",
            "cell": f"cell-{np.random.randint(1, 4)}",
            "dl_bitrate": float(dl_bitrate),
            "ul_bitrate": float(ul_bitrate),
            "dl_retx": int(dl_retx),
            "ul_tx": int(ul_tx),
            "dl_tx": int(np.random.poisson(30)),
            "ul_retx": int(np.random.poisson(2)),
            "bearer_0_dl_total_bytes": float(np.random.normal(50000, 10000)),
            "bearer_0_ul_total_bytes": float(np.random.normal(20000, 5000)),
            "attack": int(is_attack),
            "attack_number": int(is_attack * np.random.randint(1, 6))
        }

    def run(self):
        """Main loop to generate and publish data"""
        self.running = True
        logger.info("Starting data production")

        if self.dataset is not None:
            # Use actual dataset
            index = 0
            try:
                while self.running:
                    # Get data point
                    if index >= len(self.dataset):
                        index = 0  # Loop back to start
                        logger.info("Reached end of dataset, looping back to beginning")

                    record = self.dataset.iloc[index].to_dict()

                    # Convert numpy types to Python native types for JSON serialization
                    record = {k: self._convert_np_type(v) for k, v in record.items()}

                    # Add timestamp if not present
                    if 'timestamp' not in record:
                        record['timestamp'] = datetime.now().isoformat()

                    # Publish to Kafka
                    self.producer.send(
                        self.config["KAFKA_RAW_DATA_TOPIC"],
                        value=record
                    )

                    if index % 100 == 0:
                        logger.info(f"Published record {index} to Kafka")

                    index += 1

                    # Simulate data generation interval
                    time.sleep(self.config["DATA_PUBLISH_INTERVAL"])
            except KeyboardInterrupt:
                logger.info("Stopping data production")
                self.running = False
        else:
            # Use synthetic data as fallback
            logger.info("Using synthetic data")
            try:
                while self.running:
                    # Generate a simulated sample
                    sample = self._generate_synthetic_sample()

                    # Publish to Kafka
                    self.producer.send(
                        self.config["KAFKA_RAW_DATA_TOPIC"],
                        value=sample
                    )

                    if np.random.random() < 0.01:  # Log occasionally
                        logger.info(f"Published synthetic data to Kafka")

                    # Simulate data generation interval
                    time.sleep(self.config["DATA_PUBLISH_INTERVAL"])
            except KeyboardInterrupt:
                logger.info("Stopping data production")
                self.running = False


def main():
    parser = argparse.ArgumentParser(description='NWDAF Data Producer')
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
            "DATA_PUBLISH_INTERVAL": float(os.environ.get("DATA_PUBLISH_INTERVAL", "0.5")),
            "TRAINING_DATASET_PATH": os.environ.get("TRAINING_DATASET_PATH",
                                                    "/app/data/amari_ue_data_merged_with_attack_number.csv")
        }
        logger.info("Using default configuration")

    # Create and run data producer
    producer = NWDAFDataProducer(config)
    producer.run()


if __name__ == "__main__":
    main()
