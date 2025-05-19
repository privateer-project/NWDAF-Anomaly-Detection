#!/usr/bin/env python3
"""
MISP Client - Sends anomaly reports to MISP threat sharing platform
Part of the PRIVATEER Security Analytics framework
"""

import os
import sys
import json
import time
import argparse
import logging
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('misp-client')


class MISPClient:
    """
    Client for sharing detected anomalies with MISP using provided scripts.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.api_url = config.get("MISP_URL", "https://cc-cracs-201.inesctec.pt")
        self.api_key = config.get("MISP_API_KEY", "")

        # Paths for scripts and temporary files
        self.scripts_path = Path(config.get("MISP_SCRIPTS_PATH", "/app/misp_scripts"))
        self.temp_dir = Path(config.get("MISP_TEMP_DIR", "/app/temp"))

        # Create directories if they don't exist
        os.makedirs(self.temp_dir, exist_ok=True)

        # Create API key file for scripts
        self._create_key_file()

        logger.info(f"MISP Client initialized for server: {self.api_url}")

    def _create_key_file(self):
        """Create a key.py file with the API key"""
        try:
            # Create key.py in the scripts directory
            key_path = self.scripts_path / "key.py"
            with open(key_path, "w") as f:
                f.write(f'API_KEY = "{self.api_key}"  # API key')
            logger.info("API key file created")
        except Exception as e:
            logger.error(f"Could not create API key file: {e}")

    def report_anomaly(self, data: Dict):
        """
        Report an anomaly to MISP by creating an event using the provided scripts
        """
        try:
            # 1. Create event JSON file
            event_data = {
                "Event": {
                    "info": f"PRIVATEER Anomaly: {data.get('imeisv', 'unknown')}",
                    "distribution": 0,  # Your organization only
                    "threat_level_id": 2,  # Medium (1=High, 2=Medium, 3=Low, 4=Undefined)
                    "analysis": 1,  # Initial (0=Initial, 1=Ongoing, 2=Completed)
                    "date": data.get('timestamp', datetime.now().isoformat())[:10]  # Just the date part
                }
            }

            event_file = self.temp_dir / "anomaly_event.json"
            with open(event_file, "w") as f:
                json.dump(event_data, f, indent=4)

            # 2. Create the event using add-event.py
            add_event_script = self.scripts_path / "add-event.py"
            result = subprocess.run(
                ["python", str(add_event_script), str(event_file)],
                capture_output=True, text=True
            )

            if "Error" in result.stdout or result.returncode != 0:
                logger.error(f"Error creating MISP event: {result.stdout}")
                return

            # 3. Extract event ID from response
            response_text = result.stdout
            try:
                response_data = json.loads(response_text.split("Response: ")[1])
                event_id = response_data.get("Event", {}).get("id")
                if not event_id:
                    logger.error(f"Could not extract event ID from response: {response_text}")
                    return
                logger.info(f"Created MISP event with ID: {event_id}")
            except Exception as e:
                logger.error(f"Error parsing event creation response: {e}")
                return

            # 4. Add attributes to the event
            # Create attribute JSON for anomaly score
            attribute_data = {
                "Attribute": [
                    {
                        "type": "float",
                        "category": "Other",
                        "distribution": 0,
                        "value": str(data.get('anomaly_score')),
                        "comment": "Anomaly Score"
                    },
                    {
                        "type": "text",
                        "category": "Other",
                        "distribution": 0,
                        "value": f"Affected device: {data.get('imeisv')}",
                        "comment": "Device Identifier"
                    },
                    {
                        "type": "datetime",
                        "category": "Other",
                        "distribution": 0,
                        "value": data.get('timestamp'),
                        "comment": "Detection Time"
                    }
                ]
            }

            # Add network metrics as attributes
            for metric_name, metric_value in data.get('traffic_metrics', {}).items():
                attribute_data["Attribute"].append({
                    "type": "float",
                    "category": "Network traffic",
                    "distribution": 0,
                    "value": str(metric_value),
                    "comment": f"Traffic Metric: {metric_name}"
                })

            # Save attribute data to file
            attribute_file = self.temp_dir / "anomaly_attributes.json"
            with open(attribute_file, "w") as f:
                json.dump(attribute_data, f, indent=4)

            # Add attributes to event
            add_attribute_script = self.scripts_path / "add-attribute-to-event.py"
            result = subprocess.run(
                ["python", str(add_attribute_script), str(event_id), str(attribute_file)],
                capture_output=True, text=True
            )

            if "Error" in result.stdout or result.returncode != 0:
                logger.error(f"Error adding attributes to MISP event: {result.stdout}")
                return

            logger.info(f"Successfully added attributes to MISP event {event_id}")

        except Exception as e:
            logger.error(f"Error reporting to MISP: {e}")
            # In production, would add retry logic or queue the event


def main():
    parser = argparse.ArgumentParser(description='MISP Client Service')
    parser.add_argument('--config', type=str, default='/app/config.json',
                        help='Path to configuration file')

    # This component is typically called from the analytics agent
    # but can also be run as a standalone service for testing

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
            "MISP_URL": os.environ.get("MISP_URL", "https://cc-cracs-201.inesctec.pt"),
            "MISP_API_KEY": os.environ.get("MISP_API_KEY", ""),
            "MISP_SCRIPTS_PATH": os.environ.get("MISP_SCRIPTS_PATH", "/app/misp_scripts"),
            "MISP_TEMP_DIR": os.environ.get("MISP_TEMP_DIR", "/app/temp")
        }
        logger.info("Using default configuration")

    # Create MISP client
    client = MISPClient(config)

    # Test function - uncomment to test
    """
    test_data = {
        "timestamp": datetime.now().isoformat(),
        "imeisv": "test-device-123",
        "cell": "cell-1",
        "anomaly_score": 0.95,
        "traffic_metrics": {
            "dl_bitrate": 15000.0,
            "ul_bitrate": 10000.0,
            "dl_retx": 8,
            "ul_tx": 60
        }
    }
    client.report_anomaly(test_data)
    """

    logger.info("MISP Client service started in standalone mode")
    # In standalone mode, this doesn't do much - it's meant to be imported by the analytics agent
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        logger.info("MISP Client service stopped")


if __name__ == "__main__":
    main()
