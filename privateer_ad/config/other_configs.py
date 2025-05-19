import os
import sys
import logging

from pathlib import Path
from dotenv import load_dotenv
from dataclasses import dataclass

from privateer_ad.config.data_config import PathsConf

load_dotenv(Path(__file__).parent.joinpath('.env'))

DATEFORMAT = '%Y-%m-%d %H:%M:%S.%f'


def setup_logger(name):
    logger = logging.getLogger(name)
    # Clear any existing handlers to prevent duplicates
    if logger.handlers:
        logger.handlers.clear()

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(PathsConf.root.joinpath('logs.log'))
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger

def _str2bool(value):
    return str(value).lower() in {'1', 'true', 'yes', 'y', 'on'}

@dataclass
class DifferentialPrivacyConfig:
    target_epsilon: float = 5.0  # Client-side privacy budget
    target_delta: float = 1e-6  # Privacy failure probability
    max_grad_norm: float = 4.0  # Per-sample gradient clipping threshold
    server_clipping_norm: float = 8.0  # Client update clipping threshold
    noise_multiplier: float = 0.4  # Server-side noise magnitude
    secure_mode: bool = True  # Enable secure RNG for DP

@dataclass
class MLFlowConfig:
   track: bool = _str2bool(os.environ.get('MLFLOW_ENABLE_TRACKING', True))
   server_address: str = os.environ.get('MLFLOW_SERVER_ADDRESS', 'http://localhost:5001')
   experiment_name: str = os.environ.get('MLFLOW_EXPERIMENT_NAME',  'privateer-ad')
   server_run_name: str = os.environ.get('MLFLOW_SERVER_RUN', 'federated_learning_lower_dp')


@dataclass
class SecureAggregationConfig:
   num_shares: float | int = 0.8
   reconstruction_threshold: float | int = 0.8
