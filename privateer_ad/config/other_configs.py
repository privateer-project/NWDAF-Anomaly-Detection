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
    return str(value).lower() in {'1', 'true', 'yes', 'on'}


@dataclass
class DifferentialPrivacyConfig:
   target_epsilon: float = 2.0     # This value is eventually used to compute the noise_multiplier used client-side through opacus.make_private_with_epsilon(...)
   target_delta: float = 1e-7      # This value is eventually used to compute the noise_multiplier used client-side through opacus.make_private_with_epsilon(...)
   max_grad_norm: float = 2.0      # This value is only used client-side, but is set both to DifferentialPrivacyClientSideFixedClipping *and* opacus.make_private_with_epsilon(...)
   noise_multiplier: float = 0.2   # This value is only used client-side and is used as input to DifferentialPrivacyClientSideFixedClipping
   secure_mode: bool = False

@dataclass
class MLFlowConfig:
   track: bool = os.environ.get('MLFLOW_TRACKING', 'true').lower() == 'true'
   server_address: str = os.environ.get('MLFLOW_SERVER_ADDRESS', 'http://localhost:5001')
   experiment_name: str = os.environ.get('MLFLOW_EXPERIMENT_NAME',  'privateer-ad')

@dataclass
class PartitionConfig:
    num_partitions: int = 1
    num_classes_per_partition: int = 9
    partition_id: int = 0

@dataclass
class SecureAggregationConfig:
   num_shares: int = 3
   reconstruction_threshold: int = 2
