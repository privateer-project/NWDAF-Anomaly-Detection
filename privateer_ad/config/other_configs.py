import os
import sys
import logging

from pathlib import Path
from dotenv import load_dotenv
from dataclasses import dataclass

load_dotenv(Path(__file__).parent.joinpath('.env'))

DATEFORMAT = '%Y-%m-%d %H:%M:%S.%f'

@dataclass
class PathsConf:
   root: Path = Path(os.environ.get('ROOT_DIR', Path(__file__).parents[2]))
   data: Path = Path(os.environ.get('DATA_DIR', root.joinpath('data')))
   config: Path = Path(__file__).parent
   raw: Path = Path(os.environ.get('RAW_DIR', data.joinpath('raw')))
   processed: Path = Path(os.environ.get('PROCESSED_DIR', data.joinpath('processed')))
   scalers: Path = Path(os.environ.get('SCALERS_DIR', root.joinpath('scalers')))
   analysis: Path = Path(os.environ.get('ANALYSIS_DIR', root.joinpath('analysis_results')))
   raw_dataset: Path = Path(os.environ.get('RAW_DATASET', raw.joinpath('amari_ue_data_merged_with_attack_number.csv')))
   experiments_dir: Path = Path(os.environ.get('EXPERIMENTS_DIR', root.joinpath('experiments')))

def setup_logger():
    lgr = logging.getLogger('privateer')
    lgr.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(PathsConf.root.joinpath('logs.log'))
    file_handler.setFormatter(formatter)

    lgr.addHandler(console_handler)
    lgr.addHandler(file_handler)
    return lgr

logger = setup_logger()

@dataclass
class DifferentialPrivacyConfig:
   target_epsilon: float = 2.0
   target_delta: float = 1e-7
   max_grad_norm: float = 2.0
   noise_multiplier: float = 0.2
   secure_mode: bool = False

@dataclass
class MLFlowConfig:
   track: bool = os.environ.get('MLFLOW_TRACKING', 'true').lower() == 'true'
   server_address: str = os.environ.get('MLFLOW_SERVER_ADDRESS', 'http://localhost:5001')
   experiment_name: str = os.environ.get('MLFLOW_EXPERIMENT_NAME',  'privateer-ad')

@dataclass
class AutotuneConfig:
   study_name: str = 'study_1'
   storage: str = 'optuna.db'
   n_trials: int = 10
   timeout: int = 3600 * 8  # 8 hours
   target: str = 'f1-score'
   direction: str = 'maximize'
   kwargs: dict = None

@dataclass
class PartitionConfig:
    num_partitions: int = 1
    num_classes_per_partition: int = 9
    partition_id: int = 0

@dataclass
class SecureAggregationConfig:
   num_shares: int = 3
   reconstruction_threshold: int = 2
