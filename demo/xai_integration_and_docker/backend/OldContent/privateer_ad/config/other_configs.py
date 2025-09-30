import os

from pathlib import Path
from dotenv import load_dotenv
from dataclasses import dataclass

load_dotenv(Path(__file__).parent.joinpath('.env'))

def _str2bool(value):
    return str(value).lower() in {'1', 'true', 'yes', 'y', 'on'}


@dataclass
class DPConfig:
    target_epsilon: float = 0.5  # Client-side privacy budget
    target_delta: float = 1e-7  # Privacy failure probability
    max_grad_norm: float = 0.5  # Per-sample gradient clipping threshold
    secure_mode: bool = True  # Enable secure RNG for DP
    enable: bool = False  # Enable DP


@dataclass
class MLFlowConfig:
   track: bool = _str2bool(os.environ.get('MLFLOW_ENABLE_TRACKING', True))
   server_address: str = os.environ.get('MLFLOW_SERVER_ADDRESS', 'http://localhost:5001')
   experiment_name: str = os.environ.get('MLFLOW_EXPERIMENT_NAME',  'privateer-ad')
   server_run_name: str = os.environ.get('MLFLOW_SERVER_RUN', 'federated-learning')


@dataclass
class SecureAggregationConfig:
   num_shares: float | int = 3
   reconstruction_threshold: float | int = 2


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
