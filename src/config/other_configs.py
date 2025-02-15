import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.joinpath('.env'))

DATEFORMAT = '%Y-%m-%d %H:%M:%S.%f'

@dataclass
class ProjectPaths:
   root: Path = Path(os.environ.get('ROOT_DIR', Path(__file__).parents[2]))
   data: Path = Path(os.environ.get('DATA_DIR', root.joinpath('data')))
   config: Path = Path(__file__).parent
   raw: Path = Path(os.environ.get('RAW_DIR', data.joinpath('raw')))
   processed: Path = Path(os.environ.get('PROCESSED_DIR', data.joinpath('processed')))
   scalers: Path = Path(os.environ.get('SCALERS_DIR', root.joinpath('scalers')))
   analysis: Path = Path(os.environ.get('ANALYSIS_DIR', root.joinpath('analysis_results')))
   raw_dataset: Path = Path(os.environ.get('RAW_DATASET', raw.joinpath('amari_ue_data_merged_with_attack_number.csv')))
   studies: Path = Path(os.environ.get('STUDIES', root.joinpath('studies')))

@dataclass
class DifferentialPrivacyConfig:
   enable: bool = False
   target_epsilon: float = 2.0
   target_delta: float = 1e-7
   max_grad_norm: float = 2.0
   noise_multiplier: float = 0.2
   secure_mode: bool = False

@dataclass
class MLFlowConfig:
   track: bool = True
   server_address: str = os.environ.get('MLFLOW_SERVER_ADDRESS', 'http://localhost:5000')
   experiment_name: str = os.environ.get('MLFLOW_EXPERIMENT_NAME',  'privateer-ad')

@dataclass
class AutotuneConfig:
   study_name: str = 'study_0'
   storage: str = 'optuna.db'
   n_trials: int = 100
   timeout: int = 3600 * 8  # 8 hours
   target: str = 'val_loss'
   direction: str = 'minimize'

@dataclass
class PartitionConfig:
    num_partitions: int = 1
    num_classes_per_partition: int = 9
    partition_id: int = 0

@dataclass
class SecureAggregationConfig:
   num_shares: int = 3
   reconstruction_threshold: int = 2
   max_weight: float = 1000.0
   clipping_range: float = 8.0
   quantization_range: int = 4194304
   modulus_range: int = 4294967296
   timeout: float = 600.0

@dataclass
class FlowerConfig:
   enable: bool = False
   mode: str = "client"
   client_id: int = 0
   n_clients: int = 1
   server_address: str = "[::]:8081"
   num_rounds: int = 3
   num_clients_per_round: int = 2
   min_clients_per_round: int = 2
   fraction_fit: float = 1.0
   fraction_evaluate: float = 1.0