import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.joinpath('.env'))

DATEFORMAT = '%Y-%m-%d %H:%M:%S.%f'

@dataclass
class Paths:
   root: Path = Path(os.environ.get('ROOT_DIR', Path(__file__).parents[2]))
   data: Path = Path(os.environ.get('DATA_DIR', root.joinpath('data')))
   config: Path = Path(__file__).parent
   raw: Path = Path(os.environ.get('RAW_DIR', data.joinpath('raw')))
   processed: Path = Path(os.environ.get('PROCESSED_DIR', data.joinpath('processed')))
   models: Path = Path(os.environ.get('MODELS_DIR', root.joinpath('architectures')))
   scalers: Path = Path(os.environ.get('SCALERS_DIR', root.joinpath('scalers')))
   analysis: Path = Path(os.environ.get('ANALYSIS_DIR', root.joinpath('analysis_results')))
   raw_dataset: Path = Path(os.environ.get('RAW_DATASET', raw.joinpath('amari_ue_data_merged_with_attack_number.csv')))
   studies: Path = Path(os.environ.get('STUDIES', root.joinpath('studies')))
   def __post_init__(self):
       for path in [self.data, self.raw, self.processed, self.models, self.scalers, self.analysis]:
           path.mkdir(parents=True, exist_ok=True)

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
   server_address: str = os.environ.get('MLFLOW_SERVER_ADDRESS', 'http://localhost:5555')
   experiment_name: str = os.environ.get('MLFLOW_EXPERIMENT_NAME',  'tfad')

@dataclass
class PartitionConfig:
    num_partitions: int = 1
    num_classes_per_partition: int = 9
    partition_id: int = 0
