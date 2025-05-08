import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict
import yaml

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


@dataclass
class DeviceInfo:
   imeisv: str
   ip: str
   type: str
   malicious: bool
   in_attacks: List[str]

@dataclass
class AttackInfo:
   start: str
   stop: str

@dataclass
class FeatureInfo:
   dtype: str = 'str'
   drop: bool = False
   is_input: Optional[bool] = False
   process: List[str] = field(default_factory=lambda: list())

@dataclass
class MetaData:
   devices: Dict[str, DeviceInfo]
   attacks: Dict[str, AttackInfo]
   features: Dict[str, FeatureInfo]

   def __init__(self):
      super().__init__()
      with open(Path(__file__).parent.joinpath('metadata.yaml')) as f:
         data = yaml.safe_load(f)
      self.devices = {k: DeviceInfo(**v) for k, v in data['devices'].items()}
      self.attacks = {k: AttackInfo(**v) for k, v in data['attacks'].items()}
      self.features = {k: FeatureInfo(**v) for k, v in data['features'].items()}

   def get_input_features(self):
      return [feat for feat, info in self.features.items() if info.is_input]

   def get_drop_features(self):
      return [feat for feat, info in self.features.items() if info.drop]
