from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict
import yaml

@dataclass
class DeviceInfo:
   imeisv: str
   ip: str
   type: str
   malicious: bool
   in_attacks: List[int]

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
   attacks: Dict[int, AttackInfo]
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

   def get_features_dtypes(self):
      return {feat: info.dtype for feat, info in self.features.items()}
