from dataclasses import dataclass, field
from typing import List, Optional, Dict
import yaml
from config import Paths


@dataclass
class DeviceInfo:
   imeisv: str
   ip: str
   type: str
   malicious: bool

@dataclass
class AttackInfo:
   start: str
   stop: str
   devices: List[str]

@dataclass
class FeatureInfo:
   dtype: str = 'str'
   drop: bool = False
   input: Optional[bool] = False
   process: List[str] = field(default_factory=lambda: list())

@dataclass
class MetaData:
   devices: Dict[str, DeviceInfo]
   attacks: Dict[str, AttackInfo]
   features: Dict[str, FeatureInfo]

   def __init__(self):
      super().__init__()
      with open(Paths.config.joinpath('metadata.yaml')) as f:
         data = yaml.safe_load(f)
      self.devices = {k: DeviceInfo(**v) for k, v in data['devices'].items()}
      self.attacks = {k: AttackInfo(**v) for k, v in data['attacks'].items()}
      self.features = {k: FeatureInfo(**v) for k, v in data['features'].items()}
