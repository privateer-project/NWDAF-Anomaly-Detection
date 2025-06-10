"""
Metadata configuration as Python objects
This replaces the metadata.yaml file to avoid packaging/distribution issues
"""

from typing import Dict, List
from dataclasses import dataclass


@dataclass
class DeviceInfo:
    """Device information"""
    imeisv: str
    ip: str
    type: str
    malicious: bool
    in_attacks: List[int]


@dataclass
class AttackInfo:
    """Attack information"""
    start: str
    stop: str


@dataclass
class FeatureInfo:
    """Feature configuration"""
    dtype: str = 'str'
    drop: bool = False
    is_input: bool = False
    process: List[str] = None

    def __post_init__(self):
        if self.process is None:
            self.process = []


class MetadataRegistry:
    """Central registry for all metadata - replaces metadata.yaml"""

    # Device configurations
    DEVICES = {
        '1': DeviceInfo(
            imeisv='8642840401612300',
            ip='10.20.10.2',
            type='raspberry',
            malicious=True,
            in_attacks=[1, 2, 3, 4, 5]
        ),
        '2': DeviceInfo(
            imeisv='8642840401624200',
            ip='10.20.10.4',
            type='raspberry',
            malicious=True,
            in_attacks=[1, 2, 3, 4, 5]
        ),
        '3': DeviceInfo(
            imeisv='8642840401594200',
            ip='10.20.10.6',
            type='waveshare_5g_cpe_box',
            malicious=True,
            in_attacks=[5]
        ),
        '4': DeviceInfo(
            imeisv='8677660403123800',
            ip='10.20.10.8',
            type='waveshare_industrial_5g_router',
            malicious=True,
            in_attacks=[5]
        ),
        '5': DeviceInfo(
            imeisv='3557821101183501',
            ip='10.20.10.10',
            type='dwr_2101_5g',
            malicious=True,
            in_attacks=[5]
        ),
        '6': DeviceInfo(
            imeisv='8628490433231158',
            ip='10.20.10.12',
            type='huawei_p40',
            malicious=False,
            in_attacks=[1, 2, 3, 4, 5]
        ),
        '7': DeviceInfo(
            imeisv='8609960480859058',
            ip='10.20.10.16',
            type='huawei_p40',
            malicious=False,
            in_attacks=[1, 2, 3, 4, 5]
        ),
        '8': DeviceInfo(
            imeisv='8609960480666910',
            ip='10.20.10.18',
            type='huawei_p40',
            malicious=False,
            in_attacks=[1, 2, 3, 4, 5]
        ),
        '9': DeviceInfo(
            imeisv='8609960468879057',
            ip='10.20.10.20',
            type='huawei_p40',
            malicious=False,
            in_attacks=[1, 2, 3, 4, 5]
        )
    }

    # Attack configurations
    ATTACKS = {
        1: AttackInfo(start='2024-08-18 07:00:00.000', stop='2024-08-18 08:00:00.000'),
        2: AttackInfo(start='2024-08-19 07:00:00.000', stop='2024-08-19 09:41:00.000'),
        3: AttackInfo(start='2024-08-19 17:00:00.000', stop='2024-08-19 18:00:00.000'),
        4: AttackInfo(start='2024-08-21 12:00:00.000', stop='2024-08-21 13:00:00.000'),
        5: AttackInfo(start='2024-08-21 17:00:00.000', stop='2024-08-21 18:00:00.000')
    }

    # Feature configurations
    FEATURES = {
        '_time': FeatureInfo(dtype='string', drop=False, is_input=False),
        'bearer_0_dl_total_bytes': FeatureInfo(dtype='float', drop=True, is_input=False, process=['delta']),
        'bearer_0_ul_total_bytes': FeatureInfo(dtype='float', drop=True, is_input=False, process=['delta']),
        'bearer_1_dl_total_bytes': FeatureInfo(dtype='float', drop=True, is_input=False, process=['delta']),
        'bearer_1_ul_total_bytes': FeatureInfo(dtype='float', drop=True, is_input=False, process=['delta']),
        'dl_bitrate': FeatureInfo(dtype='float', drop=False, is_input=True),
        'dl_err': FeatureInfo(dtype='float', drop=True, is_input=False),
        'dl_mcs': FeatureInfo(dtype='float', drop=False, is_input=False),
        'dl_retx': FeatureInfo(dtype='float', drop=False, is_input=True),
        'dl_tx': FeatureInfo(dtype='float', drop=False, is_input=True),
        'ul_bitrate': FeatureInfo(dtype='float', drop=False, is_input=True),
        'ul_err': FeatureInfo(dtype='float', drop=True, is_input=False),
        'ul_mcs': FeatureInfo(dtype='float', drop=False, is_input=True),
        'ul_path_loss': FeatureInfo(dtype='float', drop=False, is_input=False),
        'ul_phr': FeatureInfo(dtype='float', drop=False, is_input=False),
        'ul_retx': FeatureInfo(dtype='float', drop=False, is_input=True),
        'ul_tx': FeatureInfo(dtype='float', drop=False, is_input=True),
        'cqi': FeatureInfo(dtype='float', drop=True, is_input=False),
        'epre': FeatureInfo(dtype='float', drop=True, is_input=False),
        'initial_ta': FeatureInfo(dtype='float', drop=True, is_input=False),
        'p_ue': FeatureInfo(dtype='float', drop=False, is_input=False),
        'pusch_snr': FeatureInfo(dtype='float', drop=False, is_input=False),
        'turbo_decoder_avg': FeatureInfo(dtype='float', drop=False, is_input=True),
        'attack': FeatureInfo(dtype='int', drop=False, is_input=False),
        'malicious': FeatureInfo(dtype='int', drop=False, is_input=False),
        'attack_number': FeatureInfo(dtype='int', drop=False, is_input=False),
        'imeisv': FeatureInfo(dtype='string', drop=False, is_input=False),
        '5g_tmsi': FeatureInfo(dtype='string', drop=True, is_input=False),
        'amf_ue_id': FeatureInfo(dtype='string', drop=True, is_input=False),
        'bearer_0_apn': FeatureInfo(dtype='string', drop=True, is_input=False),
        'bearer_0_ip': FeatureInfo(dtype='string', drop=True, is_input=False),
        'bearer_0_ipv6': FeatureInfo(dtype='string', drop=True, is_input=False),
        'bearer_0_pdu_session_id': FeatureInfo(dtype='string', drop=True, is_input=False),
        'bearer_0_qos_flow_id': FeatureInfo(dtype='string', drop=True, is_input=False),
        'bearer_0_sst': FeatureInfo(dtype='string', drop=True, is_input=False),
        'bearer_1_apn': FeatureInfo(dtype='string', drop=True, is_input=False),
        'bearer_1_ip': FeatureInfo(dtype='string', drop=True, is_input=False),
        'bearer_1_ipv6': FeatureInfo(dtype='string', drop=True, is_input=False),
        'bearer_1_pdu_session_id': FeatureInfo(dtype='string', drop=True, is_input=False),
        'bearer_1_qos_flow_id': FeatureInfo(dtype='string', drop=True, is_input=False),
        'bearer_1_sst': FeatureInfo(dtype='string', drop=True, is_input=False),
        'ran_id': FeatureInfo(dtype='string', drop=True, is_input=False),
        'ran_plmn': FeatureInfo(dtype='string', drop=True, is_input=False),
        'ran_ue_id': FeatureInfo(dtype='string', drop=True, is_input=False),
        'registered': FeatureInfo(dtype='bool', drop=True, is_input=False),
        'rnti': FeatureInfo(dtype='string', drop=True, is_input=False),
        't3512': FeatureInfo(dtype='int', drop=True, is_input=False),
        'tac': FeatureInfo(dtype='string', drop=True, is_input=False),
        'tac_plmn': FeatureInfo(dtype='string', drop=True, is_input=False),
        'ue_aggregate_max_bitrate_dl': FeatureInfo(dtype='float', drop=True, is_input=False),
        'ue_aggregate_max_bitrate_ul': FeatureInfo(dtype='float', drop=True, is_input=False),
        'cell': FeatureInfo(dtype='string', drop=False, is_input=False),
        'ul_n_layer': FeatureInfo(drop=True, is_input=False),
        'ul_rank': FeatureInfo(dtype='float', drop=True, is_input=False),
        'ri': FeatureInfo(dtype='float', drop=True, is_input=False),
        'turbo_decoder_max': FeatureInfo(dtype='float', drop=True, is_input=False),
        'turbo_decoder_min': FeatureInfo(dtype='float', drop=True, is_input=False),
        'cell_id': FeatureInfo(dtype='string', drop=True, is_input=False)
    }


class MetadataConfig:
    """Configuration class that uses Python objects instead of YAML files"""

    def __init__(self):
        """Initialize with predefined metadata"""
        self.devices = MetadataRegistry.DEVICES
        self.attacks = MetadataRegistry.ATTACKS
        self.features = MetadataRegistry.FEATURES

    def get_input_features(self) -> List[str]:
        """Get list of input features"""
        return [feat for feat, info in self.features.items() if info.is_input]

    def get_drop_features(self) -> List[str]:
        """Get list of features to drop"""
        return [feat for feat, info in self.features.items() if info.drop]

    def get_features_dtypes(self) -> Dict[str, str]:
        """Get feature data types mapping"""
        return {feat: info.dtype for feat, info in self.features.items()}