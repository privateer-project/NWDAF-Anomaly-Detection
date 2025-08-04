from typing import Dict, List
from dataclasses import dataclass


@dataclass
class DeviceInfo:
    """
    Device characterization for network traffic analysis.

    This dataclass encapsulates device properties within the, including identification parameters,
    network configuration and attack participation metadata. The characterization supports both
    malicious and benign device categories while tracking participation
    patterns across different attack scenarios.

    Attributes:
        imeisv (str): International Mobile Equipment Identity Software Version
                     serving as unique device identifier within the network
        ip (str): Assigned IP address for network communication and traffic
                 correlation analysis
        type (str): Hardware Device Type, e.g., 'raspberry', 'waveshare_5g_cpe_box',
        malicious (bool): Binary indicator specifying whether device exhibits
                         malicious behavior during attack scenarios
        in_attacks (List[int]): Attack scenario identifiers where device participates, enabling temporal correlation
                                of device behavior with attack patterns
    """
    imeisv: str
    ip: str
    type: str
    malicious: bool
    in_attacks: List[int]


@dataclass
class AttackInfo:
    """
    Temporal attack scenario specification for dataset annotation.

    This dataclass defines precise temporal boundaries for attack events
    within the network traffic dataset, enabling accurate labeling of
    malicious versus benign traffic periods. The temporal precision
    supports fine-grained anomaly detection training and evaluation
    across different attack methodologies.

    Attributes:
        start (str): Attack initiation timestamp in ISO format, marking
                    the precise beginning of malicious activity within
                    the network traffic timeline
        stop (str): Attack termination timestamp in ISO format, defining
                   the conclusion of malicious activity and return to
                   benign traffic patterns
    """
    start: str
    stop: str


@dataclass
class FeatureInfo:
    """
    Feature processing configuration for machine learning pipeline integration.

    This dataclass specifies comprehensive processing instructions for individual
    network traffic features, controlling data type handling and inclusion criteria.
    The configuration enables flexible feature engineering while maintaining
    consistency across training and inference phases of the anomaly detection pipeline.

    Attributes:
        dtype (str): Data type specification for proper pandas DataFrame
                    construction and type validation during loading
        drop (bool): Feature exclusion flag controlling whether feature
                    should be removed during preprocessing pipeline
        is_input (bool): Model input designation specifying whether feature
                        serves as input to machine learning models
    """

    dtype: str = 'str'
    drop: bool = False
    is_input: bool = False


class MetadataRegistry:
    """
    Centralized metadata repository for PRIVATEER dataset characterization.

    This registry maintains comprehensive metadata specifications for the
    network traffic dataset, including device configurations and attack
    scenario definitions. The registry providing type-safe access to dataset
    metadata throughout the analytics pipeline.

    The registry encompasses three primary metadata categories: device
    specifications defining network participants and their attack involvement,
    attack timeline information enabling precise temporal labeling, and controlling data preparation workflows.
    This centralized approach ensures consistency across distributed training scenarios.
    """

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
        'bearer_0_dl_total_bytes': FeatureInfo(dtype='float', drop=True, is_input=False),
        'bearer_0_ul_total_bytes': FeatureInfo(dtype='float', drop=True, is_input=False),
        'bearer_1_dl_total_bytes': FeatureInfo(dtype='float', drop=True, is_input=False),
        'bearer_1_ul_total_bytes': FeatureInfo(dtype='float', drop=True, is_input=False),
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
    """
    Configuration interface for metadata access and feature extraction.

    This class provides a structured interface for accessing dataset metadata
    through Python objects, maintaining type safety. The configuration
    supports comprehensive feature categorization, device characterization,
    and attack scenario specifications essential for privacy-preserving
    anomaly detection in 5G networks.

    The implementation centralizes metadata management to ensure consistency
    across federated learning scenarios while providing convenient accessor
    methods for different metadata categories. This approach facilitates
    reproducible experiments and standardized data processing workflows
    throughout the PRIVATEER analytics framework.

    Attributes:
        devices (Dict[str, DeviceInfo]): Device configuration registry mapping
                                       device identifiers to comprehensive
                                       device specifications
        attacks (Dict[int, AttackInfo]): Attack scenario definitions providing
                                       temporal boundaries for malicious
                                       activity periods
        features (Dict[str, FeatureInfo]): Feature processing specifications
                                         controlling data preparation and
                                         model input selection
    """



    def __init__(self):
        """Initialize metadata configuration with defined specifications"""
        self.devices = MetadataRegistry.DEVICES
        self.attacks = MetadataRegistry.ATTACKS
        self.features = MetadataRegistry.FEATURES

    def get_input_features(self) -> List[str]:
        """Get list of input features"""
        return [feat for feat, info in self.features.items() if info.is_input]

    def get_input_size(self):
        """Get input features size"""
        return len(self.get_input_features())

    def get_drop_features(self) -> List[str]:
        """Get list of features designated for exclusion"""
        return [feat for feat, info in self.features.items() if info.drop]

    def get_features_dtypes(self) -> Dict[str, str]:
        """Get data type specifications for proper DataFrame construction"""
        return {feat: info.dtype for feat, info in self.features.items()}