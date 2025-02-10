from typing import Dict, List, Tuple
import pandas as pd
from src.config import MetaData, FeatureInfo


def create_device_color_map(metadata: MetaData) -> Dict[str, str]:
    """Create color mapping for normal and malicious devices."""
    normal_users = [dev.imeisv for dev in metadata.devices.values() if not dev.malicious]
    malicious_users = [dev.imeisv for dev in metadata.devices.values() if dev.malicious]

    color_map = {imeisv: f'rgba(0, {50 + i * 40}, 0, 0.8)'
                 for i, imeisv in enumerate(normal_users)}
    color_map.update({imeisv: f'rgba({50 + i * 40}, 0, 0, 0.8)'
                      for i, imeisv in enumerate(malicious_users)})
    return color_map


def get_feature_data(df: pd.DataFrame, feature_name: str,
                     metadata: MetaData) -> pd.Series:
    """Process feature data based on metadata configuration."""
    feature_data = df[feature_name]
    if feature_data.dtype == bool:
        feature_data = feature_data.astype(int)

    feature_info = metadata.features.get(feature_name, FeatureInfo())
    if 'delta' in feature_info.process:
        feature_data = feature_data.diff().fillna(0)

    return feature_data


def split_traces(fig) -> Tuple[List[int], List[int]]:
    """Split traces into normal and malicious groups."""
    normal_traces = [i for i, trace in enumerate(fig.data) if 'Normal' in trace.name]
    malicious_traces = [i for i, trace in enumerate(fig.data) if 'Malicious' in trace.name]
    return normal_traces, malicious_traces