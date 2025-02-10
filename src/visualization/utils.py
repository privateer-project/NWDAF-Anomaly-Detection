from typing import Dict, List, Tuple
from pandas import DataFrame, Series
from config import MetaData, FeatureInfo


def create_device_color_map(metadata: MetaData) -> Dict[str, str]:
    """Create color mapping for normal and malicious devices."""
    normal_users = [dev.imeisv for dev in metadata.devices.values() if not dev.malicious]
    malicious_users = [dev.imeisv for dev in metadata.devices.values() if dev.malicious]

    color_map = {imeisv: f'rgba(0, {50 + i * 40}, 0, 0.8)'
                 for i, imeisv in enumerate(normal_users)}
    color_map.update({imeisv: f'rgba({50 + i * 40}, 0, 0, 0.8)'
                      for i, imeisv in enumerate(malicious_users)})
    return color_map


def get_feature_data(df: DataFrame, feature_name: str,
                     metadata: MetaData) -> Series:
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


def create_subplot_config(feature_name: str) -> Dict:
    """Create subplot configuration."""
    return {
        'rows': 3,
        'cols': 1,
        'vertical_spacing': 0.2,
        'subplot_titles': [
            f'{feature_name} Over Time',
            f'Distribution of {feature_name}',
            f'Distribution of {feature_name} during attack'
        ]
    }

def create_button_menu(normal_traces: List[int], malicious_traces: List[int],
                      feature_name: str) -> List[Dict]:
    """Create button menu configuration."""
    return [{
        'label': 'Show/Hide Normal Users',
        'method': 'update',
        'args': [{'visible': [True] * len(normal_traces + malicious_traces)},
                {'title': f'Analysis of {feature_name}'}],
        'args2': [{'visible': [i not in normal_traces for i in range(len(normal_traces + malicious_traces))]},
                 {'title': f'Analysis of {feature_name}'}]
    }, {
        'label': 'Show/Hide Malicious Users',
        'method': 'update',
        'args': [{'visible': [True] * len(normal_traces + malicious_traces)},
                {'title': f'Analysis of {feature_name}'}],
        'args2': [{'visible': [i not in malicious_traces for i in range(len(normal_traces + malicious_traces))]},
                 {'title': f'Analysis of {feature_name}'}]
    }]

def create_layout_config(feature_name: str, buttons: List[Dict]) -> Dict:
    """Create layout configuration."""
    return {
        'title_text': f'Analysis of {feature_name}',
        'height': 1000,
        'showlegend': True,
        'barmode': 'overlay',
        'bargap': 0.1,
        'margin': dict(b=80, t=30, pad=10),
        'updatemenus': [{
            'type': 'buttons',
            'direction': 'right',
            'x': 1.1,
            'y': 0.6,
            'xanchor': 'left',
            'yanchor': 'bottom',
            'showactive': True,
            'buttons': buttons,
            'pad': {'r': 10, 't': 10},
            'bgcolor': 'lightgray',
            'font': dict(size=12)
        }],
        'legend': dict(
            groupclick='togglegroup',
            x=1.05,
            y=0.99,
            xanchor='left',
            yanchor='top'
        )
    }