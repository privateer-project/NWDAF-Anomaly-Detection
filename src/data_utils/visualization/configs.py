from typing import List, Dict
import plotly.graph_objects as go

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
        'label': "Show/Hide Normal Users",
        'method': "update",
        'args': [{"visible": [True] * len(normal_traces + malicious_traces)},
                {"title": f"Analysis of {feature_name}"}],
        'args2': [{"visible": [i not in normal_traces for i in range(len(normal_traces + malicious_traces))]},
                 {"title": f"Analysis of {feature_name}"}]
    }, {
        'label': "Show/Hide Malicious Users",
        'method': "update",
        'args': [{"visible": [True] * len(normal_traces + malicious_traces)},
                {"title": f"Analysis of {feature_name}"}],
        'args2': [{"visible": [i not in malicious_traces for i in range(len(normal_traces + malicious_traces))]},
                 {"title": f"Analysis of {feature_name}"}]
    }]

def create_layout_config(feature_name: str, buttons: List[Dict]) -> Dict:
    """Create layout configuration."""
    return {
        'title_text': f"Analysis of {feature_name}",
        'height': 1000,
        'showlegend': True,
        'barmode': 'overlay',
        'bargap': 0.1,
        'margin': dict(b=80, t=30, pad=10),
        'updatemenus': [{
            'type': "buttons",
            'direction': "right",
            'x': 1.1,
            'y': 0.6,
            'xanchor': "left",
            'yanchor': "bottom",
            'showactive': True,
            'buttons': buttons,
            'pad': {"r": 10, "t": 10},
            'bgcolor': "lightgray",
            'font': dict(size=12)
        }],
        'legend': dict(
            groupclick="togglegroup",
            x=1.05,
            y=0.99,
            xanchor="left",
            yanchor="top"
        )
    }