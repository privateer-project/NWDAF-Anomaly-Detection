import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.config import Paths, MetaData
from .utils import create_device_color_map, get_feature_data, split_traces
from .configs import (
    create_subplot_config,
    create_button_menu,
    create_layout_config
)


class FeatureAnalyzer:
    """Analyzes and visualizes features from a dataset."""

    SKIP_FEATURES = ['_time', 'imeisv', 'instance_id', 'attack']

    def __init__(self, metadata: MetaData, paths: Paths):
        self.metadata = metadata
        self.paths = paths
        self.color_map = create_device_color_map(metadata)

    def analyze_features(self, df: pd.DataFrame, name: str):
        """Analyze and create visualizations for all features."""
        df = self._preprocess_dataframe(df)
        analysis_dir = self.paths.analysis.joinpath(name)
        analysis_dir.mkdir(parents=True, exist_ok=True)

        for feature_name in df.columns:
            if feature_name in self.SKIP_FEATURES:
                continue

            html_file = analysis_dir.joinpath(f'{feature_name}_analysis.html')
            if html_file.exists():
                print(f'File {html_file} already exists, skipping.')
                continue

            fig = self._create_feature_plot(df, feature_name)
            try:
                fig.write_html(str(html_file))
            except Exception as e:
                print(f'Error writing {html_file}: {e}')

    @staticmethod
    def _preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the input DataFrame."""
        df = df.copy()
        df['_time'] = pd.to_datetime(df['_time'])
        df['imeisv'] = df['imeisv'].astype(str)
        return df.sort_values('_time')

    def _create_feature_plot(self, df: pd.DataFrame, feature_name: str) -> go.Figure:
        """Create plot for a single feature."""
        subplot_config = create_subplot_config(feature_name)
        fig = make_subplots(**subplot_config)

        for imeisv in df['imeisv'].unique():
            self._add_device_traces(fig, df, imeisv, feature_name)

        self._add_attack_periods(fig, df)
        self._update_figure_layout(fig, feature_name)
        return fig

    def _add_device_traces(self, fig: go.Figure, df: pd.DataFrame,
                           imeisv: str, feature_name: str):
        """Add traces for a single device."""
        normal_users = [dev.imeisv for dev in self.metadata.devices.values() if not dev.malicious]

        device_df = df[df['imeisv'] == imeisv].sort_values('_time')
        feature_data = get_feature_data(device_df, feature_name, self.metadata)

        line_color = self.color_map[imeisv]
        role = 'Normal' if imeisv in normal_users else 'Malicious'
        legend_name = f'{role} - IMEISV: {imeisv}'

        # Time series plot
        fig.add_trace(
            go.Scatter(
                x=device_df['_time'],
                y=feature_data,
                name=legend_name,
                mode='lines',
                line=dict(color=line_color),
                opacity=0.5,
                legendgroup=legend_name,
                showlegend=True
            ),
            row=1, col=1
        )

        # Distribution plots
        if pd.api.types.is_numeric_dtype(feature_data):
            self._add_numeric_distributions(fig, feature_data, device_df,
                                            legend_name, line_color)
        else:
            self._add_categorical_distributions(fig, feature_data, device_df,
                                                legend_name, line_color)

    @staticmethod
    def _add_numeric_distributions(fig: go.Figure, feature_data: pd.Series,
                                   device_df: pd.DataFrame, legend_name: str,
                                   line_color: str):
        """Add distribution plots for numeric features."""
        # Overall distribution
        fig.add_trace(
            go.Histogram(
                x=feature_data.clip(upper=feature_data.quantile(0.99)),
                name=legend_name,
                nbinsx=100,
                opacity=0.7,
                histnorm='percent',
                marker_color=line_color,
                legendgroup=legend_name,
                showlegend=False
            ),
            row=2, col=1
        )

        # Distribution during attack
        attack_data = feature_data[device_df['attack'] == 1]
        fig.add_trace(
            go.Histogram(
                x=attack_data.clip(upper=attack_data.quantile(0.99)),
                name=legend_name,
                nbinsx=100,
                opacity=0.7,
                histnorm='percent',
                marker_color=line_color,
                legendgroup=legend_name,
                showlegend=False
            ),
            row=3, col=1
        )

    @staticmethod
    def _add_categorical_distributions(fig: go.Figure, feature_data: pd.Series,
                                       device_df: pd.DataFrame, legend_name: str,
                                       line_color: str):
        """Add distribution plots for categorical features."""
        # Overall distribution
        value_counts = feature_data[device_df['attack'] == 0].value_counts()
        fig.add_trace(
            go.Bar(
                x=list(value_counts.index),
                y=value_counts.values,
                name=legend_name,
                marker_color=line_color,
                legendgroup=legend_name,
                showlegend=False
            ),
            row=2, col=1
        )

        # Distribution during attack
        attack_counts = feature_data[device_df['attack'] == 1].value_counts()
        fig.add_trace(
            go.Bar(
                x=list(attack_counts.index),
                y=attack_counts.values,
                name=legend_name,
                marker_color=line_color,
                legendgroup=legend_name,
                showlegend=False
            ),
            row=3, col=1
        )

    @staticmethod
    def _add_attack_periods(fig: go.Figure, df: pd.DataFrame):
        """Add attack period highlighting to the plot."""
        attack_data = df[df['attack'] == 1]
        if not attack_data.empty:
            for _, period in attack_data.groupby('instance_id'):
                if _ != 0:
                    fig.add_vrect(
                        x0=period['_time'].iloc[0],
                        x1=period['_time'].iloc[-1],
                        fillcolor="red",
                        opacity=0.2,
                        layer="below",
                        line_width=0,
                        row=1, col=1
                    )

    @staticmethod
    def _update_figure_layout(fig: go.Figure, feature_name: str):
        """Update the figure layout with proper configuration."""
        normal_traces, malicious_traces = split_traces(fig)
        buttons = create_button_menu(normal_traces, malicious_traces, feature_name)
        layout = create_layout_config(feature_name, buttons)

        fig.update_layout(**layout)
        fig.update_xaxes(title_text="Time", row=1, col=1)
        [fig.update_xaxes(title_text=feature_name, row=row, col=1) for row in [2, 3]]
        fig.update_yaxes(title_text=feature_name, row=1, col=1)
        [fig.update_yaxes(title_text="Percent", row=row, col=1) for row in [2, 3]]
