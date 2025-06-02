import time
import threading
import queue

from datetime import datetime

import dash
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import torch

from dash import dcc, html, Input, Output, State

from privateer_ad.etl import DataProcessor
from privateer_ad.config import ModelConfig, MetadataConfig, MLFlowConfig, DataConfig, TrainingConfig
from privateer_ad.utils import load_champion_model

class PrivateerAnomalyDetector:
    """Simplified anomaly detector using only PRIVATEER components"""

    def __init__(self):
        self.data_config = DataConfig()
        self.mlflow_config = MLFlowConfig()
        self.metadata = MetadataConfig()
        self.training_config = TrainingConfig()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize DataProcessor
        self.data_config.num_workers = 0
        self.data_config.pin_memory = False
        self.data_config.batch_size = 1
        self.data_processor = DataProcessor(self.data_config)
        self.test_dataloader = self.data_processor.get_dataloader('test')

        self.threshold = 0.061  # Default threshold
        self.model = load_champion_model(self.mlflow_config.server_address, model_name='TransformerAD_DP')
        self.model.to(self.device)
        self.model.eval()
        # Calculate reconstruction error (L1 loss)
        self.loss_fn = getattr(torch.nn, self.training_config.loss_fn)(reduction='none')

        # Get input features from metadata
        self.input_features = self.metadata.get_input_features()

    def detect_anomaly(self, input_batch):
        """
        Detect anomaly for a single batch from the dataloader

        Args:
            input_batch: Batch from DataProcessor.get_dataloader()

        Returns:
            tuple: (is_anomaly, reconstruction_error, true_label)
        """
        try:
            # Extract input tensor and true label
            input_tensor = input_batch[0]['encoder_cont'].to(self.device)
            true_label = input_batch[1][0].item() if len(input_batch) > 1 else None

            # Run inference
            with torch.no_grad():
                output = self.model(input_tensor)
                reconstruction_error = self.loss_fn(input_tensor, output).mean(dim=(1, 2)).item()

                # Determine if anomaly
                is_anomaly = reconstruction_error > self.threshold

                return is_anomaly, reconstruction_error, true_label

        except Exception as e:
            print(f"❌ Error in anomaly detection: {e}")
            return False, 0.0, None

    def update_threshold(self, new_threshold):
        """Update the anomaly threshold"""
        self.threshold = new_threshold
        print(f"🎯 Threshold updated to: {new_threshold:.6f}")


class NetworkTrafficSimulator:
    """Simplified traffic simulator using DataProcessor dataloader"""

    def __init__(self, detector):
        self.detector = detector
        self.dataloader_iterator = None
        self.data_queue = queue.Queue()
        self.running = False
        self.thread = None
        self.current_sample_index = 0

        # Create iterator from dataloader
        self._reset_iterator()

    def _reset_iterator(self):
        """Reset the dataloader iterator to start from beginning"""
        self.current_sample_index = 0
        print("🔄 Dataloader iterator reset to beginning")

    def get_next_sample(self):
        """Get the next sample from the dataloader"""
        try:
            sample = next(self.detector.test_dataloader)
            self.current_sample_index += 1
            return sample
        except StopIteration:
            # End of dataset, restart from beginning
            print("📄 End of dataset reached, restarting from beginning")
            self._reset_iterator()
            return self.get_next_sample()

    def start_simulation(self, interval=0.1):
        """Start the data simulation"""
        self.running = True
        self.thread = threading.Thread(target=self._simulation_loop, args=(interval,))
        self.thread.daemon = True
        self.thread.start()
        print(f"▶️ Simulation started with {interval}s interval")

    def stop_simulation(self):
        """Stop the data simulation"""
        self.running = False
        if self.thread:
            self.thread.join()
        print("⏸️ Simulation stopped")

    def _simulation_loop(self, interval):
        """Main simulation loop"""
        while self.running:
            try:
                # Get next sample from dataloader
                sample = self.get_next_sample()

                # Detect anomaly
                is_anomaly, score, true_label = self.detector.detect_anomaly(sample)

                # Create result dictionary
                result = {
                    'timestamp': datetime.now(),
                    'sample_index': self.current_sample_index,
                    'is_anomaly': is_anomaly,
                    'reconstruction_error': score,
                    'true_label': true_label,
                    'input_tensor': sample[0]['encoder_cont'].cpu().numpy(),
                    'feature_values': {}
                }

                # Extract feature values for display
                input_flat = sample[0]['encoder_cont'].squeeze().cpu().numpy()
                if len(input_flat.shape) == 2:  # [seq_len, features]
                    # Take the last timestep for current values
                    current_features = input_flat[-1]
                    for i, feature_name in enumerate(self.detector.input_features):
                        if i < len(current_features):
                            result['feature_values'][feature_name] = float(current_features[i])

                # Put result in queue
                self.data_queue.put(result)

                time.sleep(interval)

            except Exception as e:
                print(f"❌ Error in simulation loop: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(interval)

    def get_latest_data(self):
        """Get all available data from queue"""
        data = []
        while not self.data_queue.empty():
            try:
                data.append(self.data_queue.get_nowait())
            except queue.Empty:
                break
        return data


# Initialize components
print("🔄 Initializing PRIVATEER components...")
detector = PrivateerAnomalyDetector()
simulator = NetworkTrafficSimulator(detector)

# Storage for real-time data
realtime_data = {
    'timestamp': [],
    'sample_index': [],
    'reconstruction_error': [],
    'is_anomaly': [],
    'true_label': [],
    'feature_values': {}  # Will be populated with actual feature names
}

# Initialize feature storage
for feature in detector.input_features:
    realtime_data['feature_values'][feature] = []

max_points = 200  # Keep last 200 points for display

# Create Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "PRIVATEER - Network Anomaly Detection Demo"

# App layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("🛡️ PRIVATEER Network Anomaly Detection Demo",
                    className="text-center mb-4"),
            html.P("Using TransformerAD Model with DataProcessor Test Dataset",
                   className="text-center text-muted"),
            html.Hr(),
        ])
    ]),

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("🎮 Simulation Controls", className="card-title"),
                    dbc.ButtonGroup([
                        dbc.Button("▶️ Start Simulation", id="start-btn", color="success", className="me-2"),
                        dbc.Button("⏸️ Stop Simulation", id="stop-btn", color="danger", className="me-2"),
                        dbc.Button("🔄 Reset", id="reset-btn", color="warning")
                    ]),
                    html.Hr(),
                    html.Div([
                        html.Label("🎯 Anomaly Threshold:", className="form-label"),
                        dcc.Slider(
                            id='threshold-slider',
                            min=0.01,
                            max=0.1,
                            step=0.001,
                            value=detector.threshold,
                            marks={
                                0.01: '0.01',
                                0.02: '0.02',
                                0.03: '0.03',
                                0.04: '0.04',
                                0.05: '0.05',
                                0.06: '0.06',
                                0.07: '0.07',
                                0.08: '0.08',
                                0.09: '0.09',
                            },
                            tooltip={"placement": "bottom", "always_visible": True}
                        )
                    ], className="mb-3"),
                    html.Div(id="status-indicator", className="mt-3")
                ])
            ])
        ], width=12)
    ], className="mb-4"),

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("📊 Feature Values (Latest Sample)", className="card-title"),
                    dcc.Graph(id="feature-display", style={'height': '400px'})
                ])
            ])
        ], width=6),

        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("🚨 Anomaly Detection Results", className="card-title"),
                    dcc.Graph(id="anomaly-detection", style={'height': '400px'})
                ])
            ])
        ], width=6)
    ], className="mb-4"),

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("📈 Detection Statistics", className="card-title"),
                    html.Div(id="stats-display")
                ])
            ])
        ], width=12)
    ]),

    # Interval component for real-time updates
    dcc.Interval(
        id='interval-component',
        interval=1000,  # Update every second
        n_intervals=0,
        disabled=True
    ),

    # Store simulation state
    dcc.Store(id='simulation-state', data={'running': False})

], fluid=True)


# Callbacks
@app.callback(
    Output('simulation-state', 'data', allow_duplicate=True),
    Input('threshold-slider', 'value'),
    prevent_initial_call=True
)
def update_threshold(threshold):
    detector.update_threshold(threshold)
    return {'running': True}  # Just return current state


@app.callback(
    [Output('simulation-state', 'data'),
     Output('interval-component', 'disabled'),
     Output('status-indicator', 'children')],
    [Input('start-btn', 'n_clicks'),
     Input('stop-btn', 'n_clicks'),
     Input('reset-btn', 'n_clicks')],
    [State('simulation-state', 'data')]
)
def control_simulation(start_clicks, stop_clicks, reset_clicks, state):
    ctx = dash.callback_context

    if not ctx.triggered:
        return state, True, create_status_badge("Stopped", "danger")

    button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if button_id == 'start-btn' and start_clicks:
        simulator.start_simulation(interval=0.5)  # 2 updates per second
        return {'running': True}, False, create_status_badge("Running", "success")

    elif button_id == 'stop-btn' and stop_clicks:
        simulator.stop_simulation()
        return {'running': False}, True, create_status_badge("Stopped", "danger")

    elif button_id == 'reset-btn' and reset_clicks:
        simulator.stop_simulation()
        # Clear realtime data
        for key in realtime_data:
            if key != 'feature_values':
                realtime_data[key].clear()
            else:
                for feature in realtime_data['feature_values']:
                    realtime_data['feature_values'][feature].clear()
        # Reset dataloader iterator
        simulator._reset_iterator()
        return {'running': False}, True, create_status_badge("Reset", "warning")

    return state, True, create_status_badge("Stopped", "danger")


def create_status_badge(text, color):
    return dbc.Row([
        dbc.Col([
            dbc.Badge(f"Status: {text}", color=color, className="fs-6 me-2"),
            dbc.Badge("Model: TransformerAD + DataProcessor", color="info", className="fs-6"),
            dbc.Badge(f"Sample: {simulator.current_sample_index}", color="secondary", className="fs-6 ms-2")
        ])
    ])


@app.callback(
    [Output('feature-display', 'figure'),
     Output('anomaly-detection', 'figure'),
     Output('stats-display', 'children')],
    [Input('interval-component', 'n_intervals')],
    [State('simulation-state', 'data')]
)
def update_graphs(n, state):
    if not state.get('running', False):
        return create_empty_figure("Simulation Stopped"), create_empty_figure("Simulation Stopped"), html.P(
            "Start simulation to see statistics")

    # Get new data
    new_data = simulator.get_latest_data()

    # Add new data to realtime storage
    for data_point in new_data:
        realtime_data['timestamp'].append(data_point['timestamp'])
        realtime_data['sample_index'].append(data_point['sample_index'])
        realtime_data['reconstruction_error'].append(data_point['reconstruction_error'])
        realtime_data['is_anomaly'].append(data_point['is_anomaly'])
        realtime_data['true_label'].append(data_point['true_label'])

        # Add feature values
        for feature, value in data_point['feature_values'].items():
            if feature in realtime_data['feature_values']:
                realtime_data['feature_values'][feature].append(value)

    # Limit data size
    if len(realtime_data['timestamp']) > max_points:
        for key in realtime_data:
            if key != 'feature_values':
                realtime_data[key] = realtime_data[key][-max_points:]
            else:
                for feature in realtime_data['feature_values']:
                    realtime_data['feature_values'][feature] = realtime_data['feature_values'][feature][-max_points:]

    # Create figures
    feature_fig = create_feature_figure()
    anomaly_fig = create_anomaly_figure()
    stats = create_statistics()

    return feature_fig, anomaly_fig, stats


def create_empty_figure(title):
    fig = go.Figure()
    fig.add_annotation(
        text=title,
        xref="paper", yref="paper",
        x=0.5, y=0.5, xanchor='center', yanchor='middle',
        showarrow=False, font=dict(size=20)
    )
    fig.update_layout(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        plot_bgcolor='white'
    )
    return fig


def create_feature_figure():
    if not realtime_data['timestamp']:
        return create_empty_figure("No Data Available")

    fig = go.Figure()

    # Plot the most important features
    colors = ['blue', 'green', 'orange', 'red', 'purple', 'brown', 'pink', 'gray']
    feature_names = list(realtime_data['feature_values'].keys())[:6]  # Show top 6 features

    for i, feature in enumerate(feature_names):
        if feature in realtime_data['feature_values'] and realtime_data['feature_values'][feature]:
            fig.add_trace(go.Scatter(
                x=realtime_data['timestamp'],
                y=realtime_data['feature_values'][feature],
                mode='lines',
                name=feature,
                line=dict(color=colors[i % len(colors)])
            ))

    # Highlight anomalies
    anomaly_times = [realtime_data['timestamp'][i] for i, anomaly in enumerate(realtime_data['is_anomaly']) if anomaly]

    if anomaly_times and feature_names:
        # Use first feature for anomaly markers
        first_feature = feature_names[0]
        if first_feature in realtime_data['feature_values']:
            anomaly_values = [realtime_data['feature_values'][first_feature][i]
                              for i, anomaly in enumerate(realtime_data['is_anomaly']) if anomaly]

            fig.add_trace(go.Scatter(
                x=anomaly_times,
                y=anomaly_values,
                mode='markers',
                name='Detected Anomalies',
                marker=dict(color='red', size=10, symbol='x'),
                showlegend=True
            ))

    fig.update_layout(
        title="Network Feature Values (Preprocessed by DataProcessor)",
        xaxis_title="Time",
        yaxis_title="Feature Value (Normalized)",
        hovermode='x unified',
        showlegend=True
    )

    return fig


def create_anomaly_figure():
    if not realtime_data['timestamp']:
        return create_empty_figure("No Data Available")

    fig = go.Figure()

    # Add reconstruction errors
    colors = ['red' if anomaly else 'blue' for anomaly in realtime_data['is_anomaly']]

    fig.add_trace(go.Scatter(
        x=realtime_data['timestamp'],
        y=realtime_data['reconstruction_error'],
        mode='markers+lines',
        name='Reconstruction Error',
        marker=dict(color=colors, size=6),
        line=dict(color='gray', width=1)
    ))

    # Add threshold line
    fig.add_hline(
        y=detector.threshold,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Threshold ({detector.threshold:.6f})"
    )

    # Add ground truth markers
    true_anomaly_times = [realtime_data['timestamp'][i] for i, label in enumerate(realtime_data['true_label']) if
                          label == 1]
    true_anomaly_scores = [realtime_data['reconstruction_error'][i] for i, label in
                           enumerate(realtime_data['true_label']) if label == 1]

    if true_anomaly_times:
        fig.add_trace(go.Scatter(
            x=true_anomaly_times,
            y=true_anomaly_scores,
            mode='markers',
            name='True Attacks',
            marker=dict(color='orange', size=8, symbol='diamond'),
            showlegend=True
        ))

    fig.update_layout(
        title="TransformerAD Anomaly Detection Results",
        xaxis_title="Time",
        yaxis_title="Reconstruction Error (L1 Loss)",
        hovermode='x unified'
    )

    return fig


def create_statistics():
    if not realtime_data['timestamp']:
        return html.P("No data available")

    total_points = len(realtime_data['timestamp'])
    detected_anomalies = sum(realtime_data['is_anomaly'])
    true_attacks = sum(1 for label in realtime_data['true_label'] if label == 1)

    # True positive rate
    true_positives = sum(1 for i in range(len(realtime_data['is_anomaly']))
                         if realtime_data['is_anomaly'][i] and realtime_data['true_label'][i] == 1)

    true_positive_rate = (true_positives / true_attacks) * 100 if true_attacks > 0 else 0

    # False positive rate
    false_positives = sum(1 for i in range(len(realtime_data['is_anomaly']))
                          if realtime_data['is_anomaly'][i] and realtime_data['true_label'][i] == 0)

    normal_samples = total_points - true_attacks
    false_positive_rate = (false_positives / normal_samples) * 100 if normal_samples > 0 else 0

    return dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("📊 Total Samples"),
                    html.H3(f"{total_points}", className="text-primary")
                ])
            ])
        ], width=2),

        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("🚨 Detected"),
                    html.H3(f"{detected_anomalies}", className="text-danger")
                ])
            ])
        ], width=2),

        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("🎯 True Attacks"),
                    html.H3(f"{true_attacks}", className="text-warning")
                ])
            ])
        ], width=2),

        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("✅ True Positives"),
                    html.H3(f"{true_positives}", className="text-success")
                ])
            ])
        ], width=2),

        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("✅ TPR"),
                    html.H3(f"{true_positive_rate:.1f}%", className="text-success")
                ])
            ])
        ], width=2),

        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("❌ FPR"),
                    html.H3(f"{false_positive_rate:.1f}%", className="text-info")
                ])
            ])
        ], width=2)
    ])


if __name__ == '__main__':
    print("🛡️ PRIVATEER Network Anomaly Detection Demo")
    print("=" * 50)
    print("🤖 Using TransformerAD Model with DataProcessor")
    print(f"📱 Device: {detector.device}")
    print(f"🎯 Initial Threshold: {detector.threshold}")
    print(f"📊 Input Features: {detector.input_features}")
    try:
        model_config = ModelConfig()
        print(f"📏 Sequence Length: {model_config.seq_len}")
    except:
        print("📏 Sequence Length: 12 (default)")
    print(f"📄 Test Dataset: Loaded via DataProcessor.get_dataloader('test')")
    print("=" * 50)
    print("Starting web server...")
    print("Open your browser and go to: http://127.0.0.1:8050")
    print("=" * 50)

    app.run_server(debug=True, host='127.0.0.1', port=8050)