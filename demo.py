"""
PRIVATEER Network Anomaly Detection Demo
"""
import time
import threading
import queue
import logging

from datetime import datetime

import dash
import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objects as go
import torch

from dash import dcc, html, Input, Output, State

from privateer_ad.etl import DataProcessor
from privateer_ad.config import DataConfig, MLFlowConfig, MetadataConfig, TrainingConfig
from privateer_ad.utils import load_champion_model


class PrivateerAnomalyDetector:
    """Anomaly detector"""

    def __init__(self,model_name :str = 'TransformerAD_DP'):
        self.model_name = model_name

        self.data_config = DataConfig()
        self.data_config.num_workers = 0
        self.data_config.pin_memory = False
        self.data_config.batch_size = 1
        self.data_config.prefetch_factor = None
        self.data_config.persistent_workers = False
        self.mlflow_config = MLFlowConfig()
        self.metadata = MetadataConfig()
        self.training_config = TrainingConfig()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize DataProcessor with streaming config
        self.data_processor = DataProcessor(self.data_config)
        self.test_ds = self.data_processor.get_dataset('test', only_benign=False)
        self.test_dl = self.data_processor.get_dataloader('test', only_benign=False, train=True)
        self.threshold = 0.061  # Default threshold

        # Load model
        logging.info(f"Loading {self.model_name} model...")
        self.model, self.threshold, self.loss_fn = load_champion_model(tracking_uri=self.mlflow_config.tracking_uri,
                                                                       model_name=self.model_name)
        self.model.to(self.device)
        self.model.eval()

        # Get input features from metadata
        self.input_features = self.metadata.get_input_features()
        logging.info(f"input features: {self.input_features}")

    def detect_anomaly(self, input_batch):
        """
        Detect anomaly for a single batch from the dataloader

        Args:
            input_batch:

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
            logging.error(f"❌ Error in anomaly detection: {e}")
            return False, 0.0, None

    def update_threshold(self, new_threshold):
        """Update the anomaly threshold"""
        self.threshold = new_threshold
        logging.info(f"🎯 Threshold updated to: {new_threshold:.6f}")


class NetworkTrafficSimulator:
    """Traffic simulator using DataProcessor dataloader"""

    def __init__(self, detector):
        self.detector = detector
        self.data_queue = queue.Queue()
        self.running = False
        self.thread = None
        self.current_sample_index = 0
        self.dataloader_iterator = iter(self.detector.test_dl)

    def reset_iterator(self):
        """Reset the dataloader iterator to start from beginning"""
        self.dataloader_iterator = iter(self.detector.test_dl)
        self.current_sample_index = 0
        logging.info("🔄 Dataloader iterator reset to beginning")

    def get_next_sample(self):
        """Get the next sample from the dataloader"""
        try:
            sample = next(self.dataloader_iterator)
            self.current_sample_index += 1
            return sample
        except StopIteration:
            # End of dataset, restart from beginning
            logging.warning("📄 End of dataset reached, restarting from beginning")
            self.reset_iterator()
            return self.get_next_sample()

    def start_simulation(self, interval=0.1):
        """Start the data simulation"""
        self.running = True
        self.thread = threading.Thread(target=self._simulation_loop, args=(interval,))
        self.thread.daemon = True
        self.thread.start()
        logging.info(f"▶️ Simulation started with {interval}s interval")

    def stop_simulation(self):
        """Stop the data simulation"""
        self.running = False
        if self.thread:
            self.thread.join()
        logging.warning("⏸️ Simulation stopped")

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

                try:
                    # The TimeSeriesDataSet groups by 'imeisv', so we can try to extract it
                    # Check if there's group information in the sample
                    device_id = self.detector.test_ds.transform_values('imeisv',
                                                                       sample[0]["groups"],
                                                                       inverse=True,
                                                                       group_id=True)

                except Exception as e:
                    logging.error(f"Debug: Error extracting device ID: {e}")
                    device_id = f"device_{(self.current_sample_index // 50) % 9}"

                # Create anonymized device ID
                if device_id:
                    result['anonymized_device_id'] = f"anon-{hash(str(device_id)) % 10000}"
                else:
                    result['anonymized_device_id'] = f"anon-{(self.current_sample_index // 50) % 9}"
                # Put result in queue
                self.data_queue.put(result)

                time.sleep(interval)

            except Exception as e:
                logging.error(f"❌ Error in simulation loop: {e}")
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
logging.info("🔄 Initializing PRIVATEER components...")
detector = PrivateerAnomalyDetector()
simulator = NetworkTrafficSimulator(detector)

# Storage for real-time data
realtime_data = {
    'timestamp': [],
    'sample_index': [],
    'reconstruction_error': [],
    'is_anomaly': [],
    'true_label': [],
    'anonymized_device_id': [],
    'feature_values': {}
}

# Initialize feature storage
for feature in detector.input_features:
    realtime_data['feature_values'][feature] = []

max_points = 200  # Keep last 200 points for display
min_threshold = float(np.floor(detector.threshold * .1))
max_threshold = float(np.ceil(detector.threshold * 10.))
step_threshold = 0.0001

# Create Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "PRIVATEER - Network Anomaly Detection Demo"

# App layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("🛡️ PRIVATEER Network Anomaly Detection Demo",
                    className="text-center mb-4"),
            html.P("Privacy-Preserving Anomaly Detection for 6G Networks",
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
                            min=min_threshold,
                            max=max_threshold,
                            step=step_threshold,
                            value=detector.threshold,
                            marks={
                                value: f"{value:.3f}"
                                for value in np.linspace(min_threshold, max_threshold, 10)
                            },                            tooltip={"placement": "bottom", "always_visible": True}
                        )
                    ], className="mb-3"),
                    html.Div([
                        html.Label("🔐 Privacy Protection: ", className="form-label"),
                        dbc.Badge("Anonymization Active", color="success", className="ms-2"),
                        html.Small(" - Device IDs are anonymized", className="text-muted ms-2")
                    ]),
                    html.Div(id="status-indicator", className="mt-3")
                ])
            ])
        ], width=12)
    ], className="mb-4"),

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("📊 Network Feature Values (Privacy-Preserved)", className="card-title"),
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
        ], width=8),

        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("🔒 Anonymized Devices", className="card-title"),
                    html.Div(id="device-list", style={'max-height': '200px', 'overflow-y': 'auto'})
                ])
            ])
        ], width=4)
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
    return dash.no_update

def create_status_badge(text, color):
    return dbc.Row([
        dbc.Col([
            dbc.Badge(f"Status: {text}", color=color, className="fs-6 me-2"),
            dbc.Badge(f"Model: {detector.model_name}", color="info", className="fs-6"),
            dbc.Badge(f"Device: {detector.device}", color="secondary", className="fs-6 ms-2"),
            dbc.Badge(f"Sample: {simulator.current_sample_index}", color="secondary", className="fs-6 ms-2")
        ])
    ])

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
        simulator.start_simulation(interval=0.5)
        return {'running': True}, False, create_status_badge("Running", "success")

    elif button_id == 'stop-btn' and stop_clicks:
        simulator.stop_simulation()
        return {'running': False}, True, create_status_badge("Stopped", "danger")

    elif button_id == 'reset-btn' and reset_clicks:
        simulator.stop_simulation()
        # Clear realtime data
        for key in realtime_data:
            if key not in ['feature_values', 'anonymized_device_id']:
                realtime_data[key].clear()
            elif key == 'feature_values':
                for feature in realtime_data['feature_values']:
                    realtime_data['feature_values'][feature].clear()
            elif key == 'anonymized_device_id':
                realtime_data[key].clear()
        # Reset dataloader iterator
        simulator.reset_iterator()
        return {'running': False}, True, create_status_badge("Reset", "warning")

    return state, True, create_status_badge("Stopped", "danger")


@app.callback(
    [Output('feature-display', 'figure'),
     Output('anomaly-detection', 'figure'),
     Output('stats-display', 'children'),
     Output('device-list', 'children')],
    [Input('interval-component', 'n_intervals')],
    [State('simulation-state', 'data')]
)
def update_graphs(n, state):
    if not state.get('running', False):
        return (create_empty_figure("Simulation Stopped"),
                create_empty_figure("Simulation Stopped"),
                html.P("Start simulation to see statistics"),
                html.P("No devices detected yet"))

    # Get new data
    new_data = simulator.get_latest_data()

    # Add new data to realtime storage
    for data_point in new_data:
        realtime_data['timestamp'].append(data_point['timestamp'])
        realtime_data['sample_index'].append(data_point['sample_index'])
        realtime_data['reconstruction_error'].append(data_point['reconstruction_error'])
        realtime_data['is_anomaly'].append(data_point['is_anomaly'])
        realtime_data['true_label'].append(data_point['true_label'])
        realtime_data['anonymized_device_id'].append(data_point['anonymized_device_id'])

        # Add feature values
        for feature, value in data_point['feature_values'].items():
            if feature in realtime_data['feature_values']:
                realtime_data['feature_values'][feature].append(value)

    # Limit data size
    if len(realtime_data['timestamp']) > max_points:
        for key in realtime_data:
            if key == 'feature_values':
                for feature in realtime_data['feature_values']:
                    realtime_data['feature_values'][feature] = realtime_data['feature_values'][feature][-max_points:]
            else:
                realtime_data[key] = realtime_data[key][-max_points:]

    # Create figures
    feature_fig = create_feature_figure()
    anomaly_fig = create_anomaly_figure()
    stats = create_statistics()
    device_list = create_device_list()

    return feature_fig, anomaly_fig, stats, device_list


# Add this NEW callback just for updating the sample counter
@app.callback(
    Output('status-indicator', 'children', allow_duplicate=True),
    [Input('interval-component', 'n_intervals')],
    [State('simulation-state', 'data')],
    prevent_initial_call=True
)
def update_sample_counter(n, state):
    if state.get('running', False):
        return create_status_badge("Running", "success")
    return dash.no_update


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
                name=feature.replace('_', ' ').title(),
                line=dict(color=colors[i % len(colors)])
            ))

    # Highlight anomalies
    anomaly_times = [realtime_data['timestamp'][i] for i, anomaly in enumerate(realtime_data['is_anomaly']) if anomaly]

    if anomaly_times and feature_names:
        # Use first feature for anomaly markers
        first_feature = feature_names[0]
        if first_feature in realtime_data['feature_values']:
            anomaly_values = [realtime_data['feature_values'][first_feature][i]
                              for i, anomaly in enumerate(realtime_data['is_anomaly'])
                              if anomaly and i < len(realtime_data['feature_values'][first_feature])]

            if anomaly_values:
                fig.add_trace(go.Scatter(
                    x=anomaly_times[:len(anomaly_values)],
                    y=anomaly_values,
                    mode='markers',
                    name='Detected Anomalies',
                    marker=dict(color='red', size=10, symbol='x'),
                    showlegend=True
                ))

    fig.update_layout(
        title="Network Feature Values (Normalized & Anonymized)",
        xaxis_title="Time",
        yaxis_title="Feature Value",
        hovermode='x unified',
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5)
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
        title="TransformerAD Anomaly Detection (with Differential Privacy)",
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
        ], width=3),

        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("🚨 Detected"),
                    html.H3(f"{detected_anomalies}", className="text-danger")
                ])
            ])
        ], width=3),

        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("✅ TPR"),
                    html.H3(f"{true_positive_rate:.1f}%", className="text-success")
                ])
            ])
        ], width=3),

        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("❌ FPR"),
                    html.H3(f"{false_positive_rate:.1f}%", className="text-info")
                ])
            ])
        ], width=3)
    ])


def create_device_list():
    if not realtime_data['anonymized_device_id']:
        return html.P("No devices detected yet", className="text-muted")

    # Get unique anonymized device IDs and their anomaly counts
    device_counts = {}
    for i, device_id in enumerate(realtime_data['anonymized_device_id']):
        if device_id not in device_counts:
            device_counts[device_id] = {'total': 0, 'anomalies': 0}
        device_counts[device_id]['total'] += 1
        if realtime_data['is_anomaly'][i]:
            device_counts[device_id]['anomalies'] += 1

    # Create list items
    device_items = []
    for device_id, counts in sorted(device_counts.items())[-10:]:  # Show last 10 devices
        anomaly_rate = (counts['anomalies'] / counts['total']) * 100 if counts['total'] > 0 else 0
        color = "danger" if anomaly_rate > 50 else "warning" if anomaly_rate > 20 else "success"

        device_items.append(
            dbc.ListGroupItem([
                html.Div([
                    html.Span(f"Device: {device_id}", className="fw-bold"),
                    dbc.Badge(f"{anomaly_rate:.0f}%", color=color, className="float-end")
                ]),
                html.Small(f"Samples: {counts['total']}, Anomalies: {counts['anomalies']}",
                           className="text-muted")
            ])
        )

    return dbc.ListGroup(device_items, flush=True)


if __name__ == '__main__':
    logging.info("🛡️ PRIVATEER Network Anomaly Detection Demo")
    logging.info("=" * 50)
    logging.info("🤖 Using TransformerAD Model with Differential Privacy")
    logging.info(f"📱 Device: {detector.device}")
    logging.info(f"🎯 Initial Threshold: {detector.threshold}")
    logging.info(f"📊 Input Features: {detector.input_features}")
    logging.info(f"📄 Dataset: Loaded via DataProcessor.get_dataloader('test')")
    logging.info("🔐 Privacy Protection: Anonymization Active")
    logging.info("=" * 50)
    logging.info("Starting web server...")
    logging.info("Open your browser and go to: http://127.0.0.1:8050")
    logging.info("=" * 50)

    app.run(host='127.0.0.1', port=8050, debug=True)
