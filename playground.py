#!/usr/bin/env python3
"""
PRIVATEER Network Anomaly Detection Demo with Actual Trained Model
Uses the real TransformerAD model for anomaly detection
"""

import os
import sys
import time
import threading
import queue
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import torch
from pathlib import Path

# Add the privateer_ad package to the path
sys.path.append('.')
sys.path.append('./privateer_ad')

try:
    from privateer_ad.architectures import TransformerAD, TransformerADConfig
    from privateer_ad.etl import DataProcessor
    from privateer_ad.config import get_paths, get_model_config, get_training_config
    from sklearn.metrics import roc_curve
    PRIVATEER_AVAILABLE = True
    print("✅ PRIVATEER modules loaded successfully")
except ImportError as e:
    print(f"❌ Error importing PRIVATEER modules: {e}")
    print("Make sure you're running from the project root directory")
    PRIVATEER_AVAILABLE = False

class PrivateerAnomalyDetector:
    def __init__(self, model_path="experiments/20250313-181907/model.pt"):
        self.model = None
        self.data_processor = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.seq_len = 12  # Default sequence length
        self.threshold = None
        self.history = []
        self.input_features = []

        if PRIVATEER_AVAILABLE:
            self._load_model_and_processor(model_path)
        else:
            print("⚠️ PRIVATEER modules not available, using fallback detector")
            self._init_fallback()

    def _load_model_and_processor(self, model_path):
        """Load the actual PRIVATEER model and DataProcessor"""
        try:
            print(f"🔄 Loading model and data processor...")

            # Initialize DataProcessor first
            try:
                self.data_processor = DataProcessor(partition=False)
                print("✅ DataProcessor initialized successfully")

                # Get input features from metadata
                self.input_features = self.data_processor.input_features
                print(f"📊 Input features: {self.input_features}")

            except Exception as e:
                print(f"⚠️ Error initializing DataProcessor: {e}")
                # Fallback to manual features
                self.input_features = [
                    'dl_bitrate', 'ul_bitrate', 'dl_retx', 'ul_tx',
                    'dl_tx', 'ul_retx', 'bearer_0_dl_total_bytes', 'bearer_0_ul_total_bytes'
                ]

            # Try to get configurations
            try:
                model_config = get_model_config()
                training_config = get_training_config()
                seq_len = model_config.seq_len
                input_size = len(self.input_features)
                print(f"📋 Using config: seq_len={seq_len}, input_size={input_size}")
            except Exception as e:
                # Fallback configurations
                seq_len = 12
                input_size = len(self.input_features)
                print(f"⚠️ Using fallback model configuration: seq_len={seq_len}, input_size={input_size}")

            # Create model configuration
            transformer_config = TransformerADConfig(
                seq_len=seq_len,
                input_size=input_size,
                num_layers=1,
                hidden_dim=32,
                latent_dim=16,
                num_heads=1,
                dropout=0.2
            )

            # Create and load model
            self.model = TransformerAD(transformer_config)

            # Load the state dict
            if os.path.exists(model_path):
                state_dict = torch.load(model_path, map_location=self.device)

                # Handle different state dict formats
                if isinstance(state_dict, dict):
                    # Remove common prefixes from distributed training
                    cleaned_state_dict = {}
                    for key, value in state_dict.items():
                        clean_key = key
                        for prefix in ['_module.', 'module.']:
                            if clean_key.startswith(prefix):
                                clean_key = clean_key[len(prefix):]
                        cleaned_state_dict[clean_key] = value

                    self.model.load_state_dict(cleaned_state_dict)
                else:
                    self.model.load_state_dict(state_dict)

                print("✅ Model loaded successfully")
            else:
                print(f"❌ Model file not found: {model_path}")
                self._init_fallback()
                return

            self.model.to(self.device)
            self.model.eval()
            self.seq_len = seq_len

            # Set a reasonable threshold (will be refined with data)
            self.threshold = 0.01

        except Exception as e:
            print(f"❌ Error loading model: {e}")
            import traceback
            traceback.print_exc()
            self._init_fallback()

    def _init_fallback(self):
        """Initialize fallback detector"""
        self.model = None
        self.data_processor = None
        self.threshold = 2.5
        self.input_features = [
            'dl_bitrate', 'ul_bitrate', 'dl_retx', 'ul_tx',
            'dl_tx', 'ul_retx', 'bearer_0_dl_total_bytes', 'bearer_0_ul_total_bytes'
        ]
        print("⚠️ Using statistical fallback detector")

    def _prepare_input_with_dataprocessor(self, data_sequence):
        """Prepare input data using the actual DataProcessor"""
        try:
            if len(data_sequence) < self.seq_len:
                # Pad with the last available value
                while len(data_sequence) < self.seq_len:
                    data_sequence = [data_sequence[-1] if data_sequence else {}] + data_sequence

            # Take only the last seq_len samples
            data_sequence = data_sequence[-self.seq_len:]

            # Convert to DataFrame for DataProcessor
            df_data = []
            current_time = datetime.now()

            for i, sample in enumerate(data_sequence):
                row = {}

                # Add timestamp
                row['_time'] = current_time - timedelta(seconds=self.seq_len-i-1)

                # Add all available features
                if isinstance(sample, dict):
                    for feature in self.input_features:
                        row[feature] = float(sample.get(feature, 0.0))

                    # Add non-input features that might be needed
                    row['imeisv'] = sample.get('device_id', 'demo_device')
                    row['attack'] = sample.get('attack', 0)
                    row['malicious'] = sample.get('attack', 0)
                    row['cell'] = 'demo_cell'
                else:
                    # Fallback if sample is not a dict
                    for j, feature in enumerate(self.input_features):
                        row[feature] = float(sample[j] if j < len(sample) else 0.0)
                    row['imeisv'] = 'demo_device'
                    row['attack'] = 0
                    row['malicious'] = 0
                    row['cell'] = 'demo_cell'

                df_data.append(row)

            # Create DataFrame
            df = pd.DataFrame(df_data)

            # Use DataProcessor to preprocess
            processed_df = self.data_processor.preprocess_data(df, partition_id=0)

            # Extract the input features in the correct order
            feature_data = processed_df[self.input_features].values

            # Convert to tensor
            tensor = torch.tensor(feature_data, dtype=torch.float32).unsqueeze(0).to(self.device)
            return tensor

        except Exception as e:
            print(f"❌ Error in DataProcessor preparation: {e}")
            import traceback
            traceback.print_exc()
            # Fallback to manual preparation
            return self._prepare_input_manual(data_sequence)

    def _prepare_input_manual(self, data_sequence):
        """Manual input preparation as fallback"""
        if len(data_sequence) < self.seq_len:
            # Pad with the last available value
            while len(data_sequence) < self.seq_len:
                data_sequence = [data_sequence[-1] if data_sequence else [0]*len(self.input_features)] + data_sequence

        # Take only the last seq_len samples
        data_sequence = data_sequence[-self.seq_len:]

        # Extract features
        feature_data = []
        for sample in data_sequence:
            if isinstance(sample, dict):
                features = [float(sample.get(feat, 0.0)) for feat in self.input_features]
            else:
                features = sample[:len(self.input_features)]
            feature_data.append(features)

        # Convert to tensor (no scaling in fallback)
        tensor = torch.tensor(feature_data, dtype=torch.float32).unsqueeze(0).to(self.device)
        return tensor

    def detect(self, data_point):
        """Detect anomaly using the loaded model with DataProcessor"""
        # Add to history
        self.history.append(data_point)

        if len(self.history) < self.seq_len:
            return False, 0.0  # Not enough data yet

        if self.model is None:
            # Fallback to statistical method
            return self._statistical_detect(data_point)

        try:
            # Prepare input using DataProcessor
            if self.data_processor is not None:
                input_tensor = self._prepare_input_with_dataprocessor(self.history)
            else:
                input_tensor = self._prepare_input_manual(self.history)

            # Run inference
            with torch.no_grad():
                output = self.model(input_tensor)

                # Calculate reconstruction error (MSE)
                mse = torch.mean(torch.pow(input_tensor - output, 2)).item()

                # Determine if anomaly
                is_anomaly = mse > self.threshold

                return is_anomaly, mse

        except Exception as e:
            print(f"❌ Error in model inference: {e}")
            import traceback
            traceback.print_exc()
            return self._statistical_detect(data_point)

    def _statistical_detect(self, data_point):
        """Fallback statistical detection"""
        if len(self.history) < 10:
            return False, 0.0

        # Simple combined metric
        if isinstance(data_point, dict):
            combined_metric = (data_point.get('dl_bitrate', 0) +
                             data_point.get('ul_bitrate', 0) +
                             data_point.get('dl_retx', 0) * 100)
        else:
            combined_metric = sum(data_point[:3]) if len(data_point) >= 3 else 0

        recent_values = [self._get_combined_metric(h) for h in self.history[-50:]]
        mean_val = np.mean(recent_values[:-1])
        std_val = np.std(recent_values[:-1])

        if std_val == 0:
            return False, 0.0

        z_score = abs((combined_metric - mean_val) / std_val)
        is_anomaly = z_score > self.threshold

        return is_anomaly, z_score

    def _get_combined_metric(self, data_point):
        """Extract combined metric from data point"""
        if isinstance(data_point, dict):
            return (data_point.get('dl_bitrate', 0) +
                   data_point.get('ul_bitrate', 0) +
                   data_point.get('dl_retx', 0) * 100)
        else:
            return sum(data_point[:3]) if len(data_point) >= 3 else 0

    def update_threshold(self, new_threshold):
        """Update the anomaly threshold"""
        self.threshold = new_threshold
        print(f"🎯 Threshold updated to: {new_threshold:.4f}")

# Data simulator class (same as before but with better feature mapping)
class NetworkTrafficSimulator:
    def __init__(self, csv_path=None):
        self.data_queue = queue.Queue()
        self.running = False
        self.thread = None

        # Load test data if available
        self.test_data = None
        if csv_path and os.path.exists(csv_path):
            try:
                self.test_data = pd.read_csv(csv_path)
                print(f"✅ Loaded test data with {len(self.test_data)} records")

                # Map column names to expected features
                self._map_columns()

            except Exception as e:
                print(f"❌ Error loading test data: {e}")

        # Generate synthetic data if no test data available
        if self.test_data is None:
            self.test_data = self._generate_synthetic_data()

        self.data_index = 0

    def _map_columns(self):
        """Map dataset columns to expected feature names using PRIVATEER metadata"""
        try:
            # Get expected features from DataProcessor if available
            if PRIVATEER_AVAILABLE:
                try:
                    from privateer_ad.config import get_metadata
                    metadata = get_metadata()
                    expected_features = metadata.get_input_features()
                    print(f"📋 Expected input features from metadata: {expected_features}")
                except Exception as e:
                    print(f"⚠️ Could not get metadata features: {e}")
                    expected_features = [
                        'dl_bitrate', 'ul_bitrate', 'dl_retx', 'ul_tx',
                        'dl_tx', 'ul_retx', 'bearer_0_dl_total_bytes', 'bearer_0_ul_total_bytes'
                    ]
            else:
                expected_features = [
                    'dl_bitrate', 'ul_bitrate', 'dl_retx', 'ul_tx',
                    'dl_tx', 'ul_retx', 'bearer_0_dl_total_bytes', 'bearer_0_ul_total_bytes'
                ]
        except:
            expected_features = [
                'dl_bitrate', 'ul_bitrate', 'dl_retx', 'ul_tx',
                'dl_tx', 'ul_retx', 'bearer_0_dl_total_bytes', 'bearer_0_ul_total_bytes'
            ]

        # Common column mappings
        column_mapping = {
            'dl_bitrate': 'dl_bitrate',
            'ul_bitrate': 'ul_bitrate',
            'dl_retx': 'dl_retx',
            'ul_tx': 'ul_tx',
            'dl_tx': 'dl_tx',
            'ul_retx': 'ul_retx',
            'bearer_0_dl_total_bytes': 'bearer_0_dl_total_bytes',
            'bearer_0_ul_total_bytes': 'bearer_0_ul_total_bytes',
            'attack': 'attack',
            'malicious': 'attack',  # Alternative attack column
            'imeisv': 'imeisv',
            'device_id': 'imeisv',
            '_time': '_time',
            'timestamp': '_time'
        }

        # Rename columns if they exist
        for old_name, new_name in column_mapping.items():
            if old_name in self.test_data.columns and new_name != old_name:
                self.test_data = self.test_data.rename(columns={old_name: new_name})

        # Fill missing required columns with realistic defaults
        for feature in expected_features:
            if feature not in self.test_data.columns:
                print(f"⚠️ Missing feature {feature}, generating synthetic data")
                # Generate realistic defaults based on feature name
                if 'dl_bitrate' in feature:
                    self.test_data[feature] = np.random.normal(5000, 1000, len(self.test_data))
                elif 'ul_bitrate' in feature:
                    self.test_data[feature] = np.random.normal(2000, 500, len(self.test_data))
                elif 'retx' in feature:
                    self.test_data[feature] = np.random.poisson(2, len(self.test_data))
                elif 'tx' in feature:
                    self.test_data[feature] = np.random.poisson(30, len(self.test_data))
                elif 'bytes' in feature:
                    if 'dl' in feature:
                        self.test_data[feature] = np.random.normal(50000, 10000, len(self.test_data))
                    else:
                        self.test_data[feature] = np.random.normal(20000, 5000, len(self.test_data))
                else:
                    self.test_data[feature] = np.random.normal(100, 20, len(self.test_data))

        # Ensure required metadata columns exist
        required_columns = ['attack', 'imeisv', '_time']
        for col in required_columns:
            if col not in self.test_data.columns:
                if col == 'attack':
                    self.test_data[col] = 0  # Default to no attack
                elif col == 'imeisv':
                    self.test_data[col] = 'demo_device_001'  # Default device
                elif col == '_time':
                    # Create timestamps if missing
                    self.test_data[col] = pd.date_range(
                        start='2024-01-01',
                        periods=len(self.test_data),
                        freq='1S'
                    )

        # Ensure imeisv is string type
        self.test_data['imeisv'] = self.test_data['imeisv'].astype(str)

        # Add device_id for backward compatibility
        if 'device_id' not in self.test_data.columns:
            self.test_data['device_id'] = self.test_data['imeisv']

        print(f"📊 Final dataset features: {list(self.test_data.columns)}")
        print(f"📊 Dataset shape: {self.test_data.shape}")
        print(f"📊 Attack samples: {self.test_data['attack'].sum()}")

        # Show sample of data
        print("📋 Sample data:")
        print(self.test_data[expected_features + ['attack', 'imeisv']].head(3))

    def _generate_synthetic_data(self):
        """Generate synthetic network traffic data matching the model's expected format"""
        np.random.seed(42)
        n_samples = 1000

        # Generate normal traffic patterns
        base_dl_bitrate = 5000 + np.random.normal(0, 1000, n_samples)
        base_ul_bitrate = 2000 + np.random.normal(0, 500, n_samples)
        base_dl_retx = np.random.poisson(2, n_samples)
        base_ul_retx = np.random.poisson(2, n_samples)
        base_dl_tx = np.random.poisson(30, n_samples)
        base_ul_tx = np.random.poisson(30, n_samples)
        base_dl_bytes = np.random.normal(50000, 10000, n_samples)
        base_ul_bytes = np.random.normal(20000, 5000, n_samples)

        # Add some anomalies (attacks)
        anomaly_indices = np.random.choice(n_samples, size=int(n_samples * 0.05), replace=False)

        dl_bitrate = base_dl_bitrate.copy()
        ul_bitrate = base_ul_bitrate.copy()
        dl_retx = base_dl_retx.copy()
        ul_retx = base_ul_retx.copy()
        dl_tx = base_dl_tx.copy()
        ul_tx = base_ul_tx.copy()
        dl_bytes = base_dl_bytes.copy()
        ul_bytes = base_ul_bytes.copy()

        # Inject anomalies
        for idx in anomaly_indices:
            dl_bitrate[idx] *= np.random.uniform(3, 10)  # High traffic
            ul_bitrate[idx] *= np.random.uniform(5, 15)  # Very high upload
            dl_retx[idx] *= np.random.uniform(4, 8)      # High retransmissions
            ul_retx[idx] *= np.random.uniform(4, 8)      # High retransmissions
            dl_tx[idx] *= np.random.uniform(2, 5)        # High transmissions
            ul_tx[idx] *= np.random.uniform(2, 5)        # High transmissions
            dl_bytes[idx] *= np.random.uniform(3, 8)     # High byte counts
            ul_bytes[idx] *= np.random.uniform(3, 8)     # High byte counts

        data = pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', periods=n_samples, freq='1S'),
            'dl_bitrate': dl_bitrate,
            'ul_bitrate': ul_bitrate,
            'dl_retx': dl_retx,
            'ul_retx': ul_retx,
            'dl_tx': dl_tx,
            'ul_tx': ul_tx,
            'bearer_0_dl_total_bytes': dl_bytes,
            'bearer_0_ul_total_bytes': ul_bytes,
            'device_id': np.random.choice(['Device_001', 'Device_002', 'Device_003'], n_samples),
            'attack': np.isin(range(n_samples), anomaly_indices).astype(int)
        })

        print("✅ Generated synthetic network traffic data")
        return data

    def start_simulation(self, interval=1.0):
        """Start the data simulation"""
        self.running = True
        self.thread = threading.Thread(target=self._simulation_loop, args=(interval,))
        self.thread.daemon = True
        self.thread.start()

    def stop_simulation(self):
        """Stop the data simulation"""
        self.running = False
        if self.thread:
            self.thread.join()

    def _simulation_loop(self, interval):
        """Main simulation loop"""
        while self.running:
            if self.data_index >= len(self.test_data):
                self.data_index = 0  # Loop back to start

            row = self.test_data.iloc[self.data_index].to_dict()
            row['timestamp'] = datetime.now()

            self.data_queue.put(row)
            self.data_index += 1

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
simulator = NetworkTrafficSimulator()
detector = PrivateerAnomalyDetector()

# Storage for real-time data
realtime_data = {
    'timestamp': [],
    'dl_bitrate': [],
    'ul_bitrate': [],
    'dl_retx': [],
    'ul_tx': [],
    'device_id': [],
    'is_anomaly': [],
    'anomaly_score': [],
    'true_attack': []
}

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
                                html.P("Using TransformerAD Model with DataProcessor for Real-time Anomaly Detection",
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
                            min=0.0001,
                            max=1,
                            step=0.0001,
                            value=detector.threshold,
                            marks={0.0001: '0.0001', 0.001: '0.001', 0.01: '0.01', 0.05: '0.05', 1.: '1.0'},
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
                    html.H4("📊 Real-time Network Metrics", className="card-title"),
                    dcc.Graph(id="realtime-metrics", style={'height': '400px'})
                ])
            ])
        ], width=6),

        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("🚨 Anomaly Detection (TransformerAD)", className="card-title"),
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
    Output('detector-state', 'data', allow_duplicate=True),
    Input('threshold-slider', 'value'),
    prevent_initial_call=True
)
def update_threshold(threshold):
    detector.update_threshold(threshold)
    return {'threshold': threshold}

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
            realtime_data[key].clear()
        # Reset detector history
        detector.history.clear()
        return {'running': False}, True, create_status_badge("Reset", "warning")

    return state, True, create_status_badge("Stopped", "danger")

def create_status_badge(text, color):
    model_info = "TransformerAD + DataProcessor" if detector.model is not None and detector.data_processor is not None else \
                 "TransformerAD (Manual)" if detector.model is not None else "Statistical Fallback"
    return dbc.Row([
        dbc.Col([
            dbc.Badge(f"Status: {text}", color=color, className="fs-6 me-2"),
            dbc.Badge(f"Model: {model_info}", color="info", className="fs-6")
        ])
    ])

@app.callback(
    [Output('realtime-metrics', 'figure'),
     Output('anomaly-detection', 'figure'),
     Output('stats-display', 'children')],
    [Input('interval-component', 'n_intervals')],
    [State('simulation-state', 'data')]
)
def update_graphs(n, state):
    if not state.get('running', False):
        return create_empty_figure("Simulation Stopped"), create_empty_figure("Simulation Stopped"), html.P("Start simulation to see statistics")

    # Get new data
    new_data = simulator.get_latest_data()

    # Process new data
    for row in new_data:
        # Apply anomaly detection using the actual model
        is_anomaly, score = detector.detect(row)

        # Add to realtime data
        realtime_data['timestamp'].append(row['timestamp'])
        realtime_data['dl_bitrate'].append(row.get('dl_bitrate', 0))
        realtime_data['ul_bitrate'].append(row.get('ul_bitrate', 0))
        realtime_data['dl_retx'].append(row.get('dl_retx', 0))
        realtime_data['ul_tx'].append(row.get('ul_tx', 0))
        realtime_data['device_id'].append(row.get('device_id', 'Unknown'))
        realtime_data['is_anomaly'].append(is_anomaly)
        realtime_data['anomaly_score'].append(score)
        realtime_data['true_attack'].append(row.get('attack', 0))

    # Limit data size
    if len(realtime_data['timestamp']) > max_points:
        for key in realtime_data:
            realtime_data[key] = realtime_data[key][-max_points:]

    # Create figures
    metrics_fig = create_metrics_figure()
    anomaly_fig = create_anomaly_figure()
    stats = create_statistics()

    return metrics_fig, anomaly_fig, stats

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

def create_metrics_figure():
    if not realtime_data['timestamp']:
        return create_empty_figure("No Data Available")

    fig = go.Figure()

    # Add traces for different metrics
    fig.add_trace(go.Scatter(
        x=realtime_data['timestamp'],
        y=realtime_data['dl_bitrate'],
        mode='lines',
        name='Download Bitrate',
        line=dict(color='blue')
    ))

    fig.add_trace(go.Scatter(
        x=realtime_data['timestamp'],
        y=realtime_data['ul_bitrate'],
        mode='lines',
        name='Upload Bitrate',
        line=dict(color='green')
    ))

    # Highlight anomalies
    anomaly_times = [realtime_data['timestamp'][i] for i, anomaly in enumerate(realtime_data['is_anomaly']) if anomaly]
    anomaly_dl = [realtime_data['dl_bitrate'][i] for i, anomaly in enumerate(realtime_data['is_anomaly']) if anomaly]

    if anomaly_times:
        fig.add_trace(go.Scatter(
            x=anomaly_times,
            y=anomaly_dl,
            mode='markers',
            name='Detected Anomalies',
            marker=dict(color='red', size=10, symbol='x'),
            showlegend=True
        ))

    fig.update_layout(
        title="Network Traffic Metrics (with Anomaly Highlights)",
        xaxis_title="Time",
        yaxis_title="Bitrate (bps)",
        hovermode='x unified',
        showlegend=True
    )

    return fig

def create_anomaly_figure():
    if not realtime_data['timestamp']:
        return create_empty_figure("No Data Available")

    fig = go.Figure()

    # Add anomaly scores
    colors = ['red' if anomaly else 'blue' for anomaly in realtime_data['is_anomaly']]

    fig.add_trace(go.Scatter(
        x=realtime_data['timestamp'],
        y=realtime_data['anomaly_score'],
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
        annotation_text=f"Threshold ({detector.threshold:.4f})"
    )

    # Add ground truth if available
    true_anomaly_times = [realtime_data['timestamp'][i] for i, attack in enumerate(realtime_data['true_attack']) if attack]
    true_anomaly_scores = [realtime_data['anomaly_score'][i] for i, attack in enumerate(realtime_data['true_attack']) if attack]

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
        title="TransformerAD + DataProcessor Anomaly Detection Results",
        xaxis_title="Time",
        yaxis_title="Reconstruction Error (MSE)" if detector.model else "Z-Score",
        hovermode='x unified'
    )

    return fig

def create_statistics():
    if not realtime_data['timestamp']:
        return html.P("No data available")

    total_points = len(realtime_data['timestamp'])
    detected_anomalies = sum(realtime_data['is_anomaly'])
    true_attacks = sum(realtime_data['true_attack'])

    # Calculate metrics
    if detected_anomalies > 0:
        detection_rate = (detected_anomalies / total_points) * 100
    else:
        detection_rate = 0

    # True positive rate
    true_positives = sum(1 for i in range(len(realtime_data['is_anomaly']))
                        if realtime_data['is_anomaly'][i] and realtime_data['true_attack'][i])

    if true_attacks > 0:
        true_positive_rate = (true_positives / true_attacks) * 100
    else:
        true_positive_rate = 0

    # False positive rate
    false_positives = sum(1 for i in range(len(realtime_data['is_anomaly']))
                         if realtime_data['is_anomaly'][i] and not realtime_data['true_attack'][i])

    false_negatives = sum(1 for i in range(len(realtime_data['is_anomaly']))
                         if not realtime_data['is_anomaly'][i] and realtime_data['true_attack'][i])

    if (total_points - true_attacks) > 0:
        false_positive_rate = (false_positives / (total_points - true_attacks)) * 100
    else:
        false_positive_rate = 0

    return dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("📊 Total Points"),
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
                    html.H5("✅ True Positive Rate"),
                    html.H3(f"{true_positive_rate:.1f}%", className="text-success")
                ])
            ])
        ], width=3),

        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("❌ False Positive Rate"),
                    html.H3(f"{false_positive_rate:.1f}%", className="text-info")
                ])
            ])
        ], width=3)
    ])

# Add missing store for detector state
app.layout.children.append(dcc.Store(id='detector-state', data={'threshold': detector.threshold}))

if __name__ == '__main__':
    print("🛡️ PRIVATEER Network Anomaly Detection Demo")
    print("=" * 50)
    print("🤖 Using TransformerAD Model with DataProcessor")
    print(f"📱 Device: {detector.device}")
    print(f"🎯 Initial Threshold: {detector.threshold}")
    if detector.data_processor is not None:
        print("✅ DataProcessor: Active (proper scaling & preprocessing)")
        print(f"📊 Input Features: {detector.input_features}")
    else:
        print("⚠️ DataProcessor: Not available (manual preprocessing)")
    print("=" * 50)
    print("Starting web server...")
    print("Open your browser and go to: http://127.0.0.1:8050")
    print("=" * 50)

    # Try to find test data in common locations
    test_data_paths = [
        'data/processed/test.csv',
        'privateer_ad/data/processed/test.csv',
        '../data/processed/test.csv',
        'test.csv'
    ]

    found_data = False
    for path in test_data_paths:
        if os.path.exists(path):
            print(f"✅ Found test data at: {path}")
            simulator = NetworkTrafficSimulator(path)
            found_data = True
            break

    if not found_data:
        print("ℹ️  Using synthetic data (no test.csv found)")
        simulator = NetworkTrafficSimulator()

    app.run_server(debug=True, host='127.0.0.1', port=8050)