"""
PRIVATEER Anomaly Detector Service with Web UI
Combines Kafka consumption with real-time Dash visualization
"""
import os
import sys
import json
import time
import logging
import threading
import queue
from datetime import datetime, timedelta
from collections import defaultdict

import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
import requests
from dash import dcc, html, Input, Output, State
from kafka import KafkaConsumer, KafkaProducer

from privateer_ad.config import MLFlowConfig, MetadataConfig, ModelConfig, DataConfig
from privateer_ad.utils import load_champion_model
from privateer_ad.etl import DataProcessor

# Force stdout to be unbuffered for Docker logging
sys.stdout.reconfigure(line_buffering=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


class AggregationManager:
    """Manages alert aggregation to prevent spam during attacks"""

    def __init__(self, window_seconds=60, threshold_count=6):
        self.window_seconds = window_seconds
        self.threshold_count = threshold_count
        self.device_alerts = defaultdict(list)
        self.last_sent = defaultdict(lambda: datetime.min)

    def should_send_alert(self, device_id, timestamp):
        """Determine if an alert should be sent based on aggregation rules"""
        now = datetime.fromisoformat(timestamp)

        # Clean old alerts
        cutoff = now - timedelta(seconds=self.window_seconds)
        self.device_alerts[device_id] = [
            ts for ts in self.device_alerts[device_id] if ts > cutoff
        ]

        # Add current alert
        self.device_alerts[device_id].append(now)

        # Check if we should send
        alert_count = len(self.device_alerts[device_id])
        time_since_last = (now - self.last_sent[device_id]).total_seconds()

        # Send if: first alert, threshold reached, or cooldown expired
        if (alert_count == 1 or
            alert_count >= self.threshold_count or
            time_since_last >= self.window_seconds):
            self.last_sent[device_id] = now
            return True, alert_count

        return False, alert_count


class DetectorWithUI:
    def __init__(self):
        """Initialize the detector with UI components"""
        logging.info("🔧 Initializing PRIVATEER Anomaly Detector with UI...")

        # Configuration
        self.mlflow_config = MLFlowConfig()
        self.metadata = MetadataConfig()
        self.model_config = ModelConfig()
        self.data_config = DataConfig()

        # Override for streaming
        self.data_config.batch_size = 1
        self.data_config.num_workers = 0
        self.data_config.pin_memory = False
        self.data_config.prefetch_factor = None
        self.data_config.persistent_workers = False

        # Model setup
        self.feature_list = self.metadata.get_input_features()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.threshold = None
        self.loss_fn = None
        self.model_name = self.model_config.model_type
        self.seq_len = self.data_config.seq_len  # 12 timesteps

        # Sliding window buffer for each device (stores DataFrames)
        self.device_windows = defaultdict(list)

        # Data preprocessing - Initialize DataProcessor
        self.dp = DataProcessor(self.data_config)

        # Kafka setup
        self.bootstrap_servers = os.environ.get('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092')
        self.input_topic = os.environ.get('INPUT_TOPIC', 'anonymized-data')
        self.consumer = None
        self.running = False
        self.consumer_thread = None

        # Data queue for UI
        self.data_queue = queue.Queue(maxsize=1000)

        # Aggregation manager
        self.aggregation_manager = AggregationManager(
            window_seconds=int(os.environ.get('AGGREGATION_WINDOW', '60')),
            threshold_count=int(os.environ.get('AGGREGATION_THRESHOLD', '5'))
        )

        # Endpoint configuration (placeholder)
        self.endpoint_url = os.environ.get('ALERT_ENDPOINT_URL', None)
        self.endpoint_enabled = os.environ.get('ENABLE_ENDPOINT', 'false').lower() == 'true'

        # Storage for real-time data
        self.realtime_data = {
            'timestamp': [],
            'sample_index': [],
            'reconstruction_error': [],
            'is_anomaly': [],
            'true_label': [],
            'anonymized_device_id': [],
            'feature_values': {}
        }

        # Initialize feature storage
        for feature in self.feature_list:
            self.realtime_data['feature_values'][feature] = []

        self.max_points = 200
        self.current_sample_index = 0

        # Statistics
        self.stats = {
            'total': 0,
            'anomalies': 0,
            'true_positives': 0,
            'false_positives': 0,
            'true_negatives': 0,
            'false_negatives': 0,
            'alerts_sent': 0,
            'alerts_aggregated': 0
        }

    def load_model(self):
        """Load model with retry logic"""
        max_retries = 5
        for attempt in range(max_retries):
            try:
                logging.info(f"🔄 Loading model... attempt {attempt + 1}/{max_retries}")
                self.model, self.threshold, self.loss_fn = load_champion_model(
                    tracking_uri=self.mlflow_config.tracking_uri,
                    model_name=self.model_name
                )
                self.model.to(self.device)
                self.model.eval()
                logging.info(f"✅ Model loaded successfully!")
                logging.info(f"📏 Threshold: {self.threshold}")
                return True
            except Exception as e:
                logging.error(f"❌ Failed to load model (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(10)
                else:
                    return False

    def create_sample_dataframe(self, data, timestamp):
        """Create a properly formatted DataFrame from incoming data"""
        # Create single-row DataFrame with all required columns
        sample_data = {
            '_time': timestamp,
            'imeisv': data.get('imeisv', 'unknown'),
            'cell': data.get('cell', '1'),
            'attack': int(data.get('attack', 0)),
            'malicious': int(data.get('malicious', 0)),
            'attack_number': int(data.get('attack_number', 0)),
        }

        # Add all input features
        for feature in self.feature_list:
            value = data.get(feature, 0.0)
            if value is None or value == '' or (isinstance(value, str) and value.lower() == 'nan'):
                value = 0.0
            try:
                sample_data[feature] = float(value)
            except (ValueError, TypeError):
                sample_data[feature] = 0.0

        # Create DataFrame
        df = pd.DataFrame([sample_data])

        # Ensure proper data types
        df['_time'] = pd.to_datetime(df['_time'])
        df['attack'] = df['attack'].astype(int)
        df['malicious'] = df['malicious'].astype(int)
        df['attack_number'] = df['attack_number'].astype(int)

        for feature in self.feature_list:
            df[feature] = pd.to_numeric(df[feature], errors='coerce').fillna(0.0)

        return df

    def detect_anomaly(self, data):
        """Detect anomaly using temporally-ordered sliding window with proper DataProcessor preprocessing"""
        try:
            device_id = data.get('imeisv', 'unknown')

            # Parse timestamp
            timestamp_str = data.get('_time') or data.get('original_time')
            if timestamp_str:
                try:
                    if isinstance(timestamp_str, str):
                        timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    else:
                        timestamp = timestamp_str
                except:
                    timestamp = datetime.now()
            else:
                timestamp = datetime.now()

            # Create properly formatted DataFrame for this sample
            sample_df = self.create_sample_dataframe(data, timestamp)

            # Initialize device window if needed
            if device_id not in self.device_windows:
                self.device_windows[device_id] = []

            # Add to device's window
            self.device_windows[device_id].append(sample_df)

            # Sort by timestamp to maintain temporal order
            self.device_windows[device_id].sort(key=lambda df: df['_time'].iloc[0])

            # Keep only the last seq_len samples
            if len(self.device_windows[device_id]) > self.seq_len:
                self.device_windows[device_id] = self.device_windows[device_id][-self.seq_len:]

            # Only detect if we have enough samples
            if len(self.device_windows[device_id]) < self.seq_len:
                logging.debug(
                    f"Device {device_id}: {len(self.device_windows[device_id])}/{self.seq_len} samples collected")
                return None

            # Combine window DataFrames into a single DataFrame
            window_df = pd.concat(self.device_windows[device_id], ignore_index=True)

            # Apply DataProcessor preprocessing (this handles scaling properly)
            try:
                # Clean and preprocess the data using DataProcessor
                processed_window_df = self.dp.clean_data(window_df.copy())
                processed_window_df = self.dp.apply_scale(processed_window_df)

                # Extract the scaled feature values in the correct order
                scaled_features = processed_window_df[self.feature_list].values

            except Exception as e:
                logging.error(f"❌ Error in preprocessing: {e}")
                return None

            # Create sequence tensor [batch_size=1, seq_len, features]
            sequence_tensor = torch.tensor(scaled_features, dtype=torch.float32).unsqueeze(0).to(self.device)

            # Run inference on the full sequence
            with torch.no_grad():
                output = self.model(sequence_tensor)
                # Calculate reconstruction error - mean over sequence and features dimensions
                reconstruction_error = self.loss_fn(sequence_tensor, output).mean(dim=(1, 2)).item()

            # Check if anomaly
            is_anomaly = reconstruction_error > self.threshold

            # Extract feature values from the latest (unscaled) sample for display
            latest_sample_df = self.device_windows[device_id][-1]
            feature_values = {feature: float(latest_sample_df[feature].iloc[0]) for feature in self.feature_list}

            return {
                'is_anomaly': bool(is_anomaly),
                'reconstruction_error': float(reconstruction_error),
                'threshold': float(self.threshold),
                'timestamp': timestamp,
                'device_id': device_id,
                'anonymized_device_id': f"anon-{hash(str(device_id)) % 10000}",
                'true_label': int(latest_sample_df['attack'].iloc[0]),
                'feature_values': feature_values,
                'sample_index': self.current_sample_index,
                'window_size': len(self.device_windows[device_id]),
                'temporal_span': (window_df['_time'].iloc[-1] - window_df['_time'].iloc[0]).total_seconds()
            }

        except Exception as e:
            logging.error(f"❌ Error in anomaly detection: {e}")
            import traceback
            traceback.print_exc()
            return None

    def send_to_endpoint(self, alert_data):
        """Placeholder for sending alerts to external endpoint"""
        if not self.endpoint_enabled or not self.endpoint_url:
            return

        try:
            # TODO: Implement actual endpoint communication
            # For now, just log that we would send
            logging.info(f"📤 Would send alert to endpoint: {self.endpoint_url}")
            logging.info(f"   Alert data: {json.dumps(alert_data, default=str)[:100]}...")
            self.stats['alerts_sent'] += 1

            # Example implementation:
            # response = requests.post(
            #     self.endpoint_url,
            #     json=alert_data,
            #     timeout=5
            # )
            # if response.status_code == 200:
            #     self.stats['alerts_sent'] += 1

        except Exception as e:
            logging.error(f"❌ Failed to send alert to endpoint: {e}")

    def wait_for_kafka(self, max_retries=30):
        """Wait for Kafka to be available"""
        for i in range(max_retries):
            try:
                # Test connection
                test_producer = KafkaProducer(
                    bootstrap_servers=self.bootstrap_servers,
                    request_timeout_ms=5000,
                    retries=1
                )
                test_producer.close()
                logging.info(f"✅ Connected to Kafka at {self.bootstrap_servers}")
                return True
            except Exception as e:
                logging.info(f"⏳ Waiting for Kafka... attempt {i+1}/{max_retries}")
                time.sleep(2)

        logging.error(f"❌ Failed to connect to Kafka after {max_retries} attempts")
        return False

    def kafka_consumer_loop(self):
        """Main Kafka consumer loop"""
        try:
            # Wait for Kafka to be available
            if not self.wait_for_kafka():
                logging.error("❌ Cannot start consumer - Kafka unavailable")
                return

            self.consumer = KafkaConsumer(
                self.input_topic,
                bootstrap_servers=self.bootstrap_servers,
                value_deserializer=lambda v: json.loads(v.decode('utf-8')),
                auto_offset_reset='latest',
                group_id='detector-ui-group',
                enable_auto_commit=True
            )

            logging.info("✅ Kafka consumer started")

            while self.running:
                # Poll for messages
                message_batch = self.consumer.poll(timeout_ms=100)

                if not message_batch:
                    continue

                # Process messages
                for topic_partition, messages in message_batch.items():
                    for message in messages:
                        try:
                            data = message.value
                            result = self.detect_anomaly(data)

                            if result:
                                self.current_sample_index += 1
                                self.stats['total'] += 1

                                # Update statistics
                                if result['is_anomaly']:
                                    self.stats['anomalies'] += 1

                                    # Check aggregation
                                    should_send, alert_count = self.aggregation_manager.should_send_alert(
                                        result['device_id'],
                                        result['timestamp'].isoformat()
                                    )

                                    if should_send:
                                        # Prepare alert data
                                        alert_data = {
                                            'alert_id': f"alert-{self.stats['anomalies']}",
                                            'device_id': result['device_id'],
                                            'score': result['reconstruction_error'],
                                            'threshold': result['threshold'],
                                            'timestamp': result['timestamp'].isoformat(),
                                            'alert_count': alert_count,
                                            'window_seconds': self.aggregation_manager.window_seconds
                                        }

                                        # Send to endpoint
                                        self.stats['alerts_sent'] += 1
                                        self.send_to_endpoint(alert_data)
                                    else:
                                        self.stats['alerts_aggregated'] += 1

                                    # Update accuracy stats
                                    if result['true_label'] == 1:
                                        self.stats['true_positives'] += 1
                                    elif result['true_label'] == 0:
                                        self.stats['false_positives'] += 1
                                else:
                                    if result['true_label'] == 0:
                                        self.stats['true_negatives'] += 1
                                    elif result['true_label'] == 1:
                                        self.stats['false_negatives'] += 1

                                # Log window status for first few detections per device
                                if self.stats['total'] < 20 or self.stats['total'] % 100 == 0:
                                    logging.info(f"📊 Detection #{self.stats['total']} - Device: {result['anonymized_device_id']}, "
                                               f"Window: {result['window_size']}/{self.seq_len}, "
                                               f"Anomaly: {result['is_anomaly']}")

                                # Queue for UI (non-blocking)
                                try:
                                    self.data_queue.put_nowait(result)
                                except queue.Full:
                                    # Remove oldest and retry
                                    self.data_queue.get()
                                    self.data_queue.put_nowait(result)

                        except Exception as e:
                            logging.error(f"❌ Error processing message: {e}")

        except Exception as e:
            logging.error(f"❌ Kafka consumer error: {e}")
        finally:
            if self.consumer:
                self.consumer.close()

    def start_kafka_consumer(self):
        """Start Kafka consumer in background thread"""
        self.running = True
        self.consumer_thread = threading.Thread(target=self.kafka_consumer_loop)
        self.consumer_thread.daemon = True
        self.consumer_thread.start()
        logging.info("▶️ Kafka consumer started")

    def stop_kafka_consumer(self):
        """Stop Kafka consumer"""
        self.running = False
        if self.consumer_thread:
            self.consumer_thread.join(timeout=5)
        logging.info("⏸️ Kafka consumer stopped")

    def get_latest_data(self):
        """Get all available data from queue"""
        data = []
        while not self.data_queue.empty():
            try:
                data.append(self.data_queue.get_nowait())
            except queue.Empty:
                break
        return data


# Global detector instance
detector = None


def wait_for_services():
    """Wait for required services to be available"""
    # Wait for MLflow
    mlflow_uri = os.environ.get('PRIVATEER_MLFLOW_TRACKING_URI', 'http://localhost:5001')
    mlflow_health_url = mlflow_uri.replace(':5001', ':5050') + '/health'

    max_retries = 20
    for i in range(max_retries):
        try:
            response = requests.get(mlflow_health_url, timeout=5)
            if response.status_code == 200:
                logging.info(f"✅ MLflow is available")
                break
        except:
            logging.info(f"⏳ Waiting for MLflow... {i+1}/{max_retries}")
            time.sleep(5)


# Create Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "PRIVATEER - Network Anomaly Detection"

# App layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("🛡️ PRIVATEER Network Anomaly Detection", className="text-center mb-4"),
            html.P("Real-time Privacy-Preserving Anomaly Detection for 6G Networks",
                   className="text-center text-muted"),
            html.Hr(),
        ])
    ]),

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("🎮 Detection Controls", className="card-title"),
                    dbc.ButtonGroup([
                        dbc.Button("▶️ Start Detection", id="start-btn", color="success", className="me-2"),
                        dbc.Button("⏸️ Stop Detection", id="stop-btn", color="danger")
                    ], className="mb-3"),
                    html.Div(id="status-indicator", className="mb-3"),
                    html.Hr(),
                    html.Div([
                        html.Label("🎯 Anomaly Threshold:", className="form-label"),
                        dcc.Slider(
                            id='threshold-slider',
                            min=0.001,
                            max=0.5,
                            step=0.001,
                            value=0.061,
                            marks={v: f"{v:.3f}" for v in np.linspace(0.001, 0.5, 10)},
                            tooltip={"placement": "bottom", "always_visible": True}
                        )
                    ], className="mb-3"),
                    html.Div([
                        html.Label("🔐 Privacy Protection: ", className="form-label"),
                        dbc.Badge("Kafka Pipeline Active", color="success", className="ms-2"),
                        html.Small(" - Real-time processing", className="text-muted ms-2")
                    ])
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
        n_intervals=0
    ),

    # Store for detector state
    dcc.Store(id='detector-state', data={'running': False})
], fluid=True)


def create_status_badge():
    global detector
    if detector and detector.running:
        return dbc.Row([
            dbc.Col([
                dbc.Badge("Status: Running", color="success", className="fs-6 me-2"),
                dbc.Badge(f"Model: {detector.model_name}", color="info", className="fs-6"),
                dbc.Badge(f"Device: {detector.device}", color="secondary", className="fs-6 ms-2"),
                dbc.Badge(f"Sample: {detector.current_sample_index}", color="secondary", className="fs-6 ms-2")
            ])
        ])
    else:
        return dbc.Badge("Status: Stopped", color="danger")


@app.callback(
    [Output('detector-state', 'data'),
     Output('start-btn', 'disabled'),
     Output('stop-btn', 'disabled')],
    [Input('start-btn', 'n_clicks'),
     Input('stop-btn', 'n_clicks')],
    [State('detector-state', 'data')]
)
def control_detector(start_clicks, stop_clicks, state):
    global detector

    ctx = dash.callback_context
    if not ctx.triggered:
        return state, False, True  # Initial state: start enabled, stop disabled

    button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if button_id == 'start-btn' and start_clicks:
        if detector and not detector.running:
            detector.start_kafka_consumer()
            logging.info("▶️ Detector started via UI")
        return {'running': True}, True, False  # Start disabled, stop enabled

    elif button_id == 'stop-btn' and stop_clicks:
        if detector and detector.running:
            detector.stop_kafka_consumer()
            logging.info("⏸️ Detector stopped via UI")
        return {'running': False}, False, True  # Start enabled, stop disabled

    return state, state.get('running', False), not state.get('running', False)


@app.callback(
    Output('threshold-slider', 'value'),
    Input('interval-component', 'n_intervals')
)
def update_threshold_from_model(n):
    global detector
    if detector and detector.threshold:
        return detector.threshold
    return 0.061


@app.callback(
    [Output('feature-display', 'figure'),
     Output('anomaly-detection', 'figure'),
     Output('stats-display', 'children'),
     Output('device-list', 'children'),
     Output('status-indicator', 'children')],
    [Input('interval-component', 'n_intervals')],
    [State('detector-state', 'data')]
)
def update_graphs(n, state):
    global detector

    if not detector:
        empty_fig = go.Figure()
        empty_fig.add_annotation(
            text="Initializing...",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False, font=dict(size=20)
        )
        return empty_fig, empty_fig, html.P("Initializing..."), html.P("No devices yet"), create_status_badge()

    # Check if detector is running
    if not state.get('running', False):
        return (create_empty_figure("Detection Stopped - Click Start to begin"),
                create_empty_figure("Detection Stopped - Click Start to begin"),
                create_statistics(),
                create_device_list(),
                create_status_badge())

    # Get new data
    new_data = detector.get_latest_data()

    # Add new data to realtime storage
    for data_point in new_data:
        detector.realtime_data['timestamp'].append(data_point['timestamp'])
        detector.realtime_data['sample_index'].append(data_point['sample_index'])
        detector.realtime_data['reconstruction_error'].append(data_point['reconstruction_error'])
        detector.realtime_data['is_anomaly'].append(data_point['is_anomaly'])
        detector.realtime_data['true_label'].append(data_point['true_label'])
        detector.realtime_data['anonymized_device_id'].append(data_point['anonymized_device_id'])

        # Add feature values
        for feature, value in data_point['feature_values'].items():
            if feature in detector.realtime_data['feature_values']:
                detector.realtime_data['feature_values'][feature].append(value)

    # Limit data size
    if len(detector.realtime_data['timestamp']) > detector.max_points:
        for key in detector.realtime_data:
            if key == 'feature_values':
                for feature in detector.realtime_data['feature_values']:
                    detector.realtime_data['feature_values'][feature] = \
                        detector.realtime_data['feature_values'][feature][-detector.max_points:]
            else:
                detector.realtime_data[key] = detector.realtime_data[key][-detector.max_points:]

    # Create figures
    feature_fig = create_feature_figure()
    anomaly_fig = create_anomaly_figure()
    stats = create_statistics()
    device_list = create_device_list()

    return feature_fig, anomaly_fig, stats, device_list, create_status_badge()


def create_feature_figure():
    global detector
    if not detector.realtime_data['timestamp']:
        return create_empty_figure("No Data Available")

    fig = go.Figure()

    # Plot the most important features
    colors = ['blue', 'green', 'orange', 'red', 'purple', 'brown', 'pink', 'gray']
    feature_names = list(detector.realtime_data['feature_values'].keys())[:6]

    for i, feature in enumerate(feature_names):
        if feature in detector.realtime_data['feature_values'] and detector.realtime_data['feature_values'][feature]:
            fig.add_trace(go.Scatter(
                x=detector.realtime_data['timestamp'],
                y=detector.realtime_data['feature_values'][feature],
                mode='lines',
                name=feature.replace('_', ' ').title(),
                line=dict(color=colors[i % len(colors)])
            ))

    # Highlight anomalies
    anomaly_times = [detector.realtime_data['timestamp'][i]
                     for i, anomaly in enumerate(detector.realtime_data['is_anomaly']) if anomaly]

    if anomaly_times and feature_names:
        first_feature = feature_names[0]
        if first_feature in detector.realtime_data['feature_values']:
            anomaly_values = [detector.realtime_data['feature_values'][first_feature][i]
                              for i, anomaly in enumerate(detector.realtime_data['is_anomaly'])
                              if anomaly and i < len(detector.realtime_data['feature_values'][first_feature])]

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
        title="Network Feature Values (from Kafka Stream)",
        xaxis_title="Time",
        yaxis_title="Feature Value",
        hovermode='x unified',
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5)
    )

    return fig


def create_anomaly_figure():
    global detector
    if not detector.realtime_data['timestamp']:
        return create_empty_figure("No Data Available")

    fig = go.Figure()

    # Add reconstruction errors
    colors = ['red' if anomaly else 'blue' for anomaly in detector.realtime_data['is_anomaly']]

    fig.add_trace(go.Scatter(
        x=detector.realtime_data['timestamp'],
        y=detector.realtime_data['reconstruction_error'],
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
    true_anomaly_times = [detector.realtime_data['timestamp'][i]
                          for i, label in enumerate(detector.realtime_data['true_label']) if label == 1]
    true_anomaly_scores = [detector.realtime_data['reconstruction_error'][i]
                           for i, label in enumerate(detector.realtime_data['true_label']) if label == 1]

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
        title=f"{detector.model_name} Anomaly Detection (Real-time Kafka Stream)",
        xaxis_title="Time",
        yaxis_title="Reconstruction Error (L1 Loss)",
        hovermode='x unified'
    )

    return fig


def create_statistics():
    global detector
    if detector.stats['total'] == 0:
        return html.P("No data processed yet")

    # Calculate rates
    tpr = (detector.stats['true_positives'] /
           (detector.stats['true_positives'] + detector.stats['false_negatives']) * 100
           if (detector.stats['true_positives'] + detector.stats['false_negatives']) > 0 else 0)

    fpr = (detector.stats['false_positives'] /
           (detector.stats['false_positives'] + detector.stats['true_negatives']) * 100
           if (detector.stats['false_positives'] + detector.stats['true_negatives']) > 0 else 0)

    return dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("📊 Total Samples"),
                    html.H3(f"{detector.stats['total']}", className="text-primary")
                ])
            ])
        ], width=2),

        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("🚨 Detected"),
                    html.H3(f"{detector.stats['anomalies']}", className="text-danger")
                ])
            ])
        ], width=2),

        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("✅ TPR"),
                    html.H3(f"{tpr:.1f}%", className="text-success")
                ])
            ])
        ], width=2),

        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("❌ FPR"),
                    html.H3(f"{fpr:.1f}%", className="text-info")
                ])
            ])
        ], width=2),

        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("📤 Alerts"),
                    html.H3(f"{detector.stats['alerts_sent']}", className="text-warning"),
                    html.Small(f"({detector.stats['alerts_aggregated']} aggregated)", className="text-muted")
                ])
            ])
        ], width=4)
    ])


def create_device_list():
    global detector
    if not detector.realtime_data['anonymized_device_id']:
        return html.P("No devices detected yet", className="text-muted")

    # Get unique device IDs and their anomaly counts
    device_counts = {}
    for i, device_id in enumerate(detector.realtime_data['anonymized_device_id']):
        if device_id not in device_counts:
            device_counts[device_id] = {'total': 0, 'anomalies': 0}
        device_counts[device_id]['total'] += 1
        if detector.realtime_data['is_anomaly'][i]:
            device_counts[device_id]['anomalies'] += 1

    # Create list items
    device_items = []
    for device_id, counts in sorted(device_counts.items())[-10:]:
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


if __name__ == '__main__':
    logging.info("🛡️ PRIVATEER Network Anomaly Detection Service with UI")
    logging.info("=" * 50)

    # Wait for services
    wait_for_services()

    # Initialize detector
    detector = DetectorWithUI()

    # Load model
    if not detector.load_model():
        logging.error("❌ Failed to load model. Exiting...")
        sys.exit(1)

    # Don't start automatically - wait for user to click Start button
    logging.info("🛑 Detector initialized but not started. Click 'Start Detection' to begin.")

    logging.info(f"🤖 Using {detector.model_name} Model")
    logging.info(f"📱 Device: {detector.device}")
    logging.info(f"🎯 Threshold: {detector.threshold}")
    logging.info(f"📊 Input Features: {detector.feature_list}")
    logging.info(f"🔢 Sequence Length: {detector.seq_len} timesteps")
    logging.info(f"📥 Kafka Topic: {detector.input_topic}")
    logging.info(f"🔐 Privacy Protection: Anonymization Active")
    logging.info(f"📤 Endpoint: {'Enabled' if detector.endpoint_enabled else 'Disabled'}")
    logging.info(f"🔄 Aggregation: {detector.aggregation_manager.window_seconds}s window, "
                f"{detector.aggregation_manager.threshold_count} alert threshold")
    logging.info("=" * 50)
    logging.info("Starting web server...")
    logging.info("Open your browser and go to: http://0.0.0.0:8050")
    logging.info("=" * 50)

    try:
        app.run(host='0.0.0.0', port=8050)
    finally:
        if detector and detector.running:
            detector.stop_kafka_consumer()