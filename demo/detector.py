"""
PRIVATEER Anomaly Detector Service with Web UI
"""
import os
import json
import queue
import logging
import threading

from collections import defaultdict, deque
from datetime import datetime, timedelta

import pandas as pd
import torch
import dash
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from kafka import KafkaConsumer, KafkaProducer
from dash import dcc, html, Input, Output, State

from privateer_ad.utils import load_champion_model
from privateer_ad.config import MLFlowConfig, MetadataConfig

# Configure logging
logging.basicConfig(level=logging.INFO)


class AnomalyDetectorWithUI:
    def __init__(self, model_name='TransformerAD_DP'):
        try:
            # Load model with error handling
            self.metadata = MetadataConfig()
            self.mlflow_config = MLFlowConfig()

            self.model = None
            try:
                self.model, self.threshold, self.loss_fn = load_champion_model(
                    tracking_uri=self.mlflow_config.tracking_uri,
                    model_name=model_name
                )
                self.model_name = model_name
                logging.info(f"Successfully loaded model: {model_name}")
            except Exception as e:
                logging.warning(f"Failed to load {model_name}: {e}")

            if self.model is None:
                raise ValueError("No model could be loaded. Check MLflow registry.")

            self.model.eval()
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model.to(self.device)

            # Store initial threshold for slider limits
            self.initial_threshold = self.threshold

            # Get input features for display
            self.input_features = self.metadata.get_input_features()

            # Alert aggregation
            self.alert_window = timedelta(minutes=5)
            self.device_alerts = defaultdict(list)
            self.alert_threshold = 5

            # Kafka setup
            print(os.environ.get('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092'))
            self.bootstrap_servers = os.environ.get('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092')
            self.input_topic = os.environ.get('INPUT_TOPIC', 'preprocessed-data')
            self.alert_topic = os.environ.get('ALERT_TOPIC', 'anomaly-alerts')

            self.consumer = None
            self.producer = None
            self.running = False
            self.consumer_thread = None

            # Data queue for UI updates
            self.ui_data_queue = queue.Queue(maxsize=1000)

            # Storage for real-time data
            self.realtime_data = {
                'timestamp': [],
                'sample_index': [],
                'reconstruction_error': [],
                'is_anomaly': [],
                'true_label': [],
                'device_id': [],
                'features': {}
            }

            # Initialize feature storage
            for feature in self.input_features:
                self.realtime_data['features'][feature] = []

            self.max_points = 200
            self.sample_count = 0

            self.stats = {
                'total': 0,
                'anomalies_detected': 0,
                'true_attacks': 0,
                'true_positives': 0,
                'true_negatives': 0,
                'false_positives': 0,
                'false_negatives': 0,
                'alerts_sent': 0
            }

            logging.info("AnomalyDetectorWithUI initialized successfully")

        except Exception as e:
            logging.error(f"Failed to initialize detector: {e}")
            raise

    def kafka_consumer_loop(self):
        """Main Kafka consumer loop running in background"""
        try:
            self.consumer = KafkaConsumer(
                self.input_topic,
                bootstrap_servers=self.bootstrap_servers,
                value_deserializer=lambda v: json.loads(v.decode('utf-8')),
                auto_offset_reset='latest',
                group_id='anomaly-detector-group'
            )

            self.producer = KafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode('utf-8')
            )

            logging.info("Kafka consumer started")

            while self.running:
                try:
                    messages = self.consumer.poll(timeout_ms=1000)

                    for topic_partition, records in messages.items():
                        for message in records:
                            self.process_message(message.value)

                except Exception as e:
                    logging.error(f"Error in consumer loop: {e}")
                    continue

        except Exception as e:
            logging.error(f"Failed to start Kafka consumer: {e}")
        finally:
            if self.consumer:
                self.consumer.close()
            if self.producer:
                self.producer.close()
            logging.info("Kafka consumer stopped")

    def process_message(self, input_data):
        """Process a single message from Kafka - Updated for network sequences"""
        # try:
        # Flatten the nested lists in features
        flattened_features = {}
        for feature_name, feature_values in input_data['features'].items():
            # Convert [[val1], [val2], ...] to [val1, val2, ...]
            if isinstance(feature_values, list) and len(feature_values) > 0:
                if isinstance(feature_values[0], list):
                    # Flatten nested lists
                    flattened_features[feature_name] = [item[0] if isinstance(item, list) else item
                                                        for item in feature_values]
                else:
                    flattened_features[feature_name] = feature_values
            else:
                flattened_features[feature_name] = feature_values

        # Create DataFrame from flattened feature sequences
        input_features = pd.DataFrame(flattened_features)
        # Ensure only the input features we need are selected and in correct order
        ordered_features = [col for col in self.input_features if col in input_features.columns]
        input_features = input_features[ordered_features]

        # Convert to tensor with explicit dtype
        input_tensor = torch.tensor(input_features.values, dtype=torch.float32).to(self.device)

        # Reshape to [batch_size=1, seq_len, num_features]
        input_tensor = input_tensor.reshape(1, -1, len(ordered_features))

        detection_results = self.detect_anomaly(input_tensor)
        if detection_results is None:
            return

        # Statistics tracking
        self.sample_count += 1
        self.stats['total'] += 1

        is_anomaly_detected = detection_results['is_anomaly']
        true_label = input_data['metadata'].get('attack', 0)  # 1 for attack, 0 for benign

        # Update confusion matrix components
        if true_label == 1:
            self.stats['true_attacks'] += 1
            if is_anomaly_detected:
                self.stats['true_positives'] += 1
            else:
                self.stats['false_negatives'] += 1
        else:  # true_label == 0 (benign)
            if not is_anomaly_detected:
                self.stats['true_negatives'] += 1
            else:
                self.stats['false_positives'] += 1

        if is_anomaly_detected:
            self.stats['anomalies_detected'] += 1

            # Check if we should send alert
            should_alert, count = self.should_send_alert(input_data['device_id'],
                                                         input_data['timestamp'][-1])

            if should_alert:
                alert = {
                    'alert_id': f"alert-{datetime.now().timestamp()}",
                    'device_id': input_data['device_id'],
                    'cell': input_data['metadata'].get('cell', 'unknown'),
                    'features': input_data['features'],
                    'timestamp': input_data['timestamp'],
                    'reconstruction_error': detection_results['reconstruction_error'],
                    'threshold': self.threshold,
                    'anomaly_count': count,
                    'window_seconds': self.alert_window.seconds,
                    'data': input_data
                }

                self.producer.send(self.alert_topic, value=alert)
                self.stats['alerts_sent'] += 1
                logging.info(f"ALERT sent for network sequence")

        # Queue data for UI update - use latest values from sequence
        ui_features = {}
        for feature_name, feature_sequence in input_data['features'].items():
            if feature_name in self.input_features:
                ui_features[feature_name] = feature_sequence[-1]  # Latest value

        ui_data = {
            'timestamp': input_data['timestamp'][-1],
            'sample_index': self.sample_count,
            'reconstruction_error': detection_results['reconstruction_error'],
            'is_anomaly': is_anomaly_detected,
            'true_label': true_label,
            'device_id': input_data['device_id'],
            'features': ui_features
        }

        # Non-blocking put
        try:
            self.ui_data_queue.put_nowait(ui_data)
        except queue.Full:
            self.ui_data_queue.get()
            self.ui_data_queue.put_nowait(ui_data)

        # except Exception as e:
        #     logging.error(f"Error processing message: {e}")
        #     logging.error(f"Input data type: {type(input_data)}")
        #     logging.error(f"Input data sample: {str(input_data)[:500]}...")

    def detect_anomaly(self, input_tensor):
        """Run anomaly detection on incoming data"""
        try:
            with torch.no_grad():
                output = self.model(input_tensor)
                reconstruction_error = self.loss_fn(input_tensor, output)
                reconstruction_error = reconstruction_error.mean().item()
            is_anomaly = reconstruction_error > self.threshold

            return {
                'is_anomaly': is_anomaly,
                'reconstruction_error': reconstruction_error,
                'input_tensor': input_tensor
            }

        except Exception as e:
            logging.error(f"Error in anomaly detection: {e}")
            return None

    def should_send_alert(self, device_id, timestamp):
        """Aggregate alerts to prevent spam"""
        print('timestamp', timestamp)
        current_time = datetime.fromisoformat(timestamp) if isinstance(timestamp, str) else timestamp

        # Clean old alerts
        cutoff_time = current_time - self.alert_window
        self.device_alerts[device_id] = [
            t for t in self.device_alerts[device_id]
            if t > cutoff_time
        ]

        # Add current alert
        self.device_alerts[device_id].append(current_time)

        # Check if we should send alert
        alert_count = len(self.device_alerts[device_id])
        if alert_count >= self.alert_threshold:
            self.device_alerts[device_id] = []
            return True, alert_count

        return False, alert_count

    def start_consumer(self):
        """Start Kafka consumer in background thread"""
        if not self.running:
            self.running = True
            self.consumer_thread = threading.Thread(target=self.kafka_consumer_loop)
            self.consumer_thread.daemon = True
            self.consumer_thread.start()
            logging.info("Started Kafka consumer")

    def stop_consumer(self):
        """Stop Kafka consumer"""
        if self.running:
            self.running = False
            if self.consumer_thread:
                self.consumer_thread.join(timeout=5)
            logging.info("Stopped Kafka consumer")

    def get_latest_data(self):
        """Get all available data from queue"""
        data = []
        while not self.ui_data_queue.empty():
            try:
                data.append(self.ui_data_queue.get_nowait())
            except queue.Empty:
                break
        return data

    def update_threshold(self, new_threshold):
        """Update the anomaly threshold"""
        self.threshold = float(new_threshold)
        logging.info(f"Threshold updated to: {new_threshold}")

    def calculate_tpr_fpr(self):
        """Calculate True Positive Rate and False Positive Rate"""
        if self.stats['true_attacks'] > 0:
            tpr = (self.stats['true_positives'] / self.stats['true_attacks']) * 100
        else:
            tpr = 0.0

        benign_samples = self.stats['total'] - self.stats['true_attacks']
        if benign_samples > 0:
            fpr = (self.stats['false_positives'] / benign_samples) * 100
        else:
            fpr = 0.0

        return tpr, fpr


# Initialize detector

try:
    detector = AnomalyDetectorWithUI()
except Exception as e:
    logging.error(f"Failed to initialize detector: {e}")
    exit(1)

# Create Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "PRIVATEER - Network Anomaly Detection"

# Initialize data storage with deques for efficiency
MAX_POINTS = 500
data_storage = {
    'timestamps': deque(maxlen=MAX_POINTS),
    'reconstruction_errors': deque(maxlen=MAX_POINTS),
    'thresholds': deque(maxlen=MAX_POINTS),
    'is_anomaly': deque(maxlen=MAX_POINTS),
    'true_labels': deque(maxlen=MAX_POINTS),
    'features': {}  # Will be populated dynamically
}

# Stats tracking
stats = {
    'total': 0,
    'tp': 0,
    'fp': 0,
    'tn': 0,
    'fn': 0
}

# App layout - SIMPLE AND CLEAN
app.layout = dbc.Container([
    # Header
    dbc.Row([
        dbc.Col([
            html.H2("🛡️ PRIVATEER Network Anomaly Detection", className="text-center mb-3"),
            html.Hr()
        ])
    ]),

    # Control Panel
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            dbc.Button("▶️ START", id="start-btn", color="success", size="lg", className="w-100"),
                        ], width=2),
                        dbc.Col([
                            dbc.Button("⏹️ STOP", id="stop-btn", color="danger", size="lg", className="w-100"),
                        ], width=2),
                        dbc.Col([
                            html.Div([
                                html.Label(f"Threshold: {detector.threshold:.4f}", id="threshold-label"),
                                dcc.Slider(
                                    id='threshold-slider',
                                    min=0.001,
                                    max=1.0,
                                    step=0.001,
                                    value=detector.threshold,
                                    marks={i / 10: f'{i / 10:.1f}' for i in range(11)},
                                )
                            ])
                        ], width=6),
                        dbc.Col([
                            html.Div(id="live-stats", className="text-end")
                        ], width=2)
                    ])
                ])
            ])
        ])
    ], className="mb-3"),

    # Main Plots
    dbc.Row([
        dbc.Col([
            dcc.Graph(id="main-plot", style={'height': '70vh'})
        ])
    ]),

    # Update interval
    dcc.Interval(id='update-interval', interval=500, n_intervals=0),
    dcc.Store(id='is-running', data=False)
], fluid=True)


# Control callbacks
@app.callback(
    [Output('is-running', 'data'),
     Output('start-btn', 'disabled'),
     Output('stop-btn', 'disabled')],
    [Input('start-btn', 'n_clicks'),
     Input('stop-btn', 'n_clicks')],
    [State('is-running', 'data')]
)
def control_detection(start, stop, running):
    ctx = dash.callback_context
    if not ctx.triggered:
        return False, False, True

    trigger = ctx.triggered[0]['prop_id'].split('.')[0]

    if trigger == 'start-btn':
        detector.start_consumer()
        logging.info("✅ Detection started")
        return True, True, False
    elif trigger == 'stop-btn':
        detector.stop_consumer()
        logging.info("🛑 Detection stopped")
        return False, False, True

    return running, running, not running


@app.callback(
    Output('threshold-label', 'children'),
    [Input('threshold-slider', 'value')]
)
def update_threshold(value):
    if value:
        detector.update_threshold(value)
    return f"Threshold: {value:.4f}"


# Main update callback
@app.callback(
    [Output('main-plot', 'figure'),
     Output('live-stats', 'children')],
    [Input('update-interval', 'n_intervals')],
    [State('is-running', 'data')]
)
def update_dashboard(n, running):
    if not running:
        # Return empty plot when stopped
        fig = go.Figure()
        fig.add_annotation(
            text="Detection Stopped - Press START to begin",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20, color="gray")
        )
        fig.update_layout(
            xaxis={'visible': False},
            yaxis={'visible': False},
            height=700
        )
        return fig, "Status: Stopped"

    # Get latest data from detector
    new_data = detector.get_latest_data()

    # Process new data points
    for point in new_data:
        # Store timestamp
        data_storage['timestamps'].append(point['timestamp'])
        data_storage['reconstruction_errors'].append(point['reconstruction_error'])
        data_storage['thresholds'].append(detector.threshold)
        data_storage['is_anomaly'].append(point['is_anomaly'])
        data_storage['true_labels'].append(point['true_label'])

        # Store features
        for feat_name, feat_val in point['features'].items():
            if feat_name not in data_storage['features']:
                data_storage['features'][feat_name] = deque(maxlen=MAX_POINTS)
            data_storage['features'][feat_name].append(feat_val[-1])

        # Update stats
        stats['total'] += 1
        if point['is_anomaly'] and point['true_label'] == 1:
            stats['tp'] += 1
        elif point['is_anomaly'] and point['true_label'] == 0:
            stats['fp'] += 1
        elif not point['is_anomaly'] and point['true_label'] == 1:
            stats['fn'] += 1
        else:
            stats['tn'] += 1

    # Create the plot
    fig = create_dashboard_plot()

    # Calculate metrics
    tpr = (stats['tp'] / (stats['tp'] + stats['fn']) * 100) if (stats['tp'] + stats['fn']) > 0 else 0
    fpr = (stats['fp'] / (stats['fp'] + stats['tn']) * 100) if (stats['fp'] + stats['tn']) > 0 else 0

    stats_html = html.Div([
        html.Strong(f"TPR: {tpr:.1f}%"),
        html.Br(),
        html.Strong(f"FPR: {fpr:.1f}%"),
        html.Br(),
        html.Small(f"Total: {stats['total']}")
    ])

    return fig, stats_html


def create_dashboard_plot():
    """Create the main dashboard plot with features and reconstruction error"""

    # Select top 5 features to display
    feature_names = list(data_storage['features'].keys())[:5]
    num_features = len(feature_names)

    # Create subplots: features + reconstruction error
    fig = make_subplots(
        rows=num_features + 1,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        subplot_titles=[*[f"Feature: {name}" for name in feature_names], "Reconstruction Error & Anomaly Detection"],
        row_heights=[1] * num_features + [2]  # Make error plot bigger
    )

    timestamps = list(data_storage['timestamps'])

    if not timestamps:
        fig.add_annotation(
            text="Waiting for data...",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="gray")
        )
        return fig

    # Plot features
    colors = ['blue', 'green', 'orange', 'purple', 'brown']
    for i, feat_name in enumerate(feature_names):
        if feat_name in data_storage['features']:
            feat_values = list(data_storage['features'][feat_name])

            # Ensure same length as timestamps
            if len(feat_values) == len(timestamps):
                fig.add_trace(
                    go.Scatter(
                        x=timestamps,
                        y=feat_values,
                        mode='lines',
                        name=feat_name,
                        line=dict(color=colors[i % len(colors)], width=1.5),
                        showlegend=False
                    ),
                    row=i + 1, col=1
                )

    # Plot reconstruction error and threshold
    errors = list(data_storage['reconstruction_errors'])
    thresholds = list(data_storage['thresholds'])
    anomalies = list(data_storage['is_anomaly'])
    true_labels = list(data_storage['true_labels'])

    if len(errors) == len(timestamps):
        # Base error line
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=errors,
                mode='lines',
                name='Reconstruction Error',
                line=dict(color='lightblue', width=2),
                showlegend=True
            ),
            row=num_features + 1, col=1
        )

        # Threshold line
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=thresholds,
                mode='lines',
                name='Threshold',
                line=dict(color='red', width=2, dash='dash'),
                showlegend=True
            ),
            row=num_features + 1, col=1
        )

        # Mark anomalies
        anomaly_times = [timestamps[i] for i in range(len(timestamps)) if anomalies[i]]
        anomaly_errors = [errors[i] for i in range(len(errors)) if i < len(anomalies) and anomalies[i]]

        if anomaly_times:
            # Separate by true/false positives
            tp_times = [anomaly_times[i] for i in range(len(anomaly_times))
                        if i < len(true_labels) and true_labels[i] == 1]
            tp_errors = [anomaly_errors[i] for i in range(len(anomaly_errors))
                         if i < len(true_labels) and true_labels[i] == 1]

            fp_times = [anomaly_times[i] for i in range(len(anomaly_times))
                        if i < len(true_labels) and true_labels[i] == 0]
            fp_errors = [anomaly_errors[i] for i in range(len(anomaly_errors))
                         if i < len(true_labels) and true_labels[i] == 0]

            if tp_times:
                fig.add_trace(
                    go.Scatter(
                        x=tp_times,
                        y=tp_errors,
                        mode='markers',
                        name='True Positive',
                        marker=dict(color='green', size=8, symbol='circle'),
                        showlegend=True
                    ),
                    row=num_features + 1, col=1
                )

            if fp_times:
                fig.add_trace(
                    go.Scatter(
                        x=fp_times,
                        y=fp_errors,
                        mode='markers',
                        name='False Positive',
                        marker=dict(color='orange', size=8, symbol='triangle-up'),
                        showlegend=True
                    ),
                    row=num_features + 1, col=1
                )

    # Update layout
    fig.update_xaxes(title_text="Time", row=num_features + 1, col=1)
    fig.update_yaxes(title_text="Value", row=num_features + 1, col=1)

    fig.update_layout(
        height=700,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.15,
            xanchor="center",
            x=0.5
        ),
        margin=dict(l=50, r=50, t=50, b=100),
        hovermode='x unified'
    )

    return fig


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logging.info("=" * 50)
    logging.info("PRIVATEER Network Anomaly Detection")
    logging.info(f"Device: {detector.device}")
    logging.info(f"Model: {detector.model_name}")
    logging.info(f"Initial Threshold: {detector.threshold:.6f}")
    logging.info("=" * 50)
    logging.info("Dashboard: http://0.0.0.0:8050")

    app.run(host='0.0.0.0', port=8050, debug=False)