"""
PRIVATEER Anomaly Detector Service with Web UI
"""
import os
import json
import queue
import logging
import threading

from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np
import torch
import pandas as pd
import dash
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

from kafka import KafkaConsumer, KafkaProducer
from dash import dcc, html, Input, Output, State

from privateer_ad.utils import load_champion_model
from privateer_ad.config import MLFlowConfig, MetadataConfig

# Configure logging
logging.basicConfig(level=logging.INFO)


class AnomalyDetectorWithUI:
    def __init__(self):
        try:
            # Load model with error handling
            self.metadata = MetadataConfig()
            self.mlflow_config = MLFlowConfig()

            # Try multiple model names
            model_names = ['global_TransformerAD', 'TransformerAD', 'TransformerAD_DP']
            self.model = None

            for model_name in model_names:
                try:
                    self.model, self.threshold, self.loss_fn = load_champion_model(
                        tracking_uri=self.mlflow_config.tracking_uri,
                        model_name=model_name
                    )
                    self.model_name = model_name
                    logging.info(f"Successfully loaded model: {model_name}")
                    break
                except Exception as e:
                    logging.warning(f"Failed to load {model_name}: {e}")
                    continue

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
                'feature_values': {}
            }

            # Initialize feature storage
            for feature in self.input_features:
                self.realtime_data['feature_values'][feature] = []

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
        try:
            # Input data format from anonymizer:
            # {
            #     'device_id': 'network_aggregated',
            #     'features': {
            #         'ul_retx': [0.1, 0.2, 0.15, ...],  # seq_len values
            #         'dl_bitrate': [1.5, 1.8, 1.2, ...], # seq_len values
            #     },
            #     'timestamp': 'latest_timestamp',
            #     'metadata': {'attack': 1}
            # }

            # Create DataFrame from feature sequences
            # Create tensor for model input [batch_size=1, seq_len, features]
            print("input_data['features']", input_data['features'])
            input_features = pd.DataFrame(input_data['features'])
            print("input_features", input_features.values)
            input_tensor = torch.tensor(input_features.values, dtype=torch.float32).to(self.device)
            print('input_tensor',input_tensor)
            print('input_tensor.shape', input_tensor.shape)
            input_tensor = input_tensor.reshape(1, -1, len(self.input_features))

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
                                                             input_data['timestamp'])

                if should_alert:
                    alert = {
                        'alert_id': f"alert-{datetime.now().timestamp()}",
                        'device_id': input_data['device_id'],
                        'cell': input_data['metadata'].get('cell', 'unknown'),
                        'feature_values': input_data['feature_values'],
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
            ui_feature_values = {}
            for feature_name, feature_sequence in input_data['feature_values'].items():
                if feature_name in self.input_features:
                    ui_feature_values[feature_name] = feature_sequence[-1]  # Latest value

            ui_data = {
                'timestamp': input_data['timestamp'],
                'sample_index': self.sample_count,
                'reconstruction_error': detection_results['reconstruction_error'],
                'is_anomaly': is_anomaly_detected,
                'true_label': true_label,
                'device_id': input_data['device_id'],
                'feature_values': ui_feature_values
            }

            # Non-blocking put
            try:
                self.ui_data_queue.put_nowait(ui_data)
            except queue.Full:
                self.ui_data_queue.get()
                self.ui_data_queue.put_nowait(ui_data)

        except Exception as e:
            logging.error(f"Error processing message: {e}")
            logging.error(f"Input data type: {type(input_data)}")
            logging.error(f"Input data sample: {str(input_data)[:500]}...")

    def detect_anomaly(self, input_tensor):
        """Run anomaly detection on incoming data"""
        try:
            with torch.no_grad():
                print('input_tensor', np.shape(input_tensor))
                output = self.model(input_tensor)
                print('output', output)
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
                        dbc.Button("⏸️ Stop Detection", id="stop-btn", color="danger"),
                    ]),
                    html.Hr(),
                    html.Div([
                        html.Label("🎯 Anomaly Threshold:", className="form-label"),
                        dcc.Slider(
                            id='threshold-slider',
                            min=detector.initial_threshold * 0.5,
                            max=detector.initial_threshold * 1.5,
                            step=0.001,
                            value=detector.threshold,
                            marks={
                                detector.initial_threshold * 0.5: f"{detector.initial_threshold * 0.5:.3f}",
                                detector.initial_threshold: f"{detector.initial_threshold:.3f}",
                                detector.initial_threshold * 1.5: f"{detector.initial_threshold * 1.5:.3f}"
                            },
                            tooltip={"placement": "bottom", "always_visible": True}
                        )
                    ], className="mb-3"),
                    html.Div([
                        html.Label("🔐 Privacy Protection: ", className="form-label"),
                        dbc.Badge("Network Aggregation Active", color="success", className="ms-2"),
                        html.Small(" - Cross-device aggregated analytics", className="text-muted ms-2")
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
                    html.H4("📊 Network Feature Values", className="card-title"),
                    dcc.Graph(id="feature-display", style={'height': '400px'})
                ])
            ])
        ], width=6),

        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("🚨 Network Anomaly Detection", className="card-title"),
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
                    html.H4("🌐 Network Status", className="card-title"),
                    html.Div(id="network-status", style={'max-height': '200px', 'overflow-y': 'auto'})
                ])
            ])
        ], width=4)
    ]),

    dcc.Interval(id='interval-component', interval=1000, n_intervals=0),
    dcc.Store(id='detector-state', data={'running': False})
], fluid=True)


# Callbacks
@app.callback(
    [Output('detector-state', 'data'),
     Output('start-btn', 'disabled'),
     Output('stop-btn', 'disabled')],
    [Input('start-btn', 'n_clicks'),
     Input('stop-btn', 'n_clicks')],
    [State('detector-state', 'data')]
)
def control_detector(start_clicks, stop_clicks, state):
    ctx = dash.callback_context
    if not ctx.triggered:
        return state, False, True

    button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if button_id == 'start-btn' and start_clicks:
        detector.start_consumer()
        return {'running': True}, True, False
    elif button_id == 'stop-btn' and stop_clicks:
        detector.stop_consumer()
        return {'running': False}, False, True

    return state, state.get('running', False), not state.get('running', False)


@app.callback(
    Output('status-indicator', 'children'),
    [Input('threshold-slider', 'value'),
     Input('detector-state', 'data')]
)
def update_status_indicator(threshold, state):
    if threshold:
        detector.update_threshold(threshold)
    running = state.get('running', False)
    status_text = "Running" if running else "Stopped"
    status_color = "success" if running else "secondary"

    return html.Div([
        dbc.Badge(f"Threshold: {threshold:.3f}", color="info", className="me-2"),
        dbc.Badge(f"Status: {status_text}", color=status_color),
        dbc.Badge(f"Model: {detector.model_name}", color="primary", className="ms-2")
    ])


@app.callback(
    [Output('feature-display', 'figure'),
     Output('anomaly-detection', 'figure'),
     Output('stats-display', 'children'),
     Output('network-status', 'children')],
    [Input('interval-component', 'n_intervals')],
    [State('detector-state', 'data')]
)
def update_graphs(n, state):
    if not state.get('running', False):
        empty_fig = create_empty_figure("Detection Stopped - Click Start to begin")
        return empty_fig, empty_fig, html.P("Start detection to see statistics"), html.P("Network monitoring inactive")

    # Get new data
    latest_data = detector.get_latest_data()
    # Add new data to realtime storage
    for data_point in latest_data:
        detector.realtime_data['timestamp'].append(data_point['timestamp'])
        detector.realtime_data['sample_index'].append(data_point['sample_index'])
        detector.realtime_data['reconstruction_error'].append(data_point['reconstruction_error'])
        detector.realtime_data['is_anomaly'].append(data_point['is_anomaly'])
        detector.realtime_data['true_label'].append(data_point['true_label'])
        detector.realtime_data['device_id'].append(data_point['device_id'])

        # Add feature values
        for feature, value in data_point['feature_values'].items():
            if feature in detector.realtime_data['feature_values']:
                detector.realtime_data['feature_values'][feature].append(value)

    # Limit data size
    if len(detector.realtime_data['timestamp']) > detector.max_points:
        for key in detector.realtime_data:
            if key != 'feature_values':
                detector.realtime_data[key] = detector.realtime_data[key][-detector.max_points:]
            else:
                for feature in detector.realtime_data[key]:
                    detector.realtime_data[key][feature] = \
                        detector.realtime_data[key][feature][-detector.max_points:]
    return create_feature_figure(), create_anomaly_figure(), create_statistics(), create_network_status()


def create_feature_figure():
    if not detector.realtime_data['timestamp']:
        return create_empty_figure("No Data Available")

    fig = go.Figure()
    colors = ['blue', 'green', 'orange', 'red', 'purple', 'brown']
    feature_names = list(detector.realtime_data['feature_values'].keys())[:6]

    for i, feature in enumerate(feature_names):
        if feature in detector.realtime_data['feature_values']:
            fig.add_trace(go.Scatter(
                x=detector.realtime_data['timestamp'],
                y=detector.realtime_data['feature_values'][feature],
                mode='lines',
                name=feature.replace('_', ' ').title(),
                line=dict(color=colors[i % len(colors)])
            ))

    fig.update_layout(
        title="Aggregated Network Feature Values",
        xaxis_title="Time",
        yaxis_title="Feature Value",
        hovermode='x unified'
    )
    return fig


def create_anomaly_figure():
    if not detector.realtime_data['timestamp']:
        return create_empty_figure("No Data Available")

    fig = go.Figure()

    # Categorize points by TP, TN, FP, FN
    timestamps = detector.realtime_data['timestamp']
    errors = detector.realtime_data['reconstruction_error']
    predictions = detector.realtime_data['is_anomaly']
    true_labels = detector.realtime_data['true_label']

    # Create lists for each category
    tp_times, tp_errors = [], []
    tn_times, tn_errors = [], []
    fp_times, fp_errors = [], []
    fn_times, fn_errors = [], []

    # Categorize for confusion matrix visualization
    for i in range(len(timestamps)):
        is_anomaly = predictions[i]
        actual_label = true_labels[i]

        if is_anomaly and actual_label == 1:  # True Positive
            tp_times.append(timestamps[i])
            tp_errors.append(errors[i])
        elif not is_anomaly and actual_label == 0:  # True Negative
            tn_times.append(timestamps[i])
            tn_errors.append(errors[i])
        elif is_anomaly and actual_label == 0:  # False Positive
            fp_times.append(timestamps[i])
            fp_errors.append(errors[i])
        elif not is_anomaly and actual_label == 1:  # False Negative
            fn_times.append(timestamps[i])
            fn_errors.append(errors[i])

    # Add baseline line
    if len(timestamps) > 1:
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=errors,
            mode='lines',
            name='Network Reconstruction Error',
            line=dict(color='lightblue', width=2),
            hoverinfo='skip'
        ))

    # Add True Positives (Correctly detected attacks)
    if tp_times:
        fig.add_trace(go.Scatter(
            x=tp_times,
            y=tp_errors,
            mode='markers',
            name='True Positive (TP)',
            marker=dict(color='green', size=10, symbol='circle'),
            hovertemplate='<b>True Positive</b><br>Time: %{x}<br>Error: %{y:.4f}<extra></extra>'
        ))

    # Add True Negatives (Correctly identified benign)
    if tn_times:
        fig.add_trace(go.Scatter(
            x=tn_times,
            y=tn_errors,
            mode='markers',
            name='True Negative (TN)',
            marker=dict(color='blue', size=6, symbol='circle'),
            hovertemplate='<b>True Negative</b><br>Time: %{x}<br>Error: %{y:.4f}<extra></extra>'
        ))

    # Add False Positives (Incorrectly flagged as attacks)
    if fp_times:
        fig.add_trace(go.Scatter(
            x=fp_times,
            y=fp_errors,
            mode='markers',
            name='False Positive (FP)',
            marker=dict(color='orange', size=10, symbol='triangle-up'),
            hovertemplate='<b>False Positive</b><br>Time: %{x}<br>Error: %{y:.4f}<extra></extra>'
        ))

    # Add False Negatives (Missed attacks)
    if fn_times:
        fig.add_trace(go.Scatter(
            x=fn_times,
            y=fn_errors,
            mode='markers',
            name='False Negative (FN)',
            marker=dict(color='red', size=10, symbol='triangle-down'),
            hovertemplate='<b>False Negative</b><br>Time: %{x}<br>Error: %{y:.4f}<extra></extra>'
        ))

    # Add threshold line
    fig.add_hline(
        y=detector.threshold,
        line_dash="dash",
        line_color="black",
        line_width=3,
        annotation_text=f"Threshold ({detector.threshold:.6f})",
        annotation_position="top right"
    )

    fig.update_layout(
        title=f"{detector.model_name} - Network Anomaly Detection",
        xaxis_title="Time",
        yaxis_title="Reconstruction Error",
        hovermode='closest'
    )

    return fig


def create_statistics():
    """Proper TPR/FPR calculation and display"""
    if detector.stats['total'] == 0:
        return html.P("No data processed yet")

    # Calculate TPR and FPR
    tpr, fpr = detector.calculate_tpr_fpr()

    return dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("📊 Total"),
                    html.H3(f"{detector.stats['total']}", className="text-primary")
                ])
            ])
        ], width=2),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("🚨 Detected"),
                    html.H3(f"{detector.stats['anomalies_detected']}", className="text-danger")
                ])
            ])
        ], width=2),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("✅ TPR"),
                    html.H3(f"{tpr:.1f}%", className="text-success"),
                    html.Small(f"TP: {detector.stats['true_positives']}")
                ])
            ])
        ], width=2),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("❌ FPR"),
                    html.H3(f"{fpr:.1f}%", className="text-warning"),
                    html.Small(f"FP: {detector.stats['false_positives']}")
                ])
            ])
        ], width=2),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("🎯 Attacks"),
                    html.H3(f"{detector.stats['true_attacks']}", className="text-info")
                ])
            ])
        ], width=2),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("📤 Alerts"),
                    html.H3(f"{detector.stats['alerts_sent']}", className="text-secondary")
                ])
            ])
        ], width=2)
    ])


def create_network_status():
    if not detector.realtime_data['device_id']:
        return html.P("Network monitoring inactive", className="text-muted")

    # Since we're doing network aggregation, show network-level stats
    total_sequences = len(detector.realtime_data['device_id'])
    anomaly_sequences = sum(detector.realtime_data['is_anomaly'])
    anomaly_rate = (anomaly_sequences / total_sequences) * 100 if total_sequences > 0 else 0

    color = "danger" if anomaly_rate > 50 else "warning" if anomaly_rate > 20 else "success"

    return dbc.ListGroup([
        dbc.ListGroupItem([
            html.Div([
                html.Span("Network Aggregated Analytics", className="fw-bold"),
                dbc.Badge(f"{anomaly_rate:.1f}%", color=color, className="float-end")
            ]),
            html.Small(f"Total Sequences: {total_sequences}, Anomalous: {anomaly_sequences}")
        ]),
        dbc.ListGroupItem([
            html.Div([
                html.Span("Privacy Status", className="fw-bold"),
                dbc.Badge("Active", color="success", className="float-end")
            ]),
            html.Small("Cross-device data aggregation enabled")
        ])
    ], flush=True)


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
    logging.info("🛡️ PRIVATEER Network Anomaly Detection Service")
    logging.info(f"📱 Device: {detector.device}")
    logging.info(f"📥 Kafka Topic: {detector.input_topic}")
    logging.info(f"🎯 Initial Threshold: {detector.initial_threshold}")
    logging.info(f"📏 Threshold Range: {detector.initial_threshold * 0.5:.3f} - {detector.initial_threshold * 1.5:.3f}")
    app.run(host='0.0.0.0', port=8050, debug=False)