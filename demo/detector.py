# demo/detector.py
"""
PRIVATEER Anomaly Detector Service with Web UI
Combines Kafka consumption with real-time Dash visualization
"""
import os
import json
import torch
import numpy as np
import threading
import queue
from kafka import KafkaConsumer, KafkaProducer
from collections import defaultdict
from datetime import datetime, timedelta
import sys

sys.path.append('/app')

import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go

from privateer_ad.utils import load_champion_model
from privateer_ad.config import MLFlowConfig, MetadataConfig


class AnomalyDetectorWithUI:
    def __init__(self):
        # Load model
        mlflow_config = MLFlowConfig()
        self.metadata = MetadataConfig()

        self.model, self.threshold, self.loss_fn = load_champion_model(
            mlflow_config.tracking_uri,
            model_name='TransformerAD_DP'
        )
        self.model.eval()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        # Get input features for display
        self.input_features = self.metadata.get_input_features()

        # Alert aggregation
        self.alert_window = timedelta(minutes=5)
        self.device_alerts = defaultdict(list)
        self.alert_threshold = 5

        # Kafka setup
        self.bootstrap_servers = os.environ.get('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092')
        self.input_topic = os.environ.get('INPUT_TOPIC', 'preprocessed-data')
        self.alert_topic = os.environ.get('ALERT_TOPIC', 'anomaly-alerts')

        self.consumer = None
        self.producer = None
        self.running = False
        self.consumer_thread = None

        # Data queue for UI updates
        self.data_queue = queue.Queue(maxsize=1000)

        # Storage for real-time data (similar to demo.py)
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
        for feature in self.input_features:
            self.realtime_data['feature_values'][feature] = []

        self.max_points = 200  # Keep last 200 points
        self.sample_count = 0

        # Statistics
        self.stats = {
            'total': 0,
            'anomalies': 0,
            'true_positives': 0,
            'false_positives': 0,
            'alerts_sent': 0
        }

    def detect_anomaly(self, tensor_data):
        """Run anomaly detection on preprocessed tensor"""
        tensor = torch.tensor(tensor_data, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            output = self.model(tensor)
            reconstruction_error = self.loss_fn(tensor, output).mean().item()

        is_anomaly = reconstruction_error > self.threshold

        return {
            'is_anomaly': is_anomaly,
            'reconstruction_error': reconstruction_error,
            'threshold': self.threshold
        }

    def should_send_alert(self, device_id, timestamp):
        """Aggregate alerts to prevent spam"""
        current_time = datetime.fromisoformat(timestamp)

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

    def kafka_consumer_loop(self):
        """Main Kafka consumer loop running in background"""
        self.consumer = KafkaConsumer(
            self.input_topic,
            bootstrap_servers=self.bootstrap_servers,
            value_deserializer=lambda v: json.loads(v.decode('utf-8')),
            auto_offset_reset='latest'
        )

        self.producer = KafkaProducer(
            bootstrap_servers=self.bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )

        print("Kafka consumer started")

        while self.running:
            try:
                # Poll with timeout to allow checking self.running
                messages = self.consumer.poll(timeout_ms=1000)

                for topic_partition, records in messages.items():
                    for message in records:
                        self.process_message(message.value)

            except Exception as e:
                print(f"Error in consumer loop: {e}")
                continue

        self.consumer.close()
        self.producer.close()
        print("Kafka consumer stopped")

    def process_message(self, data):
        """Process a single message from Kafka"""
        try:
            # Detect anomaly
            result = self.detect_anomaly(data['tensor'])

            # Update statistics
            self.sample_count += 1
            self.stats['total'] += 1

            if result['is_anomaly']:
                self.stats['anomalies'] += 1

                # Check ground truth
                if data['metadata']['attack'] == 1:
                    self.stats['true_positives'] += 1
                else:
                    self.stats['false_positives'] += 1

                # Check if we should send alert
                device_id = data['device_id']
                timestamp = data['timestamp']
                should_alert, count = self.should_send_alert(device_id, timestamp)

                if should_alert:
                    alert = {
                        'alert_id': f"alert-{datetime.now().timestamp()}",
                        'device_id': device_id,
                        'timestamp': timestamp,
                        'reconstruction_error': result['reconstruction_error'],
                        'threshold': result['threshold'],
                        'anomaly_count': count,
                        'window_minutes': self.alert_window.seconds // 60,
                        'metadata': data['metadata']
                    }

                    self.producer.send(self.alert_topic, value=alert)
                    self.stats['alerts_sent'] += 1
                    print(f"ALERT sent for device {device_id}")

            # Extract feature values from tensor for visualization
            # Take the last timestep from the sequence [1, seq_len, features]
            last_features = data['tensor'][0][-1]  # Last timestep
            feature_dict = {
                feature: last_features[i]
                for i, feature in enumerate(self.input_features)
            }

            # Queue data for UI update
            ui_data = {
                'timestamp': datetime.fromisoformat(data['timestamp']),
                'sample_index': self.sample_count,
                'reconstruction_error': result['reconstruction_error'],
                'is_anomaly': result['is_anomaly'],
                'true_label': data['metadata']['attack'],
                'anonymized_device_id': data['device_id'],
                'feature_values': feature_dict
            }

            # Non-blocking put
            try:
                self.data_queue.put_nowait(ui_data)
            except queue.Full:
                # Remove oldest item and retry
                self.data_queue.get()
                self.data_queue.put_nowait(ui_data)

        except Exception as e:
            print(f"Error processing message: {e}")

    def start_consumer(self):
        """Start Kafka consumer in background thread"""
        self.running = True
        self.consumer_thread = threading.Thread(target=self.kafka_consumer_loop)
        self.consumer_thread.daemon = True
        self.consumer_thread.start()

    def stop_consumer(self):
        """Stop Kafka consumer"""
        self.running = False
        if self.consumer_thread:
            self.consumer_thread.join(timeout=5)

    def get_latest_data(self):
        """Get all available data from queue"""
        data = []
        while not self.data_queue.empty():
            try:
                data.append(self.data_queue.get_nowait())
            except queue.Empty:
                break
        return data

    def update_threshold(self, new_threshold):
        """Update the anomaly threshold"""
        self.threshold = new_threshold
        print(f"Threshold updated to: {new_threshold}")


# Global detector instance
detector = AnomalyDetectorWithUI()

# Create Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "PRIVATEER - Network Anomaly Detection"

# App layout (similar to demo.py)
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
                            min=0.001,
                            max=0.5,
                            step=0.001,
                            value=detector.threshold,
                            marks={v: f"{v:.3f}" for v in np.linspace(0.001, 0.5, 10)},
                            tooltip={"placement": "bottom", "always_visible": True}
                        )
                    ], className="mb-3"),
                    html.Div([
                        html.Label("🔐 Privacy Protection: ", className="form-label"),
                        dbc.Badge("Kafka Pipeline Active", color="success", className="ms-2"),
                        html.Small(" - Real-time processing", className="text-muted ms-2")
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
        n_intervals=0
    ),

    # Store for detector state
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
    Input('threshold-slider', 'value')
)
def update_threshold(threshold):
    detector.update_threshold(threshold)
    return dbc.Badge(f"Threshold: {threshold:.3f}", color="info")


@app.callback(
    [Output('feature-display', 'figure'),
     Output('anomaly-detection', 'figure'),
     Output('stats-display', 'children'),
     Output('device-list', 'children')],
    [Input('interval-component', 'n_intervals')],
    [State('detector-state', 'data')]
)
def update_graphs(n, state):
    if not state.get('running', False):
        empty_fig = create_empty_figure("Detection Stopped - Click Start to begin")
        return empty_fig, empty_fig, html.P("Start detection to see statistics"), html.P("No devices yet")

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

    return feature_fig, anomaly_fig, stats, device_list


def create_feature_figure():
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
        title="TransformerAD Anomaly Detection (Real-time Kafka Stream)",
        xaxis_title="Time",
        yaxis_title="Reconstruction Error (L1 Loss)",
        hovermode='x unified'
    )

    return fig


def create_statistics():
    if detector.stats['total'] == 0:
        return html.P("No data processed yet")

    # Calculate rates
    tpr = (detector.stats['true_positives'] / detector.stats['anomalies'] * 100
           if detector.stats['anomalies'] > 0 else 0)

    fpr = (detector.stats['false_positives'] / detector.stats['anomalies'] * 100
           if detector.stats['anomalies'] > 0 else 0)

    return dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("📊 Total Samples"),
                    html.H3(f"{detector.stats['total']}", className="text-primary")
                ])
            ])
        ], width=3),

        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("🚨 Detected"),
                    html.H3(f"{detector.stats['anomalies']}", className="text-danger")
                ])
            ])
        ], width=3),

        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("✅ TPR"),
                    html.H3(f"{tpr:.1f}%", className="text-success")
                ])
            ])
        ], width=3),

        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("📤 Alerts Sent"),
                    html.H3(f"{detector.stats['alerts_sent']}", className="text-warning")
                ])
            ])
        ], width=3)
    ])


def create_device_list():
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


# Main
if __name__ == '__main__':
    print("🛡️ PRIVATEER Network Anomaly Detection Service with UI")
    print("=" * 50)
    print(f"🤖 Using TransformerAD Model")
    print(f"📱 Device: {detector.device}")
    print(f"🎯 Threshold: {detector.threshold}")
    print(f"📊 Input Features: {detector.input_features}")
    print(f"📥 Kafka Topic: {detector.input_topic}")
    print(f"🔐 Privacy Protection: Anonymization Active")
    print("=" * 50)
    print("Starting web server...")
    print("Open your browser and go to: http://0.0.0.0:8050")
    print("=" * 50)

    app.run(host='0.0.0.0', port=8050, debug=False)