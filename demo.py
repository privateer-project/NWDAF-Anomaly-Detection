"""
PRIVATEER Federated Learning Model Integrated Dashboard
"""
import time
import threading
import queue
import logging
import os
import hashlib
from collections import deque

from datetime import datetime, timedelta

import dash
import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objects as go
import requests
import torch
from urllib3.exceptions import InsecureRequestWarning
import urllib3

from dash import dcc, html, Input, Output, State

from privateer_ad.architectures.transformer_ad import TransformerAD
from privateer_ad.etl import DataProcessor
from privateer_ad.config import (
    DataConfig,
    MetadataConfig,
    ModelConfig,
    PathConfig,
    TrainingConfig,
    MLFlowConfig
)
from privateer_ad.utils import load_model_weights, load_mlflow_model_from_run

exported_anomalies = []

SHAP_MAX_SEQ_LEN = int(os.getenv('SHAP_MAX_SEQ_LEN', '12'))

DEFAULT_ANON_EPSILON = float(os.getenv('ANONYMIZER_EPSILON', '0.0'))
ANONYMIZER_SENSITIVITY = float(os.getenv('ANONYMIZER_SENSITIVITY', '0.0'))
EPSILON_MIN = float(os.getenv('ANONYMIZER_EPSILON_MIN', '0.01'))
EPSILON_MAX = float(os.getenv('ANONYMIZER_EPSILON_MAX', '1.0'))
EPSILON_STEP = float(os.getenv('ANONYMIZER_EPSILON_STEP', '0.01'))
SENSITIVE_FEATURES = tuple(os.getenv('ANONYMIZER_SENSITIVE_FEATURES', 'dl_bitrate,ul_bitrate').split(','))
EXPERIMENT_MODEL_ID = os.getenv('PRIVATEER_EXPERIMENT_ID', 'experiments/20250313-181907')

def _env_as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {'1', 'true', 'yes', 'on'}

def _env_as_str(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value is None:
        return default
    value = value.strip()
    if not value:
        return None
    return value

MISP_BASE_URL = os.getenv('MISP_URL') or "https://10.160.3.60"
MISP_API_KEY = os.getenv('MISP_API_KEY') or "QJVfx7A4SwMY7UsgNtzNE47Qtv5D4Qf0cBAPSCdr"
MISP_VERIFY_SSL = _env_as_bool(os.getenv('MISP_VERIFY_SSL'), default=False)
MISP_TIMEOUT = float(os.getenv('MISP_TIMEOUT', '5.0'))
MISP_USERNAME = os.getenv('MISP_USERNAME') or "infili@misp.testing"
MISP_PASSWORD = os.getenv('MISP_PASSWORD') or "]3L2>9bzS'RFM*Z"

MLFLOW_RUN_ID = _env_as_str('PRIVATEER_MLFLOW_RUN_ID')
MLFLOW_RUN_NAME = _env_as_str('PRIVATEER_MLFLOW_RUN_NAME', 'bright-chimp-326')
MLFLOW_ARTIFACT_PATH = _env_as_str('PRIVATEER_MLFLOW_ARTIFACT_PATH', 'TransformerAD')
INFERENCE_DATASET_FILENAME = os.getenv('PRIVATEER_INFERENCE_DATASET', 'test.csv')

urllib3.disable_warnings(InsecureRequestWarning)


XAI_SHAP_BASE_URL = "http://localhost:5000/xai/shap"


def _shap_iframe_src(view: str, serial: int) -> str:
    return f"{XAI_SHAP_BASE_URL}/{view}?refresh={serial}"

SHAP_IFRAME_STYLE = {
    "width": "1400px",
    "height": "1000px",
    "border": "0",
    "transform": "scale(0.6)",
    "transformOrigin": "0 0"
}

SHAP_CONTAINER_STYLE = {
    "width": "840px",
    "height": "600px",
    "margin": "0 auto",
    "overflow": "hidden"
}

FEATURE_DISPLAY_NAMES = {
    'dl_bitrate': 'DL Rate',
    'ul_bitrate': 'UL Rate'
}

FEATURE_COLORS = {
    'dl_bitrate': '#0d6efd',
    'ul_bitrate': '#198754'
}

RAW_TRACE_COLOR = '#6c757d'


def _build_shap_iframe(view: str, serial: int) -> html.Iframe:
    """Construct an iframe pointing to the given SHAP view."""
    return html.Iframe(
        id=f"shap-{view}-frame",
        src=_shap_iframe_src(view, serial),
        style=SHAP_IFRAME_STYLE
    )


def _shap_placeholder(message: str = "no anomalies detected") -> html.Div:
    """Create a placeholder element shown when SHAP data is unavailable."""
    return html.Div(
        message,
        className="text-center text-muted fw-bold",
        style={
            "display": "flex",
            "alignItems": "center",
            "justifyContent": "center",
            "height": "100%",
            "width": "100%",
            "backgroundColor": "#f8f9fa",
            "border": "1px dashed #ced4da",
            "borderRadius": "8px"
        }
    )


iframe_refresh_serial = 0
REALTIME_INTERVAL_MS = 500  # Base interval for standard dashboard updates (milliseconds)
XAI_INTERVAL_MS = 5000      # XAI refresh cadence (milliseconds)
XAI_WINDOW_SECONDS = 5
THROUGHPUT_WINDOW_SECONDS_DEFAULT = int(os.getenv('THROUGHPUT_WINDOW_SECONDS', '20'))
last_xai_anomaly_timestamp = None


class MISPClient:
    """Lightweight client for pushing anomaly events into MISP."""

    def __init__(self,
                 base_url: str | None = MISP_BASE_URL,
                 api_key: str | None = MISP_API_KEY,
                 verify_ssl: bool = MISP_VERIFY_SSL,
                 timeout: float = MISP_TIMEOUT):
        self.base_url = base_url.rstrip('/') if base_url else None
        self.api_key = api_key
        self.verify_ssl = verify_ssl
        self.timeout = timeout
        self._warned_unconfigured = False

    def _configured(self) -> bool:
        return bool(self.base_url and self.api_key)

    def publish_anomaly(self,
                        *,
                        ip: str,
                        detection_time: datetime,
                        device_id: str,
                        reconstruction_error: float,
                        threshold: float) -> None:
        if not self._configured():
            if not self._warned_unconfigured:
                logging.info("MISP client not configured; skipping event publication.")
                self._warned_unconfigured = True
            return

        event_payload = {
            "Event": {
                "info": f"PRIVATEER anomaly detected for device {device_id}",
                "distribution": 0,
                "threat_level_id": 3,
                "analysis": 0,
                "Attribute": [
                    {
                        "category": "Network activity",
                        "type": "ip-dst",
                        "value": ip,
                        "to_ids": True,
                        "comment": "Device IP observed during anomaly detection."
                    },
                    {
                        "category": "Other",
                        "type": "text",
                        "value": detection_time.isoformat(),
                        "to_ids": False,
                        "comment": "Detection timestamp (UTC)."
                    },
                    {
                        "category": "Other",
                        "type": "text",
                        "value": f"reconstruction_error={reconstruction_error:.6f}",
                        "to_ids": False,
                        "comment": f"Model threshold at detection time: {threshold:.6f}"
                    }
                ]
            }
        }

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": self.api_key
        }

        try:
            response = requests.post(
                f"{self.base_url}/events/add",
                headers=headers,
                json=event_payload,
                timeout=self.timeout,
                verify=self.verify_ssl
            )
            if not response.ok:
                logging.warning(
                    "Failed to publish MISP event (status %s): %s",
                    response.status_code,
                    response.text
                )
        except Exception as exc:
            logging.warning("Unable to publish anomaly to MISP: %s", exc)


class DemoAnonymizer:
    """Apply lightweight anonymization and noise injection for demo samples."""

    def __init__(self,
                 epsilon: float = DEFAULT_ANON_EPSILON,
                 sensitivity: float = ANONYMIZER_SENSITIVITY,
                 sensitive_features: tuple[str, ...] = SENSITIVE_FEATURES):
        self.epsilon = max(float(epsilon), 1e-6)
        self.sensitivity = max(float(sensitivity), 1e-6)
        # Normalize feature names to strip whitespace and keep consistent casing
        self.sensitive_features = tuple(name.strip() for name in sensitive_features if name.strip())
        self._rng = np.random.default_rng()

    def update_epsilon(self, new_epsilon: float) -> None:
        """Update epsilon used by the Laplace mechanism."""
        self.epsilon = max(float(new_epsilon), 1e-6)

    def anonymize(self, device_id: str | None, feature_values: dict[str, float]) -> dict:
        """Hash the device identifier and obfuscate sensitive feature values."""
        #hashed_device = self._hash_device(device_id)
        obfuscated_features: dict[str, float] = {}
        quality_loss: dict[str, float] = {}
        scale = self.sensitivity / self.epsilon

        for feature, value in feature_values.items():
            if feature in self.sensitive_features and value is not None:
                noisy_value = float(value + self._rng.laplace(0.0, scale))
                obfuscated_features[feature] = noisy_value
                quality_loss[feature] = abs(noisy_value - float(value))
            else:
                obfuscated_features[feature] = value

        return {
            #'anonymized_device_id': hashed_device,
            'anonymized_device_id': device_id,
            'feature_values': obfuscated_features,
            'quality_loss': quality_loss
        }

    @staticmethod
    def _hash_device(device_id: str | None) -> str:
        if device_id is None:
            return "anon-0000"
        digest = hashlib.sha256(str(device_id).encode('utf-8')).hexdigest()
        return f"anon-{int(digest[:8], 16) % 10000:04d}"


class PrivateerAnomalyDetector:
    """Core anomaly detection engine using TransformerAD with differential privacy."""

    def __init__(
        self,
        device_override: str | torch.device | None = None,
        model_name: str = 'TransformerAD_DP',
        experiment_id: str | None = None,
        mlflow_run_name: str | None = MLFLOW_RUN_NAME,
        mlflow_run_id: str | None = MLFLOW_RUN_ID,
        mlflow_artifact_path: str | None = MLFLOW_ARTIFACT_PATH,
    ):
        """Initialize detector with specified model and privacy configurations."""
        self.model_name = model_name
        self.experiment_id = experiment_id or EXPERIMENT_MODEL_ID
        self.mlflow_run_name = mlflow_run_name
        self.mlflow_run_id = mlflow_run_id
        self.mlflow_artifact_path = mlflow_artifact_path or 'global_TransformerAD'

        self.data_config = DataConfig()
        self.data_config.num_workers = 0
        self.data_config.pin_memory = False
        self.data_config.batch_size = 1
        self.data_config.prefetch_factor = None
        self.data_config.persistent_workers = False
        self.data_config.seq_len = SHAP_MAX_SEQ_LEN
        self.metadata = MetadataConfig()
        self.input_features = self.metadata.get_input_features()
        self.mlflow_config = MLFlowConfig()
        self.paths_config = PathConfig()
        self.inference_dataset_path = (self.paths_config.processed_dir / INFERENCE_DATASET_FILENAME).as_posix()

        if device_override is not None:
            self.device = torch.device(device_override)
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize DataProcessor with streaming config
        self.data_processor = DataProcessor(self.data_config)
        self.test_ds = self.data_processor.get_dataset(self.inference_dataset_path, only_benign=False)
        self.test_dl = self.data_processor.get_dataloader(self.inference_dataset_path, only_benign=False, train=True)
        self.threshold = 0.1216268390417099  # Default threshold
        self.loss_fn = None
        self.model = None

        if not self._load_model_from_mlflow():
            self._load_local_model()

        logging.info(f"input features: {self.input_features}")

    def _load_model_from_mlflow(self) -> bool:
        """Attempt to pull the model + metadata from MLflow."""
        if not (self.mlflow_run_id or self.mlflow_run_name):
            return False

        tracking_candidates: list[str] = []
        if self.mlflow_config.tracking_uri:
            tracking_candidates.append(self.mlflow_config.tracking_uri)

        local_tracking = f"file://{PathConfig().root_dir / 'mlruns'}"
        if local_tracking not in tracking_candidates:
            tracking_candidates.append(local_tracking)

        for tracking_uri in tracking_candidates:
            try:
                model, threshold, loss_fn, run = load_mlflow_model_from_run(
                    tracking_uri=tracking_uri,
                    run_id=self.mlflow_run_id,
                    run_name=self.mlflow_run_name,
                    artifact_path=self.mlflow_artifact_path,
                    experiment_name=self.mlflow_config.experiment_name,
                )
                self.model = model.to(self.device)
                self.model.eval()
                self.threshold = float(threshold)
                self.loss_fn = loss_fn
                run_label = run.data.tags.get('mlflow.runName') or self.mlflow_run_name or self.mlflow_run_id
                self.model_name = run_label or self.model_name
                self.experiment_id = f"mlflow:{run.info.experiment_id}/{run.info.run_id}"
                logging.info(
                    "Loaded MLflow model '%s' (run_id=%s) from %s with threshold %.6f",
                    run.data.tags.get('mlflow.runName') or self.mlflow_run_name or self.mlflow_run_id,
                    run.info.run_id,
                    tracking_uri,
                    self.threshold,
                )
                return True
            except Exception as exc:
                logging.warning(
                    "Failed to load MLflow run %s (tracking_uri=%s): %s",
                    self.mlflow_run_name or self.mlflow_run_id,
                    tracking_uri,
                    exc
                )
        return False

    def _load_local_model(self) -> None:
        """Fallback to loading weights from the local experiments directory."""
        logging.info(f"Falling back to experiment artifacts at {self.experiment_id}...")
        model_config = ModelConfig(
            model_name=self.model_name,
            input_size=len(self.input_features),
            seq_len=SHAP_MAX_SEQ_LEN
        )
        paths_config = PathConfig()
        state_dict = load_model_weights(self.experiment_id, paths_config)
        self.model = TransformerAD(model_config=model_config)
        load_result = self.model.load_state_dict(state_dict, strict=False)
        if isinstance(load_result, tuple):
            missing_keys, unexpected_keys = load_result
        else:
            missing_keys = getattr(load_result, 'missing_keys', [])
            unexpected_keys = getattr(load_result, 'unexpected_keys', [])
        if missing_keys or unexpected_keys:
            logging.warning(
                "Model state dict mismatch; missing keys: %s, unexpected keys: %s",
                missing_keys,
                unexpected_keys
            )
        self.model.to(self.device)
        self.model.eval()
        loss_fn_name = TrainingConfig().loss_fn_name
        self.loss_fn = getattr(torch.nn, loss_fn_name)(reduction='none')

    def detect_anomaly(self, input_batch):
        """
        Run anomaly detection on input batch.

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
        """Update anomaly detection threshold for real-time adjustment."""
        self.threshold = new_threshold
        logging.info(f"🎯 Threshold updated to: {new_threshold:.6f}")


class NetworkTrafficSimulator:
    """Simulates network traffic using real dataset for demonstration purposes."""
    def __init__(self, detector, anonymizer, device_label: str | None = None):
        """Initialize simulator with anomaly detector and anonymization logic."""
        self.detector = detector
        self.anonymizer = anonymizer
        self.device_label = device_label or ('gpu' if self.detector.device.type == 'cuda' else 'cpu')
        self.data_queue = queue.Queue()
        self.running = False
        self.thread = None
        self.current_sample_index = 0
        self.dataloader_iterator = iter(self.detector.test_dl)
        self._fallback_devices = list(self.detector.metadata.devices.keys()) or ['default']
        self.misp_client = MISPClient()
        self.latency_window = deque(maxlen=1000)

    def reset_iterator(self):
        """Reset dataloader to beginning for continuous simulation."""
        self.dataloader_iterator = iter(self.detector.test_dl)
        self.current_sample_index = 0
        logging.info("🔄 Dataloader iterator reset to beginning")

    def get_next_sample(self):
        """Fetch next sample from dataset, cycling back when exhausted."""
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
        """Begin traffic simulation in separate thread with specified interval."""
        if self.running:
            logging.debug("Simulation already running for %s, skipping start", self.device_label)
            return
        self.running = True
        self.thread = threading.Thread(target=self._simulation_loop, args=(interval,))
        self.thread.daemon = True
        self.thread.start()
        logging.info(f"▶️ Simulation started on {self.device_label} with {interval}s interval")

    def stop_simulation(self):
        """Stop simulation and clean up thread resources."""
        self.running = False
        if self.thread:
            self.thread.join()
        logging.warning("⏸️ Simulation stopped")

    def _simulation_loop(self, interval):
        """Main simulation execution loop running in background thread."""
        while self.running:
            try:
                # Get next sample from dataloader
                sample = self.get_next_sample()

                # Detect anomaly and record latency
                start = time.perf_counter()
                is_anomaly, score, true_label = self.detector.detect_anomaly(sample)
                inference_latency_ms = (time.perf_counter() - start) * 1000
                self.latency_window.append(inference_latency_ms)
                avg_latency = (sum(self.latency_window) / len(self.latency_window)
                               if self.latency_window else inference_latency_ms)
                logging.info(
                    "⏱️ Detection latency: %.2f ms (avg %.2f ms over %d samples)",
                    inference_latency_ms,
                    avg_latency,
                    len(self.latency_window)
                )

                # Create result dictionary
                result = {
                    'timestamp': datetime.now(),
                    'sample_index': self.current_sample_index,
                    'runtime_device': self.device_label,
                    'is_anomaly': is_anomaly,
                    'reconstruction_error': score,
                    'true_label': true_label,
                    'input_tensor': sample[0]['encoder_cont'].cpu().numpy(),
                    'feature_values': {},
                    'shap': None
                }

                # Extract feature values for display
                input_flat = sample[0]['encoder_cont'].squeeze().cpu().numpy()
                if len(input_flat.shape) == 2:  # [seq_len, features]
                    # Take the last timestep for current values
                    current_features = input_flat[-1]
                    for i, feature_name in enumerate(self.detector.input_features):
                        if i < len(current_features):
                            result['feature_values'][feature_name] = float(current_features[i])

                device_id = None
                group_key = None
                if hasattr(self.detector.test_ds, 'group_ids'):
                    for candidate in ('imeisv', 'device_id', 'device'):
                        if candidate in self.detector.test_ds.group_ids:
                            group_key = candidate
                            break

                groups_tensor = sample[0].get("groups") if isinstance(sample[0], dict) else None
                if group_key and groups_tensor is not None:
                    try:
                        device_id = self.detector.test_ds.transform_values(
                            group_key,
                            groups_tensor,
                            inverse=True,
                            group_id=True
                        )
                        if hasattr(device_id, 'item'):
                            device_id = device_id.item()
                    except KeyError:
                        logging.debug("Group key %s not found in dataset transformers", group_key)
                    except Exception as e:
                        logging.debug("Unable to extract device ID using key %s: %s", group_key, e)

                if not device_id:
                    fallback_idx = (self.current_sample_index - 1) % len(self._fallback_devices)
                    device_id = self._fallback_devices[fallback_idx]

                device_id = str(device_id)
                device_info = self.detector.metadata.devices.get(device_id)
                ip = device_info.ip if device_info else device_id

                shap_payload = None

                if is_anomaly:
                    shap_payload = self._calculate_shap(sample[0]['encoder_cont'])
                    info_misp = {
                        'ip': ip,
                        'time': result['timestamp']
                    }
                    print("Anomaly detected, info_misp:", info_misp)
                    exported_anomalies.append(info_misp)
                    self.misp_client.publish_anomaly(
                        ip=ip,
                        detection_time=result['timestamp'],
                        device_id=device_id,
                        reconstruction_error=score,
                        threshold=self.detector.threshold
                    )

                if shap_payload:
                    result['shap'] = shap_payload

                # Apply anonymization logic for identifiers and features
                result['raw_feature_values'] = dict(result['feature_values']) 
                anonymized_payload = self.anonymizer.anonymize(ip, result['feature_values'])
                result.update(anonymized_payload)
                result['raw_device_id'] = device_id
                # Put result in queue
                self.data_queue.put(result)

                time.sleep(interval)

            except Exception as e:
                logging.error(f"❌ Error in simulation loop: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(interval)

    def get_latest_data(self):
        """Retrieve all pending simulation results from queue."""
        data = []
        while not self.data_queue.empty():
            try:
                data.append(self.data_queue.get_nowait())
            except queue.Empty:
                break
        return data

    def _calculate_shap(self, tensor: torch.Tensor):
        """Call XAI backend to obtain SHAP values for the provided tensor."""
        try:
            model_cfg = getattr(self.detector.model, 'model_config', None)
            shap_seq_len = None
            if model_cfg is not None:
                shap_seq_len = getattr(model_cfg, 'seq_len', None)
            if shap_seq_len is None:
                shap_seq_len = tensor.shape[1]

            shap_seq_len = max(1, min(shap_seq_len, tensor.shape[1], SHAP_MAX_SEQ_LEN))
            if tensor.shape[1] > shap_seq_len:
                tensor = tensor[:, :shap_seq_len, :]
            tensor_cpu = tensor.detach().to(torch.float32).cpu()
            payload = {
                'data': tensor_cpu.tolist(),
                'shape': list(tensor_cpu.shape),
                'dtype': str(tensor_cpu.dtype)
            }
            response = requests.post(
                'http://localhost:5000/api/shap/calculate/string_json',
                json=payload,
                timeout=10
            )
            if response.status_code != 200:
                try:
                    detail = response.text
                except Exception:
                    detail = '<no response body>'
                logging.warning(f"SHAP request failed with status {response.status_code}: {detail}")
                return None
            shap_json = response.json()
            contribution = []
            if isinstance(shap_json, dict):
                if isinstance(shap_json.get('both'), dict):
                    contribution = shap_json['both'].get('contribution', [])
                else:
                    contribution = shap_json.get('contribution', [])
            return {
                'shap_values': shap_json.get('shap_values', {}),
                'contribution': contribution,
                'feature_names': shap_json.get('feature_names', []),
            }
        except Exception as e:
            logging.warning(f"Unable to retrieve SHAP values: {e}")
            return None


# Initialize components
logging.info("🔄 Initializing PRIVATEER components...")
detector = PrivateerAnomalyDetector()
anonymizer = DemoAnonymizer()
simulator = NetworkTrafficSimulator(detector, anonymizer)
simulators = [simulator]

# Spin up a parallel CPU simulator when GPU is available so we can benchmark both
cpu_detector = None
cpu_simulator = None
if torch.cuda.is_available() and detector.device.type != 'cpu':
    cpu_detector = PrivateerAnomalyDetector(device_override='cpu')
    cpu_simulator = NetworkTrafficSimulator(cpu_detector, anonymizer, device_label='cpu')
    simulators.append(cpu_simulator)
    logging.info("Enabling dual-device simulation: gpu + cpu")
else:
    logging.info("Running single-device simulation on %s", detector.device.type)

# Storage for real-time data
realtime_data = {
    'timestamp': [],
    'sample_index': [],
    'runtime_device': [],
    'reconstruction_error': [],
    'is_anomaly': [],
    'true_label': [],
    'anonymized_device_id': [],
    'raw_feature_values': {},  
    'feature_values': {},
    'shap_values': [],
    'raw_inputs': []
}

# Initialize feature storage
for feature in detector.input_features:
    realtime_data['feature_values'][feature] = []
    realtime_data['raw_feature_values'][feature] = [] 

max_points = 200  # Keep last 200 points for display
min_threshold = float(np.floor(detector.threshold * .1))
max_threshold = float(np.ceil(detector.threshold * 10.))
step_threshold = 0.0001

# Create Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "PRIVATEER - Federated Learning Model Integrated Dashboard"

# App layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("🛡️ PRIVATEER Federated Learning Model Integrated Dashboard",
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
                    html.H4("Runtime Controls", className="card-title"),
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
                        html.Label("🔏 Privacy ε (Laplace Noise):", className="form-label"),
                        dcc.Slider(
                            id='epsilon-slider',
                            min=EPSILON_MIN,
                            max=EPSILON_MAX,
                            step=EPSILON_STEP,
                            value=anonymizer.epsilon,
                            marks={
                                float(f"{value:.2f}"): f"{value:.2f}"
                                for value in np.linspace(EPSILON_MIN, EPSILON_MAX, 5)
                            },
                            tooltip={"placement": "bottom", "always_visible": True}
                        ),
                        html.Small(f"Current ε: {anonymizer.epsilon:.2f}", id='epsilon-display', className="text-muted")
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
                    html.H4("📊 Feature Influence SHAP (Last anomaly detected)", className="card-title"),
                    html.Div(
                        id="shap-timeseries-container",
                        children=_shap_placeholder(),
                        style=SHAP_CONTAINER_STYLE
                    )
                ])
            ])
        ], width=6),

        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("📉 Real-Time False Positive Rate", className="card-title"),
                    dcc.Graph(id="fpr-trend", style={'height': '600px'})
                ])
            ])
        ], width=6)
    ], className="mb-4"),

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("🪟 Window Contribution to Decision (Last anomaly detected)", className="card-title"),
                    html.Div(
                        id="shap-window-container",
                        children=_shap_placeholder(),
                        style=SHAP_CONTAINER_STYLE
                    )
                ])
            ])
        ], width=6),

        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("📈 Feature Contribution to Decision (Last anomaly detected)", className="card-title"),
                    html.Div(
                        id="shap-features-container",
                        children=_shap_placeholder(),
                        style=SHAP_CONTAINER_STYLE
                    )
                ])
            ])
        ], width=6)
    ], className="mb-4"),

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("⚡ Inference Throughput (CPU vs GPU)", className="card-title"),
                    html.Div([
                        html.Label("Aggregation window (seconds)", className="form-label"),
                        dbc.Input(
                            id='throughput-window-seconds',
                            type='number',
                            min=1,
                            step=1,
                            value=THROUGHPUT_WINDOW_SECONDS_DEFAULT,
                            debounce=True,
                            style={'maxWidth': '200px'}
                        ),
                        html.Small(
                            "Counts how many samples each device processed in the last N seconds.",
                            className="text-muted"
                        )
                    ], className="mb-3"),
                    dcc.Graph(
                        id="throughput-graph",
                        style={'height': '520px', 'width': '520px', 'margin': '0 auto'}
                    )
                ])
            ])
        ], width="auto")
    ], className="mb-4", justify="center"),

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

    # dcc.Interval(
    #     id='interval-component',
    #     interval=3000,  # Deprecated: original three-second refresh
    #     n_intervals=0,
    #     disabled=True
    # ),
    dcc.Interval(
        id='realtime-interval',
        interval=REALTIME_INTERVAL_MS,
        n_intervals=0,
        disabled=True
    ),
    dcc.Interval(
        id='xai-interval',
        interval=XAI_INTERVAL_MS,
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
    """Callback to handle threshold slider changes in real-time."""
    for sim in simulators:
        sim.detector.update_threshold(threshold)
    return dash.no_update


@app.callback(
    [Output('epsilon-display', 'children'),
     Output('simulation-state', 'data', allow_duplicate=True)],
    Input('epsilon-slider', 'value'),
    prevent_initial_call=True
)
def update_epsilon(epsilon):
    """Handle epsilon slider updates for anonymization noise."""
    anonymizer.update_epsilon(epsilon)
    return f"Current ε: {anonymizer.epsilon:.2f}", dash.no_update

def create_status_badge(text, color):
    """Generate status indicator with current simulation state."""
    device_labels = ", ".join(sorted({sim.device_label for sim in simulators}))
    sample_total = sum(sim.current_sample_index for sim in simulators)
    return dbc.Row([
        dbc.Col([
            dbc.Badge(f"Status: {text}", color=color, className="fs-6 me-2"),
            dbc.Badge(f"Model: {detector.model_name} [{detector.experiment_id}]", color="info", className="fs-6"),
            dbc.Badge(f"Devices: {device_labels}", color="secondary", className="fs-6 ms-2"),
            dbc.Badge(f"Samples: {sample_total}", color="secondary", className="fs-6 ms-2"),
            dbc.Badge(f"ε: {anonymizer.epsilon:.3f}", color="warning", className="fs-6 ms-2")
        ])
    ])


def collect_latest_data():
    """Gather pending samples from all active simulators."""
    aggregated = []
    for sim in simulators:
        aggregated.extend(sim.get_latest_data())
    return aggregated

@app.callback(
    [Output('simulation-state', 'data'),
     Output('realtime-interval', 'disabled'),
     Output('xai-interval', 'disabled'),
     Output('status-indicator', 'children')],
    [Input('start-btn', 'n_clicks'),
     Input('stop-btn', 'n_clicks'),
     Input('reset-btn', 'n_clicks')],
    [State('simulation-state', 'data')]
)
def control_simulation(start_clicks, stop_clicks, reset_clicks, state):
    """Handle simulation control buttons and state management."""
    ctx = dash.callback_context
    global last_xai_anomaly_timestamp

    if not ctx.triggered:
        return state, True, True, create_status_badge("Stopped", "danger")

    button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if button_id == 'start-btn' and start_clicks:
        for sim in simulators:
            sim.start_simulation(interval=0.5)
        return {'running': True}, False, False, create_status_badge("Running", "success")

    elif button_id == 'stop-btn' and stop_clicks:
        for sim in simulators:
            sim.stop_simulation()
        last_xai_anomaly_timestamp = None
        return {'running': False}, True, True, create_status_badge("Stopped", "danger")

    elif button_id == 'reset-btn' and reset_clicks:
        for sim in simulators:
            sim.stop_simulation()
        last_xai_anomaly_timestamp = None
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
        for sim in simulators:
            sim.reset_iterator()
        return {'running': False}, True, True, create_status_badge("Reset", "warning")

    return state, True, True, create_status_badge("Stopped", "danger")


@app.callback(
    [Output('feature-display', 'figure'),
     Output('anomaly-detection', 'figure'),
     Output('fpr-trend', 'figure'),
     Output('stats-display', 'children'),
     Output('device-list', 'children'),
     Output('throughput-graph', 'figure')],
    [Input('realtime-interval', 'n_intervals'),
     Input('throughput-window-seconds', 'value')],
    [State('simulation-state', 'data')]
)
def update_graphs(n, throughput_window, state):
    """Main callback for updating all dashboard visualizations."""
    if not state.get('running', False):
        return (
            create_empty_figure("Simulation Stopped"),
            create_empty_figure("Simulation Stopped"),
            create_empty_figure("Simulation Stopped"),
            html.P("Start simulation to see statistics"),
            html.P("No devices detected yet"),
            create_empty_figure("Simulation Stopped")
        )

    # Get new data
    new_data = collect_latest_data()

    # Add new data to realtime storage
    for data_point in new_data:
        realtime_data['timestamp'].append(data_point['timestamp'])
        realtime_data['sample_index'].append(data_point['sample_index'])
        realtime_data['runtime_device'].append(data_point.get('runtime_device', detector.device.type))
        realtime_data['reconstruction_error'].append(data_point['reconstruction_error'])
        realtime_data['is_anomaly'].append(data_point['is_anomaly'])
        realtime_data['true_label'].append(data_point['true_label'])
        realtime_data['anonymized_device_id'].append(data_point['anonymized_device_id'])
        realtime_data['raw_inputs'].append(data_point['input_tensor'])

        # Add feature values (anonymized)
        for feature, value in data_point['feature_values'].items():
            if feature in realtime_data['feature_values']:
                realtime_data['feature_values'][feature].append(value)

        # Add feature values (raw) with fallback reconstruction from input tensor
        raw_fp = data_point.get('raw_feature_values')
        if not raw_fp:
            tensor = data_point.get('input_tensor')
            raw_fp = {}
            if tensor is not None:
                tensor_np = np.squeeze(np.array(tensor))
                if tensor_np.ndim == 2:
                    last_step = tensor_np[-1]
                    for idx, feature in enumerate(detector.input_features):
                        if idx < len(last_step):
                            raw_fp[feature] = float(last_step[idx])
        if raw_fp:
            for feature, value in raw_fp.items():
                if feature in realtime_data['raw_feature_values']:
                    realtime_data['raw_feature_values'][feature].append(value)
        
        if data_point.get('shap'):
            realtime_data['shap_values'].append(data_point['shap'])
        else:
            realtime_data['shap_values'].append(None)

    # Limit data size
    if len(realtime_data['timestamp']) > max_points:
        for key in realtime_data:
            if key == 'feature_values':
                for feature in realtime_data['feature_values']:
                    realtime_data['feature_values'][feature] = realtime_data['feature_values'][feature][-max_points:]
            elif key == 'raw_feature_values':   # <-- ADD THIS ELIF
                for feature in realtime_data['raw_feature_values']:
                    realtime_data['raw_feature_values'][feature] = realtime_data['raw_feature_values'][feature][-max_points:]
            elif key in ('shap_values', 'raw_inputs'):
                realtime_data[key] = realtime_data[key][-max_points:]
            else:
                realtime_data[key] = realtime_data[key][-max_points:]

    feature_fig = create_feature_figure()
    anomaly_fig = create_anomaly_figure()
    fpr_fig = create_fpr_figure()
    stats = create_statistics()
    device_list = create_device_list()
    throughput_fig = create_throughput_figure(throughput_window or THROUGHPUT_WINDOW_SECONDS_DEFAULT)

    return (
        feature_fig,
        anomaly_fig,
        fpr_fig,
        stats,
        device_list,
        throughput_fig
    )


@app.callback(
    [Output('shap-timeseries-container', 'children'),
     Output('shap-window-container', 'children'),
     Output('shap-features-container', 'children')],
    [Input('xai-interval', 'n_intervals')],
    [State('simulation-state', 'data')]
)
def update_xai_sections(n, state):
    """Refresh XAI panels at a slower cadence, focusing on the latest anomalies."""
    global iframe_refresh_serial, last_xai_anomaly_timestamp
    if not state.get('running', False):
        last_xai_anomaly_timestamp = None
        return _shap_placeholder(), _shap_placeholder(), _shap_placeholder()

    cutoff_time = datetime.now() - timedelta(seconds=XAI_WINDOW_SECONDS)
    target_index = None
    fallback_index = None

    # Walk data history from newest to oldest until we find the latest anomaly.
    for idx in range(len(realtime_data['timestamp']) - 1, -1, -1):
        if not realtime_data['is_anomaly'][idx]:
            continue

        sample_time = realtime_data['timestamp'][idx]
        if sample_time >= cutoff_time:
            target_index = idx
            break

        if fallback_index is None:
            fallback_index = idx

    if target_index is None:
        target_index = fallback_index

    if target_index is None:
        last_xai_anomaly_timestamp = None
        return _shap_placeholder(), _shap_placeholder(), _shap_placeholder()

    anomaly_timestamp = realtime_data['timestamp'][target_index]
    shap_payload = realtime_data['shap_values'][target_index]

    # Lazily fetch SHAP data the first time we reference this anomaly.
    if shap_payload is None:
        raw_input = realtime_data['raw_inputs'][target_index]
        try:
            tensor = torch.tensor(raw_input, dtype=torch.float32)
            shap_payload = simulator._calculate_shap(tensor)
            if shap_payload:
                realtime_data['shap_values'][target_index] = shap_payload
            else:
                logging.warning("SHAP backend returned empty payload for anomaly at %s", anomaly_timestamp)
        except Exception as exc:
            logging.warning("Failed to fetch SHAP data for anomaly at %s: %s", anomaly_timestamp, exc)
            shap_payload = None

    if not shap_payload:
        last_xai_anomaly_timestamp = None
        return _shap_placeholder(), _shap_placeholder(), _shap_placeholder()

    if last_xai_anomaly_timestamp != anomaly_timestamp:
        iframe_refresh_serial += 1
        last_xai_anomaly_timestamp = anomaly_timestamp

    iframe_timeseries = _build_shap_iframe("timeseries", iframe_refresh_serial)
    iframe_window = _build_shap_iframe("window", iframe_refresh_serial)
    iframe_features = _build_shap_iframe("features", iframe_refresh_serial)
    return iframe_timeseries, iframe_window, iframe_features


# Add this NEW callback just for updating the sample counter
@app.callback(
    Output('status-indicator', 'children', allow_duplicate=True),
    [Input('realtime-interval', 'n_intervals')],
    [State('simulation-state', 'data')],
    prevent_initial_call=True
)
def update_sample_counter(n, state):
    """Update sample counter display during active simulation."""
    if state.get('running', False):
        return create_status_badge("Running", "success")
    return dash.no_update


def create_empty_figure(title):
    """Generate placeholder figure when no data is available."""
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


def create_throughput_figure(window_seconds):
    """Show how many samples each device processed in the latest aggregation window."""
    if not realtime_data['timestamp']:
        return create_empty_figure("No Data Available")

    try:
        window_seconds = float(window_seconds)
    except (TypeError, ValueError):
        window_seconds = THROUGHPUT_WINDOW_SECONDS_DEFAULT
    if window_seconds <= 0:
        window_seconds = THROUGHPUT_WINDOW_SECONDS_DEFAULT

    latest_ts = realtime_data['timestamp'][-1]
    cutoff = latest_ts - timedelta(seconds=window_seconds)

    counts_by_device: dict[str, int] = {}
    for ts, device_label in zip(realtime_data['timestamp'], realtime_data.get('runtime_device', [])):
        if ts < cutoff:
            continue
        label = (device_label or 'cpu').upper()
        counts_by_device[label] = counts_by_device.get(label, 0) + 1

    if not counts_by_device:
        return create_empty_figure("No Data Available")

    fig = go.Figure()
    labels = sorted(counts_by_device.keys())
    fig.add_trace(go.Bar(
        x=labels,
        y=[counts_by_device[label] for label in labels],
        marker=dict(color=['#7480ff' if lbl == 'CPU' else '#ff8b73' for lbl in labels])
    ))

    fig.update_layout(
        title=f"Samples Processed in the Last {int(window_seconds)}s",
        xaxis_title="Device",
        yaxis_title="Samples in window",
        hovermode='x unified',
        showlegend=False,
        height=520,
        width=520
    )
    return fig


def create_fpr_figure():
    """Plot running false positive rate over time."""
    if not realtime_data['timestamp']:
        return create_empty_figure("No Data Available")

    fp_count = 0
    tn_count = 0
    fpr_values = []
    for predicted, label in zip(realtime_data['is_anomaly'], realtime_data['true_label']):
        if label == 0 and predicted:
            fp_count += 1
        if label == 0 and not predicted:
            tn_count += 1
        denominator = fp_count + tn_count
        fpr_values.append((fp_count / denominator) * 100 if denominator else 0.0)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=realtime_data['timestamp'],
        y=fpr_values,
        mode='lines+markers',
        name='False Positive Rate',
        line=dict(color='purple'),
        marker=dict(size=6, color='purple')
    ))
    if fpr_values:
        fig.add_hline(
            y=fpr_values[-1],
            line_dash="dot",
            line_color="gray",
            annotation_text=f"Current FPR: {fpr_values[-1]:.1f}%",
            annotation_position="top right"
        )
    fig.update_layout(
        title="False Positive Rate Over Time",
        xaxis_title="Time",
        yaxis_title="FPR (%)",
        hovermode='x unified'
    )
    return fig


def create_feature_figure():
    """Build real-time network feature visualization comparing raw vs anonymized."""
    if not realtime_data['timestamp']:
        return create_empty_figure("No Data Available")

    fig = go.Figure()

    # Limit visualization to downlink bitrate only
    feature_names = [f for f in ('dl_bitrate',) if f in realtime_data['feature_values']]

    base_colors = ['#0d6efd', '#20c997', '#fd7e14']

    for i, feature in enumerate(feature_names):
        display_name = FEATURE_DISPLAY_NAMES.get(feature, feature.replace('_', ' ').title())
        feature_color = FEATURE_COLORS.get(feature, base_colors[i % len(base_colors)])

        # ANONYMIZED (drawn first so raw overlay is legible)
        anon_series = realtime_data['feature_values'].get(feature, [])
        if anon_series:
            fig.add_trace(go.Scatter(
                x=realtime_data['timestamp'],
                y=anon_series,
                mode='lines',
                name=f"{display_name} (anonymized)",
                line=dict(color=feature_color, width=3)
            ))

        # RAW
        raw_series = realtime_data['raw_feature_values'].get(feature, [])
        if raw_series:
            fig.add_trace(go.Scatter(
                x=realtime_data['timestamp'],
                y=raw_series,
                mode='lines+markers',
                name=f"{display_name} (raw)",
                line=dict(color=RAW_TRACE_COLOR, dash='dash', width=2),
                marker=dict(symbol='circle-open', size=6, line=dict(width=1.5, color=RAW_TRACE_COLOR)),
                opacity=0.85
            ))

    # Highlight anomalies using the first feature’s anonymized values (as before)
    anomaly_times = [realtime_data['timestamp'][i] for i, anomaly in enumerate(realtime_data['is_anomaly']) if anomaly]
    if anomaly_times and feature_names:
        first_feature = feature_names[0]
        anon_series = realtime_data['feature_values'].get(first_feature, [])
        if anon_series:
            anomaly_values = [anon_series[i]
                              for i, anomaly in enumerate(realtime_data['is_anomaly'])
                              if anomaly and i < len(anon_series)]
            if anomaly_values:
                fig.add_trace(go.Scatter(
                    x=anomaly_times[:len(anomaly_values)],
                    y=anomaly_values,
                    mode='markers',
                    name='Detected Anomalies',
                    marker=dict(color='red', size=10, symbol='x'),
                    showlegend=True
                ))

    primary_label = FEATURE_DISPLAY_NAMES.get(feature_names[0], "Network Feature") if feature_names else "Network Feature"
    fig.update_layout(
        title=f"{primary_label} — Raw vs Anonymized",
        xaxis_title="Time",
        yaxis_title=f"{primary_label} Value",
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.12,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255,255,255,0.8)"
        ),
        margin=dict(t=90)
    )
    return fig


def create_anomaly_figure():
    """Create reconstruction error plot with threshold and ground truth markers."""
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
    """Calculate and display detection performance metrics."""
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
    """Show anonymized device list with anomaly rates."""
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
    logging.info("🛡️ PRIVATEER Federated Learning Model Integrated Dashboard")
    logging.info("=" * 50)
    logging.info("🤖 Using TransformerAD Model with Differential Privacy")
    logging.info(f"📱 Device: {detector.device}")
    logging.info(f"🎯 Initial Threshold: {detector.threshold}")
    logging.info(f"📊 Input Features: {detector.input_features}")
    logging.info(f"📄 Dataset: Loaded via DataProcessor.get_dataloader('test')")
    logging.info("🔐 Privacy Protection: Anonymization Active")
    logging.info("=" * 50)
    logging.info("Starting web server...")
    logging.info("Open your browser and go to: http://127.0.0.1:8056")
    logging.info("=" * 50)

    app.run(host='127.0.0.1', port=8056, debug=True)
