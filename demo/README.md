# PRIVATEER-AD: Privacy-Preserving Anomaly Detection for 6G Networks

![PRIVATEER Logo](https://img.shields.io/badge/PRIVATEER-6G%20Security-blue) ![Python](https://img.shields.io/badge/python-3.12+-blue.svg) ![PyTorch](https://img.shields.io/badge/PyTorch-2.7.0-orange) ![License](https://img.shields.io/badge/license-Apache%202.0-green)

PRIVATEER-AD is a federated learning framework for privacy-preserving anomaly detection in 6G networks, featuring differential privacy, transformer-based models, and real-time visualization.

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Docker & Docker Compose
- CUDA (optional, for GPU acceleration)
- 8GB+ RAM recommended

### Installation

#### 1. Clone Repository
```bash
git clone <repository-url>
cd privateer-ad
```

#### 2. Install CSPRNG (Cryptographically Secure Pseudo-Random Number Generator)
```bash
# Install PyTorch CSPRNG for enhanced privacy
git clone https://github.com/pytorch/csprng.git
cd csprng
python setup.py install
cd ..
```

#### 3. Install PRIVATEER-AD
```bash
# Install in development mode
pip install -e .
```

#### 4. Download Dataset
```bash
# Download and prepare the 5G DDoS dataset
python -c "from privateer_ad.etl import Downloader; Downloader().download_extract()"
python -c "from privateer_ad.etl import DataProcessor; DataProcessor().prepare_datasets()"
```

## 🔧 Basic Usage

### Training a Model

#### Single Machine Training
```bash
# Train TransformerAD with Differential Privacy
train-ad train-eval
```
### Train with custom parameters
Pass environment variables with prefix as defined in settings.

For example:
```bash
PRIVATEER_DP_DP_ENABLED=false train-ad train-eval
```

#### Hyperparameter Tuning
```bash
# Auto-tune model hyperparameters
autotune-ad
```

### Federated Learning Simulation

#### Start MLflow Server
```bash
mlflow server --host 0.0.0.0 --port 5001 --backend-store-uri file:///mlruns
```

#### Run Federated Simulation
```bash
# 3 clients, 80 rounds, with secure aggregation
flwr run . --run-config n-clients=3,num-server-rounds=80,epochs=1
```

#### Custom Federation Config
```bash
# Edit pyproject.toml for different settings
[tool.flwr.app.config]
n-clients = 5
num-server-rounds = 100
epochs = 2
```

### Model Prediction
```bash
# Make predictions on test set
predict-ad --model_name TransformerAD_DP --data_path test
```

## 🎮 Real-Time Demo

### Docker Compose Demo (Recommended)

#### 1. Start All Services
```bash
cd demo
docker-compose up -d
```

#### 2. Access Services
- **Anomaly Detection UI**: http://localhost:8050
- **MLflow Tracking**: http://localhost:5050
- **Kafka**: localhost:9092

#### 3. Monitor Services
```bash
# View logs
docker-compose logs -f anomaly-detector
docker-compose logs -f data-producer
docker-compose logs -f anonymizer

# Check service health
docker-compose ps
```

### Manual Demo Setup

#### 1. Start MLflow
```bash
mlflow server --host 0.0.0.0 --port 5001
```

#### 2. Run Standalone Demo
```bash
# Train a model first
train-ad

# Run interactive demo with built-in data simulation
python demo.py
```

#### 3. Access Demo UI
- Open browser: http://127.0.0.1:8050
- Click "Start Simulation" to begin real-time detection

## 📊 Model Architecture

PRIVATEER-AD uses a **TransformerAD** architecture with:
- **Attention mechanisms** for temporal pattern recognition
- **Differential Privacy** (ε=0.3, δ=1e-8) via Opacus
- **Federated Learning** with SecAgg+ secure aggregation
- **Sequence length**: 12 timesteps
- **Input features**: 9 network metrics (bitrates, retransmissions, etc.)

## 🔒 Privacy Features

- **Differential Privacy**: Formal privacy guarantees during training
- **Secure Aggregation**: Encrypted gradient sharing in federated learning
- **Data Anonymization**: Device ID hashing and IP masking
- **Temporal Obfuscation**: Privacy-preserving time series processing

## 📈 Configuration

### Environment Variables
```bash
# Core settings
export PRIVATEER_MLFLOW_TRACKING_URI="http://localhost:5001"
export PRIVATEER_DATA_BATCH_SIZE=4096
export PRIVATEER_DATA_SEQ_LEN=12

# Privacy settings
export PRIVATEER_DP_TARGET_EPSILON=0.3
export PRIVATEER_DP_TARGET_DELTA=1e-8
export PRIVATEER_DP_MAX_GRAD_NORM=0.7

# Federated learning
export PRIVATEER_FL_NUM_ROUNDS=100
export PRIVATEER_FL_SECURE_AGGREGATION_ENABLED=true
```

### Configuration Files
- `privateer_ad/config/settings.py`: Main configuration
- `pyproject.toml`: Federated learning settings
- `demo/docker-compose.yml`: Demo environment setup

## 🛠️ Development

### Project Structure
```
privateer_ad/
├── architectures/          # Model definitions (TransformerAD)
├── config/                 # Configuration management
├── etl/                    # Data processing and loading
├── fl/                     # Federated learning components
├── train/                  # Training pipelines and optimization
├── evaluate/               # Model evaluation and metrics
├── predict/                # Inference utilities
├── visualizations/         # Plotting and dashboard components
└── utils.py               # Utility functions

demo/                      # Real-time demonstration
├── docker-compose.yml     # Multi-service demo setup
├── detector.py           # Anomaly detection service with UI
├── producer.py           # Data streaming simulation
└── anonymizer.py         # Privacy protection service
```

### Adding New Models
```python
# 1. Create model in privateer_ad/architectures/
class YourModel(nn.Module):
    def __init__(self, config):
        # Implementation

# 2. Register in __init__.py
from .your_model import YourModel
__all__ = ['TransformerAD', 'YourModel']

# 3. Update config
class ModelConfig(BaseSettings):
    model_type: str = Field(default='YourModel')
```

## 🔧 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   export PRIVATEER_DATA_BATCH_SIZE=1024
   export PRIVATEER_DATA_NUM_WORKERS=0
   ```

2. **MLflow Connection Error**
   ```bash
   # Check MLflow server is running
   curl http://localhost:5001/health
   ```

3. **Dataset Download Issues**
   ```bash
   # Manual download
   wget https://zenodo.org/api/records/13900057/files-archive -O data.zip
   unzip data.zip -d data/raw/
   ```

4. **Docker Permission Issues**
   ```bash
   sudo usermod -aG docker $USER
   newgrp docker
   ```

## 📚 FAQ

### 1. **What makes PRIVATEER-AD different from other anomaly detection tools?**
PRIVATEER-AD combines federated learning with differential privacy, enabling privacy-preserving anomaly detection across distributed 6G networks without exposing sensitive network data.

### 2. **How does the differential privacy mechanism work?**
We use Opacus framework with ε=0.3 and δ=1e-8 parameters, adding calibrated noise to gradients during training to prevent individual data point inference while maintaining model utility.

### 3. **What is the TransformerAD architecture?**
TransformerAD uses multi-head attention mechanisms to capture temporal dependencies in network traffic sequences, with positional encoding for time-aware anomaly detection in 12-timestep windows.

### 4. **How does federated learning work in PRIVATEER?**
Multiple edge nodes train local models on their data, then securely aggregate gradients using SecAgg+ protocol without sharing raw data, enabling collaborative learning while preserving privacy.

### 5. **What types of attacks can PRIVATEER detect?**
The system detects DDoS variants (SYN Flood, ICMP Flood, UDP Fragmentation, DNS Flood, GTP-U Flood) and network anomalies in real-time 5G/6G environments.

### 6. **How do you handle data imbalance in network traffic?**
We use balanced sampling during validation, ROC-curve optimization for threshold selection, and weighted metrics to handle the typical 96% benign / 4% malicious traffic ratio.

### 7. **What is the computational overhead of privacy mechanisms?**
Differential privacy adds ~10-15% training overhead, while federated learning reduces individual node computation by distributing the workload across participants.

### 8. **How does the real-time demo work?**
The demo simulates 5G network traffic via Kafka streams, applies real-time anonymization, performs sliding-window anomaly detection, and visualizes results through a Dash web interface.

### 9. **What privacy guarantees does the system provide?**
Formal differential privacy guarantees (ε-δ privacy), secure multi-party computation for aggregation, and data anonymization ensure individual device privacy while enabling collective threat detection.

### 10. **How can PRIVATEER be deployed in production 6G networks?**
The framework supports containerized deployment, edge computing integration, real-time streaming architectures, and standard 3GPP NWDAF interfaces for seamless 6G network integration.

## 📄 License

Apache License 2.0 - See LICENSE file for details.

## 🏆 Acknowledgments

This work is supported by the European Union's Horizon Europe research and innovation programme under grant agreement No 101096110 (PRIVATEER project).