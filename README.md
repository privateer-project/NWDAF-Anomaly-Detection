# Project Setup and Model Training

## 1. Clone the Repository

```bash
git clone https://github.com/privateer-project/NWDAF-Anomaly-Detection/
cd NWDAF-Anomaly-Detection
git checkout dev
```

## 2. Create a Virtual Environment

Ensure you have **Python >= 3.9, < 3.12** installed.

```bash
python -m venv venv
source venv/bin/activate  # For Linux/Mac
# or
venv\Scripts\activate  # For Windows
```

## 3. Install Requirements

Install all dependencies from [pyproject.toml](pyproject.toml):

```bash
pip install -e .
```

## 4. Run the MLflow server on port **5001**:

```bash
mlflow server --backend-store-uri sqlite:///mlflow.db \
              --default-artifact-root ./mlruns \
              --host 0.0.0.0 \
              --port 5001
```

## 5. Configure Environment Variables

In the [config](https://github.com/privateer-project/NWDAF-Anomaly-Detection/tree/dev/privateer_ad/config) folder, create a `.env` file. Copy the content from `.env.example` (which should live in the same directory as your configuration files). Then update the `ROOT_DIR` variable to point to the **absolute path** of the project's [root directory](.).

Here is an example of the variables to include:

```bash
ROOT_DIR=<path>/<to>/<project>/<root>/<directory>
ANALYSIS_DIR=${ROOT_DIR}/analysis_results
SCALERS_DIR=${ROOT_DIR}/scalers
MODELS_DIR=${ROOT_DIR}/models
DATA_DIR=${ROOT_DIR}/data
RAW_DIR=${DATA_DIR}/raw
RAW_DATASET=${RAW_DIR}/amari_ue_data_merged_with_attack_number.csv
DATA_URL=https://zenodo.org/api/records/13900057/files-archive
EXPERIMENTS_DIR=${ROOT_DIR}/experiments
PROCESSED_DIR=${DATA_DIR}/processed
FLWR_SERVER_ADDRESS=[::]:8081
MLFLOW_EXPERIMENT_NAME=privateer_ad
MLFLOW_SERVER_ADDRESS=http://localhost:5001
PYTHONPATH=${ROOT_DIR}:${PYTHONPATH}
```

## 6. (Optional) Training on GPU

If you plan to train the model on a GPU:

1. Verify you have the appropriate CUDA version installed.
2. Ensure that your PyTorch version is compatible with your CUDA version.
3. Install the GPU-enabled dependencies (e.g., `torch` with CUDA support).

For example, if you are using CUDA 11.7:

```bash
pip install torch==2.0.0+cu117 torchvision==0.15.1+cu117 torchaudio==2.0.0 --extra-index-url https://download.pytorch.org/whl/cu117
```

Additional CUDA versions can be found on the [official PyTorch website](https://pytorch.org).

## 7. Data Extraction and Transformation

From the project's [root directory](.), run:

```bash
PYTHONPATH=./ python privateer_ad/data_utils/download.py
```

Followed by:

```bash
PYTHONPATH=./ python privateer_ad/data_utils/transform.py
```

These commands will create a [data](./data) folder in the project root with [data/raw](./data/raw) and [data/processed](./data/processed) subfolders.

## 8. Train the Model

Finally, execute the training script from the [root directory](.):

```bash
PYTHONPATH=./ python privateer_ad/train/train.py
```

MLflow will track experiments and store artifacts in the [mlruns](./mlruns) directory by default (or any path you configured).
