from typing import Tuple, Optional

import fire
import numpy as np
import torch

import mlflow
from torch import nn
from tqdm import tqdm
from sklearn.metrics import classification_report

from privateer_ad import logger
from privateer_ad.config import ModelConfig, TrainingConfig, MLFlowConfig

from privateer_ad.etl import DataProcessor

#todo fully integrate with mlflow model loading
def make_predictions(
        mlflow_model_path: str,
        data_path: str,
        threshold: float,
        seq_len: Optional[int] = None,
        device: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load a trained model and make predictions on the specified dataset.

    Parameters:
    -----------
    model_path : str
        Path to the saved model state dictionary or the experiment directory (eg. '20250317-100421').
    data_path : str
        Path to the CSV file containing the data to evaluate or 'train', 'val', 'test' for prepared datasets.
    threshold : float
        Precalculated threshold.
    batch_size : Optional[int]
        Override batch size for prediction (uses config default if None)
    seq_len : Optional[int]
        Override sequence length (uses config default if None)
    device : Optional[str]
        Override device selection ('auto', 'cpu', 'cuda')

    Returns:
    --------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        - inputs: Array of shape (n_samples, seq_len, n_features) containing input data
        - losses: Array of shape (n_samples,) containing reconstruction errors
        - predictions: Array of shape (n_samples,) containing binary predictions
        - labels: Array of shape (n_samples,) containing integer labels
    """

    # Inject configurations
    model_config = ModelConfig()
    training_config = TrainingConfig()

    logger.info(f'Starting predictions for model: {mlflow_model_path}')
    logger.info(f'Making predictions on dataset: {data_path}')

    # Setup device
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device(device)
    logger.info(f'Using device: {device}')

    # Setup data processing
    logger.info('Setting up data processing...')
    dp = DataProcessor()

    # Use provided parameters or fall back to config
    pred_seq_len = seq_len or model_config.seq_len

    # Get dataloader
    try:
        dl = dp.get_dataloader(data_path, only_benign=False)   # Always include both classes for prediction
        logger.info(f'Data loader created successfully for {data_path}')
    except Exception as e:
        logger.error(f'Failed to create data loader: {e}')
        raise

    # Get sample for model configuration
    sample_batch = next(iter(dl))
    sample_input = sample_batch[0]['encoder_cont'][:1]
    input_size = sample_input.shape[-1]

    logger.info(f'Detected input size: {input_size}')
    logger.info(f'Using sequence length: {pred_seq_len}')

    # Create model
    logger.info('Creating model...')
    mlflow_conf = MLFlowConfig()
    mlflow.set_tracking_uri(mlflow_conf.server_address)

    model = mlflow.pytorch.load_model(mlflow_model_path)
    print(model)

    # Move model to device and set to evaluation mode
    model = model.to(device)
    model.eval()

    # Setup loss function
    criterion_fn = getattr(nn, training_config.loss_fn)(reduction='none')

    # Collect predictions
    inputs = []
    losses = []
    predictions = []
    labels = []

    logger.info('Making predictions...')
    with torch.no_grad():
        for batch in tqdm(dl, desc='Computing reconstruction errors'):
            x = batch[0]['encoder_cont'].to(device)
            targets = np.squeeze(batch[1][0])
            labels.extend(targets.cpu().tolist())

            output = model(x)
            batch_rec_errors = criterion_fn(x, output)

            loss_per_sample = batch_rec_errors.mean(dim=(1, 2))
            losses.extend(loss_per_sample.cpu().tolist())

            inputs.extend(x.cpu().tolist())
            predictions.extend((loss_per_sample > threshold).cpu().tolist())

    # Convert to numpy arrays
    inputs = np.asarray(inputs)
    losses = np.array(losses)
    predictions = np.asarray(predictions)
    labels = np.array(labels)

    # Balance the dataset for fair metrics calculation
    ben_labels = labels[labels == 0]
    mal_labels = labels[labels == 1]
    _len = len(labels[labels == 1])

    ben_predictions = predictions[labels == 0]
    mal_predictions = predictions[labels == 1]

    # Keep same n samples for balanced metrics
    ben_predictions = ben_predictions[:_len]
    mal_predictions = mal_predictions[:_len]
    balanced_labels = np.concatenate([ben_labels[:_len], mal_labels[:_len]])
    balanced_predictions = np.concatenate([ben_predictions[:_len], mal_predictions[:_len]])

    # Log classification report
    logger.info('Classification Report:')
    logger.info(classification_report(y_true=balanced_labels, y_pred=balanced_predictions))

    logger.info('Predictions completed successfully')
    return inputs, losses, predictions, labels

def main():
    """Main function with Fire integration for CLI usage."""
    fire.Fire(make_predictions)


if __name__ == '__main__':
    mlflow_model_path = 'mlflow-artifacts:/304908286791224575/177683639fde4f9b8baa6c4b4a8cfffe/artifacts/model'
    try:
        make_predictions(
            mlflow_model_path=mlflow_model_path, data_path='test', threshold=0.028)
    except Exception as e:
        logger.error(f'Prediction failed: {e}')
