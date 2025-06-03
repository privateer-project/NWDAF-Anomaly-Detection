import logging

from typing import Tuple, Optional

import fire
import numpy as np
import torch

from torch import nn
from tqdm import tqdm
from sklearn.metrics import classification_report

from privateer_ad.config import TrainingConfig, MLFlowConfig, DataConfig

from privateer_ad.etl import DataProcessor
from privateer_ad.utils import load_champion_model


def make_predictions(
        model_name: str ='TransformerAD_DP',
        data_path: str = 'test',
        seq_len: Optional[int] = None,
        device: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load a trained model and make predictions on the specified dataset.

    Parameters:
    -----------
    model_name : str Model name to load from MLFlow (default: 'TransformerAD_DP')
    data_path : str
        Path to the CSV file containing the data to evaluate or 'train', 'val', 'test' for preprocessed datasets.
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
    mlflow_conf = MLFlowConfig()
    training_config = TrainingConfig()
    data_config = DataConfig()
    # Use provided parameters or fall back to config
    data_config.seq_len = seq_len or data_config.seq_len


    # Setup device
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device(device)
    logging.info(f'Using device: {device}')

    # Setup data processing
    logging.info('Setting up data processing...')
    dp = DataProcessor(data_config=data_config)

    try:
        dl = dp.get_dataloader(data_path, only_benign=False, train=False)
        logging.info(f'Data loader created successfully for {data_path}')
    except Exception as e:
        logging.error(f'Failed to create data loader: {e}')
        raise
    # Create model
    logging.info('Loading model...')
    model, threshold, loss_fn = load_champion_model(mlflow_conf.tracking_uri, model_name=model_name)
    print(model)
    logging.info(f'Starting predictions for model: {model_name}')
    logging.info(f'Making predictions on dataset: {data_path}')
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

    logging.info('Making predictions...')
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
    logging.info('Classification Report:')
    logging.info(classification_report(y_true=balanced_labels, y_pred=balanced_predictions))
    return inputs, losses, predictions, labels

def main():
    """Main function with Fire integration for CLI usage."""
    fire.Fire(make_predictions)


if __name__ == '__main__':
    try:
        make_predictions(data_path='test')
    except Exception as e:
        logging.error(f'Prediction failed: {e}')
