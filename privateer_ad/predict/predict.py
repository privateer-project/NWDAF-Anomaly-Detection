from typing import Tuple, Optional
import fire
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from sklearn.metrics import classification_report
from pathlib import Path

from privateer_ad import logger
from privateer_ad.config import (get_paths,
                                 get_model_config,
                                 get_training_config
                                 )
from privateer_ad.architectures import TransformerAD, TransformerADConfig
from privateer_ad.etl import DataProcessor


def make_predictions(
        model_path: str,
        data_path: str,
        threshold: float,
        batch_size: Optional[int] = None,
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
    paths_config = get_paths()
    model_config = get_model_config()
    training_config = get_training_config()

    logger.info(f'Starting predictions for model: {model_path}')
    logger.info(f'Making predictions on dataset: {data_path}')

    # Setup device
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device(device)
    logger.info(f'Using device: {device}')

    # Setup data processing
    logger.info('Setting up data processing...')
    dp = DataProcessor(partition=False)

    # Use provided parameters or fall back to config
    pred_batch_size = batch_size or training_config.batch_size
    pred_seq_len = seq_len or model_config.seq_len

    # Get dataloader
    try:
        dl = dp.get_dataloader(
            data_path,
            batch_size=pred_batch_size,
            seq_len=pred_seq_len,
            only_benign=False  # Always include both classes for prediction
        )
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

    # Setup model configuration
    transformer_config = TransformerADConfig(
        seq_len=pred_seq_len,
        input_size=input_size,
        num_layers=model_config.num_layers,
        hidden_dim=model_config.hidden_dim,
        latent_dim=model_config.latent_dim,
        num_heads=model_config.num_heads,
        dropout=model_config.dropout
    )

    # Create model
    logger.info('Creating model...')
    model = TransformerAD(transformer_config)

    # Load model weights
    logger.info(f'Loading model from: {model_path}')
    model_state_dict = _load_model_weights(model_path, paths_config)

    try:
        model.load_state_dict(model_state_dict)
        logger.info('Model weights loaded successfully')
    except Exception as e:
        logger.error(f'Failed to load model weights: {e}')
        raise

    # Move model to device and set to evaluation mode
    model = model.to(device)
    model.eval()

    # Setup loss function
    criterion_fn = getattr(nn, training_config.loss_function)(reduction='none')

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


def _load_model_weights(model_path: str, paths_config) -> dict:
    """
    Load model weights from various path formats.

    Args:
        model_path: Path to model file or experiment directory
        paths_config: Paths configuration

    Returns:
        Model state dictionary
    """
    model_path = Path(model_path)

    # Try different path resolutions
    possible_paths = [
        model_path,  # Direct path
        paths_config.experiments_dir / model_path / 'model.pt',  # Experiment directory
        paths_config.models_dir / model_path,  # Models directory
        Path(model_path).with_suffix('.pt'),  # Add .pt extension
    ]

    for path in possible_paths:
        if path.exists():
            logger.info(f'Loading model from: {path}')
            try:
                state_dict = torch.load(path, map_location='cpu')

                # Handle different state dict formats
                if isinstance(state_dict, dict):
                    # Remove common prefixes from distributed training
                    cleaned_state_dict = {}
                    for key, value in state_dict.items():
                        # Remove common prefixes
                        clean_key = key
                        for prefix in ['_module.', 'module.']:
                            if clean_key.startswith(prefix):
                                clean_key = clean_key[len(prefix):]
                        cleaned_state_dict[clean_key] = value

                    return cleaned_state_dict
                else:
                    raise ValueError(f'Unexpected state dict format: {type(state_dict)}')

            except Exception as e:
                logger.warning(f'Failed to load from {path}: {e}')
                continue

    raise FileNotFoundError(f'Could not find model file at any of: {[str(p) for p in possible_paths]}')


def main():
    """Main function with Fire integration for CLI usage."""
    fire.Fire(make_predictions)


if __name__ == '__main__':
    # Example usage - you can remove this or update paths as needed
    try:
        make_predictions(
            model_path='20250313-114045/model.pt',
            data_path='test',  # Use 'test' instead of raw dataset path
            threshold=0.028
        )
    except Exception as e:
        logger.error(f'Prediction failed: {e}')