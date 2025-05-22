from typing import Tuple

import fire
import numpy as np
import torch

from torch import nn
from tqdm import tqdm
from sklearn.metrics import classification_report

from privateer_ad import logger
from privateer_ad.config import HParams, PathsConf
from privateer_ad.models import TransformerAD
from privateer_ad.etl.transform import DataProcessor


def make_predictions(model_path: str,
                     data_path: str,
                     threshold: float
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
    Returns:
    --------
    Tuple[np.ndarray, np.ndarray, np.ndarray]:
        - inputs: Array of shape (n_samples, seq_len, n_features) containing input data
        - losses: Array of shape (n_samples,) containing reconstruction errors
        - labels: Array of shape (n_samples,) containing integer labels
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    paths = PathsConf()
    hparams = HParams()

    if paths.experiments_dir.joinpath(model_path).exists():
        model_path = paths.experiments_dir.joinpath(model_path)

    # Load Data
    dp = DataProcessor(partition=False)
    dl = dp.get_dataloader(data_path,
                           batch_size=hparams.batch_size,
                           seq_len=hparams.seq_len,
                           only_benign=False)

    # Load model
    model = TransformerAD()

    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    state_dict = {key.removeprefix('_module.'): value for key, value in state_dict.items()}
    model.load_state_dict(state_dict)
    model = model.to(device)

    model.to(device)
    inputs = []
    losses = []
    predictions = []
    labels = []
    criterion_fn = getattr(nn, hparams.loss)(reduction='none')
    model.eval()

    with torch.no_grad(): # Collect results
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

    inputs = np.asarray(inputs)
    losses = np.array(losses)
    predictions = np.asarray(predictions)
    labels = np.array(labels)

    ben_labels = labels[labels == 0]
    mal_labels = labels[labels == 1]
    _len = len(labels[labels == 1])

    ben_predictions = predictions[labels == 0]
    mal_predictions = predictions[labels == 1]

    # Keep same n samples for balanced metrics
    ben_predictions = ben_predictions[:_len]
    mal_predictions = mal_predictions[:_len]
    labels = np.concatenate([ben_labels[:_len], mal_labels[:_len]])
    predictions = np.concatenate([ben_predictions[:_len], mal_predictions[:_len]])
    logger.info(classification_report(y_true=labels, y_pred=predictions))
    return inputs, losses, predictions, labels

def main():
    fire.Fire(make_predictions)

if __name__ == '__main__':
    paths = PathsConf()
    data_path = paths.raw_dataset
    make_predictions(model_path='20250313-114045/model.pt',
                     data_path=paths.raw_dataset,
                     threshold=0.028)
