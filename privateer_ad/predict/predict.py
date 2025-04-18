from typing import List, Tuple

import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from fire import Fire
from sklearn.metrics import classification_report
from privateer_ad.config import AttentionAutoencoderConfig, PathsConf, HParams
from privateer_ad.models import AttentionAutoencoder
from privateer_ad.data_utils.transform import DataProcessor
from pathlib import Path

def make_predictions(
        model_path,
        data_path,
        threshold: float = 0.026970019564032555
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

    model_path = Path(model_path)
    if paths.experiments_dir.joinpath(model_path).exists():
        model_path = paths.experiments_dir.joinpath(model_path)

    # Load Data
    dp = DataProcessor()
    dl = dp.get_dataloader(data_path,
                           use_pca=hparams.use_pca,
                           batch_size=hparams.batch_size,
                           seq_len=hparams.seq_len,
                           only_benign=False)

    # Load model
    model = AttentionAutoencoder(config=AttentionAutoencoderConfig())
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    state_dict = {key.removeprefix('_module.'): value for key, value in state_dict.items()}
    model.load_state_dict(state_dict)
    model = model.to(device)

    model.to(device)
    inputs: List[float] = []
    losses: List[float] = []
    predictions: List[int] = []
    labels: List[int] = []
    criterion_fn = getattr(nn, hparams.loss)(reduction='none')
    model.eval()

    with torch.no_grad(): # Collect results
        for batch in tqdm(dl, desc="Computing reconstruction errors"):
            x = batch[0]['encoder_cont'].to(device)
            targets = np.squeeze(batch[1][0])
            labels.extend(targets.cpu().tolist())

            output = model(x)
            batch_rec_errors = criterion_fn(x, output)

            loss_per_sample = batch_rec_errors.mean(dim=(1, 2))
            losses.extend(loss_per_sample.cpu().tolist())

            inputs.extend(x.cpu().tolist())
            predictions.extend((loss_per_sample > threshold).cpu().tolist())

    inputs_np = np.asarray(inputs)
    losses_np = np.array(losses)
    predictions_np = np.asarray(predictions)
    labels_np = np.array(labels, dtype=int)

    # Keep same n samples for balanced metrics
    ben_labels = labels_np[labels_np == 0]
    ben_predictions = predictions_np[labels_np == 0]

    mal_labels = labels_np[labels_np == 1]
    mal_predictions = predictions_np[labels_np == 1]

    _len = len(mal_labels)
    ben_labels = ben_labels[:_len]
    ben_predictions = ben_predictions[:_len]
    mal_labels = mal_labels[:_len]
    mal_predictions = mal_predictions[:_len]
    labels = np.concatenate([ben_labels, mal_labels])
    predictions = np.concatenate([ben_predictions, mal_predictions])
    print(classification_report(labels, predictions))
    return inputs_np, losses_np, predictions_np, labels_np

def main():
    Fire(make_predictions)
