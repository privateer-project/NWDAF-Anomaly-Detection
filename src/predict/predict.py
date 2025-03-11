from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from tqdm import tqdm

from src.config import AttentionAutoencoderConfig, MetaData, PathsConf
from src.models import AttentionAutoencoder
from src.data_utils.transform import DataProcessor
from src.data_utils.load import NWDAFDataloader


def make_predictions(
        model_path: Path,
        data_path: Path,
        criterion: str,
        batch_size: int,
        seq_len: int,
        threshold: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load a trained model and make predictions on the specified dataset.
    Parameters:
    -----------
    model_path : str
        Path to the saved model state dictionary.
    data_path : str
        Path to the CSV file containing the data to evaluate.
    criterion : str
        Name of the PyTorch loss function to use for reconstruction error calculation.
    batch_size : int
        Batch size for data loading.
    seq_len : int
        Sequence length for time series processing.
    Returns:
    --------
    Tuple[np.ndarray, np.ndarray, np.ndarray]:
        - inputs: Array of shape (n_samples, seq_len, n_features) containing input data
        - losses: Array of shape (n_samples,) containing reconstruction errors
        - labels: Array of shape (n_samples,) containing integer labels
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dp = DataProcessor(metadata=MetaData(), paths=PathsConf())
    nwdaf_dl = NWDAFDataloader(batch_size=batch_size, seq_len=seq_len)
    model = AttentionAutoencoder(config=AttentionAutoencoderConfig())

    # Prepare data
    df = pd.read_csv(data_path)
    print([True if 'pca' in col else False for col in df.columns])
    if not any([True if 'pca' in col else False for col in df.columns]):
        df = dp.process_data(df)
    ds = nwdaf_dl.get_ts_dataset(df)
    dl = ds.to_dataloader(train=False)

    # Prepare model
    model_state_dict = torch.load(model_path)
    model.load_state_dict(model_state_dict)
    model.to(device)
    inputs: List[float] = []
    losses: List[float] = []
    predictions: List[int] = []
    labels: List[int] = []

    model.eval()

    # Set loss function
    evaluation_criterion = getattr(nn, criterion)(reduction='none')

    with torch.no_grad(): # Collect results
        for batch in tqdm(dl, desc="Computing reconstruction errors"):
            x = batch[0]['encoder_cont'].to(device)
            targets = np.squeeze(batch[1][0])
            labels.extend(targets.cpu().tolist())

            output = model(x)
            batch_rec_errors = evaluation_criterion(x, output)

            loss_per_sample = batch_rec_errors.mean(dim=(1, 2))
            losses.extend(loss_per_sample.cpu().tolist())

            inputs.extend(x.cpu().tolist())
            predictions.extend((loss_per_sample > threshold).cpu().tolist())

    return np.asarray(inputs), np.array(losses), np.asarray(predictions), np.array(labels, dtype=int)

if __name__ == '__main__':
    paths = PathsConf()

    model_path = paths.experiments_dir.joinpath('20250310-143759').joinpath('model.pt')
    data_path = paths.raw_dataset
    batch_size = 32
    seq_len = 15
    criterion = 'L1Loss'
    threshold = 0.8252406716346741

    inputs, losses, predictions, labels = make_predictions(
        model_path=model_path,
        data_path=data_path,
        batch_size=batch_size,
        seq_len=seq_len,
        criterion=criterion,
        threshold=threshold,
    )
