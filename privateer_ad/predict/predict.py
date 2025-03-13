from typing import List, Tuple

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from config import AttentionAutoencoderConfig, PathsConf, HParams
from evaluate.evaluator import ModelEvaluator
from models import AttentionAutoencoder
from data_utils.transform import DataProcessor

def make_predictions(
        model,
        dataloader,
        criterion_fn,
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
    model.to(device)
    inputs: List[float] = []
    losses: List[float] = []
    predictions: List[int] = []
    labels: List[int] = []

    model.eval()

    with torch.no_grad(): # Collect results
        for batch in tqdm(dataloader, desc="Computing reconstruction errors"):
            x = batch[0]['encoder_cont'].to(device)
            targets = np.squeeze(batch[1][0])
            labels.extend(targets.cpu().tolist())

            output = model(x)
            batch_rec_errors = criterion_fn(x, output)

            loss_per_sample = batch_rec_errors.mean(dim=(1, 2))
            losses.extend(loss_per_sample.cpu().tolist())

            inputs.extend(x.cpu().tolist())
            predictions.extend((loss_per_sample > threshold).cpu().tolist())

    return np.asarray(inputs), np.array(losses), np.asarray(predictions), np.array(labels, dtype=int)

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    paths = PathsConf()
    dp = DataProcessor()
    hparams = HParams()
    threshold = 0.026970019564032555

    criterion = hparams.loss
    criterion_fn = getattr(nn, criterion)(reduction='none')

    evaluator = ModelEvaluator(criterion=criterion,
                               device=device)

    batch_size = hparams.batch_size
    seq_len = hparams.seq_len
    use_pca = hparams.use_pca
    dl = dp.get_dataloader(paths.raw_dataset,
                           use_pca=use_pca,
                           batch_size=batch_size,
                           seq_len=seq_len,
                           only_benign=False)

    # Load model
    model_path = paths.experiments_dir.joinpath('20250312-180942').joinpath('model.pt')
    trained_model = AttentionAutoencoder(config=AttentionAutoencoderConfig())
    model_state_dict = torch.load(model_path)
    trained_model.load_state_dict(model_state_dict)
    trained_model = trained_model.to(device)

    inputs, losses, predictions, labels = make_predictions(
        model=trained_model,
        dataloader=dl,
        criterion_fn=getattr(nn, criterion)(reduction='none'),
        threshold=threshold,
    )
