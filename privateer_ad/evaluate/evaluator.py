from typing import Tuple, List
from tqdm import tqdm

import mlflow
import numpy as np
import torch

from sklearn.metrics import roc_curve, roc_auc_score, classification_report

from privateer_ad.visualizations.plotter import Visualizer


class ModelEvaluator:
    def __init__(self, criterion, device: torch.device):
        self.criterion = getattr(torch.nn, criterion)(reduction='none')
        self.device = device
        self.visualizer = Visualizer()

    def compute_reconstruction_errors(self, model, dataloader) -> Tuple[np.ndarray, np.ndarray]:
        losses: List[float] = []
        labels: List[int] = []
        model.to(self.device)
        model.eval()

        with torch.no_grad():
            for inputs in tqdm(dataloader, desc='Computing reconstruction errors'):
                x = inputs[0]['encoder_cont'].to(self.device)
                targets = np.squeeze(inputs[1][0])
                output = model(x)
                batch_rec_errors = self.criterion(x, output)
                loss_per_sample = batch_rec_errors.mean(dim=(1, 2))
                losses.extend(loss_per_sample.cpu().tolist())
                labels.extend(targets.tolist())
        return np.array(losses), np.array(labels, dtype=int)

    @staticmethod
    def find_optimal_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
        fpr, tpr, thresholds = roc_curve(y_true=y_true, y_score=y_score)
        optimal_idx = np.argmin(np.sqrt(np.power(fpr, 2) + np.power(1 - tpr, 2)))
        return thresholds[optimal_idx]

    def evaluate(self, model, dataloader, threshold: int = None, prefix='', step=0) -> Tuple[dict[str, float], dict]:
        rec_errors, y_true = self.compute_reconstruction_errors(model, dataloader)
        if threshold is None:
            threshold = self.find_optimal_threshold(y_true, rec_errors)
        y_pred = (rec_errors >= threshold).astype(int)

        target_names = ['benign', 'malicious']
        labels = [0, 1]

        metrics = {'roc_auc': roc_auc_score(y_true=y_true, y_score=rec_errors),
                   'loss': np.mean(rec_errors)}

        metrics.update(classification_report(y_true=y_true,
                                             y_pred=y_pred,
                                             labels=labels,
                                             target_names=target_names,
                                             output_dict=True)['macro avg'])
        metrics = {f'{prefix}_' + k: v for k, v in metrics.items()}
        metrics['threshold'] = threshold
        self.visualizer.visualize(y_true=y_true,
                                  y_pred=y_pred,
                                  scores=rec_errors,
                                  threshold=threshold,
                                  target_names=target_names,
                                  prefix=prefix
                                  )

        if mlflow.active_run():
            mlflow.log_metrics(metrics, step=step)
            for name, fig in self.visualizer.figures.items():
                mlflow.log_figure(fig, f'{name}_{step}.png')
        return metrics, self.visualizer.figures
