import logging
from typing import Tuple, List

import mlflow
import numpy as np
import torch

from tqdm import tqdm
from sklearn.metrics import roc_curve, roc_auc_score, classification_report

from privateer_ad.config import TrainingConfig
from privateer_ad.visualizations import Visualizer


class ModelEvaluator:
    def __init__(self, device: torch.device, loss_fn: str=None):
        logging.info('Instantiate ModelEvaluator...')

        self.device = device
        self.loss_fn = loss_fn or TrainingConfig().loss_fn_name

        self.visualizer = Visualizer()

    def compute_anomaly_scores(self, model, dataloader) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        x: List[float] = []
        y_true: List[int] = []
        y_score: List[float] = []

        loss_fn = getattr(torch.nn, self.loss_fn)(reduction='none')
        model.to(self.device)
        model.eval()

        with torch.no_grad():
            progress_bar = tqdm(dataloader, desc='Computing reconstruction errors')

            for inputs in progress_bar:
                batch_input = inputs[0]['encoder_cont'].to(self.device)
                batch_y_true = np.squeeze(inputs[1][0])
                batch_output = model(batch_input)

                batch_score = loss_fn(batch_input, batch_output)
                batch_y_score_per_sample = batch_score.mean(dim=(1, 2))

                x.extend(batch_input.tolist())
                y_true.extend(batch_y_true.tolist())
                y_score.extend(batch_y_score_per_sample.cpu().tolist())

        return np.array(x, dtype=np.float32), np.array(y_true, dtype=np.int32), np.array(y_score, dtype=np.float32)

    @staticmethod
    def find_optimal_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
        fpr, tpr, thresholds = roc_curve(y_true=y_true, y_score=y_score)
        optimal_idx = np.argmin(np.sqrt(np.power(fpr, 2) + np.power(1 - tpr, 2)))
        return thresholds[optimal_idx]

    def evaluate(self, model, dataloader, threshold: int = None, prefix='', step=0) -> Tuple[dict[str, float], dict]:
        x, y_true, anomaly_scores = self.compute_anomaly_scores(model, dataloader)
        if threshold is None:
            threshold = self.find_optimal_threshold(y_true, anomaly_scores)

        y_pred = (anomaly_scores >= threshold).astype(int)

        target_names = ['Benign', 'Malicious']
        metrics = {'roc_auc': float(roc_auc_score(y_true=y_true, y_score=anomaly_scores)),
                   'loss': float(np.mean(anomaly_scores))}

        metrics.update(classification_report(y_true=y_true,
                                             y_pred=y_pred,
                                             target_names=target_names,
                                             output_dict=True)['macro avg'])
        metrics = {f'{prefix}_' + k: v for k, v in metrics.items()}
        metrics[f'{prefix}_threshold'] = float(threshold)
        self.visualizer.visualize(y_true=y_true,
                                  y_pred=y_pred,
                                  scores=anomaly_scores,
                                  threshold=threshold,
                                  target_names=target_names,
                                  prefix=prefix
                                  )

        if mlflow.active_run():
            mlflow.log_text(classification_report(y_true=y_true,
                                                  y_pred=y_pred,
                                                  target_names=target_names),
                            f'{str(step).zfill(3)}_{prefix}_classification_report.txt')
            mlflow.log_metrics(metrics, step=step)
            for name, fig in self.visualizer.figures.items():
                mlflow.log_figure(fig, f'{str(step).zfill(3)}_{name}.png')
        metrics_logs = '\n'.join([f'{key}: {value}' for key, value in metrics.items()])
        logging.info(f'Test metrics:\n{metrics_logs}')
        return metrics, self.visualizer.figures
