from typing import Dict, Tuple, List

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import roc_curve, accuracy_score, precision_score, roc_auc_score, recall_score, f1_score

from src.visualization.plotter import Visualizer


class ModelEvaluator:
    def __init__(self, criterion: str, device: torch.device):
        self.device = device
        self.criterion = criterion
        self.visualizer = Visualizer()

    def compute_reconstruction_errors(self, model, dataloader) -> Tuple[np.ndarray, np.ndarray]:
        losses: List[float] = []
        labels: List[int] = []
        model.eval()
        evaluation_criterion = getattr(nn, self.criterion)(reduction='none')
        with torch.no_grad():
            for inputs in tqdm(dataloader, desc="Computing reconstruction errors"):
                x = inputs[0]['encoder_cont'].to(self.device)
                targets = np.squeeze(inputs[1][0])
                if model._get_name() == 'TransformerAD':
                    output = model(x)
                    batch_rec_errors = evaluation_criterion(output['transformer_output'], output['ae_output'])
                else:
                    output = model(x)
                    batch_rec_errors = evaluation_criterion(x, output)

                loss_per_sample = batch_rec_errors.mean(dim=(1, 2))
                losses.extend(loss_per_sample.cpu().tolist())
                labels.extend(targets.tolist())
        return np.array(losses), np.array(labels, dtype=int)

    @staticmethod
    def find_optimal_threshold(rec_errors: np.ndarray,
                               labels: np.ndarray) -> float:
        fpr, tpr, thresholds = roc_curve(labels, rec_errors)
        optimal_idx = np.argmin(np.sqrt(np.power(fpr, 2) + np.power(1 - tpr, 2)))
        return thresholds[optimal_idx]

    @staticmethod
    def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
        return {
            'ae_roc_auc_score': roc_auc_score(y_true, scores),
            'ae_accuracy': accuracy_score(y_true, y_pred),
            'ae_precision': precision_score(y_true, y_pred),
            'ae_recall': recall_score(y_true, y_pred),
            'ae_f1': f1_score(y_true, y_pred)
        }

    def evaluate(self, model, test_dataloader) -> Tuple[dict[str, float], dict]:
        rec_errors, labels = self.compute_reconstruction_errors(model, test_dataloader)
        threshold = self.find_optimal_threshold(rec_errors, labels)
        y_pred = (rec_errors >= threshold).astype(int)

        metrics = self.compute_metrics(labels, y_pred, rec_errors)
        metrics['threshold'] = threshold
        self.visualizer.visualize(y_true=labels,
                                  y_pred=y_pred,
                                  scores=rec_errors,
                                  threshold=threshold,
                                  class_names=['Benign', 'Malicious'],
                                  prefix='test')
        return  metrics, self.visualizer.figures
