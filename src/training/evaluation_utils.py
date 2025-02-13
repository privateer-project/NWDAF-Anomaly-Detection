from typing import Dict, Tuple, List

import numpy as np
import torch
import torch.nn as nn
from matplotlib.figure import Figure
from tqdm import tqdm
from sklearn.metrics import (roc_curve, accuracy_score, precision_score, roc_auc_score,
                             recall_score, f1_score, ConfusionMatrixDisplay, RocCurveDisplay)

class ModelEvaluator:
    def __init__(self, criterion: str, device: torch.device):
        self.device = device
        self.criterion = criterion

    def compute_reconstruction_errors(self, model, dataloader) -> Tuple[np.ndarray, np.ndarray]:
        rec_errors: List[float] = []
        labels: List[int] = []
        model.eval()
        evaluation_criterion = getattr(nn, self.criterion)(reduction='none')
        with torch.no_grad():
            for inputs in tqdm(dataloader, desc="Computing reconstruction errors"):
                x = inputs[0]['encoder_cont'].to(self.device)
                targets = inputs[0]['encoder_target'][:, 0]
                output = model(x)
                rec_error = evaluation_criterion(x, output).mean(dim=(1, 2))
                rec_errors.extend(rec_error.cpu().tolist())
                labels.extend(targets.tolist())
        return np.array(rec_errors), np.array(labels, dtype=int)

    @staticmethod
    def find_optimal_threshold(rec_errors: np.ndarray,
                               labels: np.ndarray) -> float:
        fpr, tpr, thresholds = roc_curve(labels, rec_errors)
        optimal_idx = np.argmin(np.sqrt(np.power(fpr, 2) + np.power(1 - tpr, 2)))
        return thresholds[optimal_idx]

    @staticmethod
    def compute_metrics(labels: np.ndarray, rec_errors: np.ndarray,
                        threshold: float) -> Dict[str, float]:
        predictions = (rec_errors >= threshold).astype(int)

        return {
            'ae_roc_auc_score': roc_auc_score(labels, predictions),
            'ae_accuracy': accuracy_score(labels, predictions),
            'ae_precision': precision_score(labels, predictions),
            'ae_recall': recall_score(labels, predictions),
            'ae_f1': f1_score(labels, predictions)
        }

    @staticmethod
    def generate_plots(labels: np.ndarray, predictions: np.ndarray) -> Tuple[Figure, Figure]:
        cm_display = ConfusionMatrixDisplay.from_predictions(
            y_true=labels,
            y_pred=predictions,
            colorbar=False,
            cmap='inferno'
        )
        cm_display.figure_.suptitle('Confusion Matrix')

        roc_display = RocCurveDisplay.from_predictions(
            y_true=labels,
            y_pred=predictions,
            plot_chance_level=True
        )
        roc_display.figure_.suptitle('ROC Curve')
        return cm_display.figure_, roc_display.figure_

    def evaluate(self, model, test_dataloader) -> [dict[str, float], dict]:
        rec_errors, labels = self.compute_reconstruction_errors(model, test_dataloader)
        threshold = self.find_optimal_threshold(rec_errors, labels)

        metrics = self.compute_metrics(labels, rec_errors, threshold)
        metrics['threshold'] = threshold

        predictions = (rec_errors >= threshold).astype(int)
        cm_fig, roc_fig = self.generate_plots(labels, predictions)
        return  metrics, {'confusion_matrix.png': cm_fig,
                          'roc_curve.png': roc_fig}
