from pathlib import Path
from typing import Dict, Tuple, List

import mlflow
import numpy as np
import torch
import torch.nn as nn
from matplotlib.figure import Figure
from tqdm import tqdm
from sklearn.metrics import (
    roc_curve, accuracy_score, precision_score, roc_auc_score,
    recall_score, f1_score, ConfusionMatrixDisplay,
    RocCurveDisplay
)

from config import MLFlowConfig, Paths, HParams


class ModelEvaluator:
    """Evaluates trained models and generates performance metrics and visualizations."""

    def __init__(self, model: nn.Module, hparams: HParams,
                 device: torch.device, paths: Paths,
                 mlflow_config: MLFlowConfig):
        self.model = model
        self.device = device
        self.hparams = hparams
        self.paths = paths
        self.mlflow_config = mlflow_config

    def compute_reconstruction_errors(self, dataloader) -> Tuple[np.ndarray, np.ndarray]:
        """Compute reconstruction errors for the given dataloader."""
        rec_errors: List[float] = []
        labels: List[int] = []
        self.model.eval()
        evaluation_criterion = getattr(nn, self.hparams.loss)(reduction='none')
        with torch.no_grad():
            for inputs in tqdm(dataloader, desc="Computing reconstruction errors"):
                x = inputs[0]['encoder_cont'].to(self.device)
                targets = inputs[0]['encoder_target'][:, 0]
                output = self.model(x)
                rec_error = evaluation_criterion(x, output).mean(dim=(1, 2))
                rec_errors.extend(rec_error.cpu().tolist())
                labels.extend(targets.tolist())

        return np.array(rec_errors), np.array(labels, dtype=int)

    @staticmethod
    def find_optimal_threshold(rec_errors: np.ndarray,
                               labels: np.ndarray) -> float:
        """Find optimal threshold using ROC curve."""
        fpr, tpr, thresholds = roc_curve(labels, rec_errors)
        # Find threshold that minimizes distance to perfect classifier (0,1)
        optimal_idx = np.argmin(np.sqrt(np.power(fpr, 2) + np.power(1 - tpr, 2)))
        return thresholds[optimal_idx]

    @staticmethod
    def compute_metrics(labels: np.ndarray, rec_errors: np.ndarray,
                        threshold: float) -> Dict[str, float]:
        """Compute classification metrics."""
        predictions = (rec_errors >= threshold).astype(int)

        return {
            'ae_roc_auc_score': roc_auc_score(labels, predictions),
            'ae_accuracy': accuracy_score(labels, predictions),
            'ae_precision': precision_score(labels, predictions),
            'ae_recall': recall_score(labels, predictions),
            'ae_f1': f1_score(labels, predictions)
        }

    def generate_plots(self, labels: np.ndarray, predictions: np.ndarray) -> Tuple[Figure, Figure]:
        """Generate and save confusion matrix and ROC curve plots."""
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

    def evaluate(self,test_dataloader, save_path: Path=None) -> Dict[str, float]:
        """
        Evaluate model performance on test data.

        Returns:
            Dictionary containing evaluation metrics
        """
        # Compute reconstruction errors
        rec_errors, labels = self.compute_reconstruction_errors(test_dataloader)

        # Find optimal threshold
        threshold = self.find_optimal_threshold(rec_errors, labels)

        # Compute metrics
        metrics = self.compute_metrics(labels, rec_errors, threshold)

        # Generate plots
        predictions = (rec_errors >= threshold).astype(int)
        cm_fig, roc_fig = self.generate_plots(labels, predictions)

        if save_path:
            plots_dir = save_path.joinpath('plots')
            plots_dir.mkdir(parents=True, exist_ok=True)
            cm_path = plots_dir.joinpath('confusion_matrix.png')
            roc_path = plots_dir.joinpath('roc_curve.png')
            cm_fig.savefig(cm_path)
            roc_fig.savefig(roc_path)

        # Log to MLFlow if enabled
        if self.mlflow_config.track:
            mlflow.log_param('threshold', threshold)
            mlflow.log_metrics(metrics)
            mlflow.log_figure(cm_fig, 'confusion_matrix.png')
            mlflow.log_figure(roc_fig, 'roc_curve.png')
        return metrics
