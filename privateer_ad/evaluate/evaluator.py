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
    """
    Evaluation framework for anomaly detection models.

    This class provides a complete evaluation pipeline for anomaly detection models,
    handling everything from raw model predictions through threshold optimization to
    final performance metrics. It's designed specifically for reconstruction-based
    anomaly detection approaches where the model learns to reconstruct normal patterns
    and flags deviations as anomalous.
    """
    def __init__(self, device: torch.device, loss_fn: str=None):
        """
        Initialize the model evaluator with device and loss function configuration.

        Sets up the evaluation environment by configuring the computational device
        and loss function. The loss function determines how reconstruction errors
        are computed, which directly affects threshold selection and final performance
        metrics.

        Args:
            device (torch.device): The device where model inference will be performed.
                                  This should match the device used during training for
                                  optimal performance.
            loss_fn (str, optional): Name of the PyTorch loss function to use for
                                   computing reconstruction errors. If not provided,
                                   uses the default from TrainingConfig.
        """
        logging.info('Instantiate ModelEvaluator...')

        self.device = device
        self.loss_fn = loss_fn or TrainingConfig().loss_fn_name

        self.visualizer = Visualizer()

    def compute_anomaly_scores(self, model, dataloader) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute reconstruction errors and ground truth labels for the entire dataset.

        This method performs inference across the complete dataset, computing
        reconstruction errors that serve as anomaly scores. The process handles
        batched data efficiently while maintaining memory constraints through
        proper tensor management and gradient computation disabled.

        The reconstruction errors are computed by comparing the model's output
        with its input, using the configured loss function. These errors form
        the basis for threshold selection and final anomaly classification.

        Args:
            model: The trained anomaly detection model to evaluate. Should be a
                  PyTorch model that takes input tensors and produces reconstructions.
            dataloader: DataLoader providing batched evaluation data with inputs
                       and corresponding ground truth labels.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
                - Input data arrays for potential further analysis
                - Ground truth binary labels (0 for normal, 1 for anomalous)
                - Computed anomaly scores (reconstruction errors)

        Note:
            The model is automatically moved to the configured device and set to
            evaluation mode. All computations are performed with gradients disabled
            for efficiency and memory conservation.
        """
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
        """
        Determine the optimal decision threshold using ROC curve analysis.

        This method finds the threshold that minimizes the Euclidean distance to
        the perfect classifier point (0, 1) in ROC space. This approach balances
        true positive and false positive rates effectively, providing a reasonable
        default threshold for anomaly detection scenarios.

        The optimization seeks the point on the ROC curve closest to perfect classification.
        This geometric approach often yields thresholds that work well in practice without requiring
        domain-specific tuning.

        Args:
            y_true (np.ndarray): Binary ground truth labels where 1 indicates
                               anomalous samples and 0 indicates normal samples.
            y_score (np.ndarray): Continuous anomaly scores, typically reconstruction
                                errors, where higher values suggest higher likelihood
                                of being anomalous.

        Returns:
            float: The optimal threshold value for binary classification. Samples
                  with scores above this threshold should be classified as anomalous.

        Note:
            This method assumes that higher scores indicate higher anomaly likelihood.
            The returned threshold represents the decision boundary that optimally
            separates normal and anomalous samples based on the ROC analysis.
        """

        fpr, tpr, thresholds = roc_curve(y_true=y_true, y_score=y_score)
        optimal_idx = np.argmin(np.sqrt(np.power(fpr, 2) + np.power(1 - tpr, 2)))
        return thresholds[optimal_idx]

    def evaluate(self, model, dataloader, threshold: int = None, prefix='', step=0) -> Tuple[dict[str, float], dict]:
        """
        Perform comprehensive model evaluation with metrics computation and visualization.

        This method orchestrates the complete evaluation process, from computing
        reconstruction errors through threshold optimization to final performance
        assessment. It generates a comprehensive set of metrics suitable for
        understanding model performance in anomaly detection scenarios.

        The evaluation process includes automatic threshold selection when not
        provided, classification performance metrics, and visualization generation.
        All results are automatically logged to MLflow when an active experiment
        session is available, facilitating experiment tracking and comparison.

        Args:
            model: The trained anomaly detection model to evaluate.
            dataloader: DataLoader providing evaluation data with ground truth labels.
            threshold (int, optional): Decision threshold for binary classification.
                                     If None, an optimal threshold is computed automatically.
            prefix (str, optional): Prefix for metric names in logging and reporting.
                                   Useful for distinguishing between different evaluation
                                   sets (e.g., 'test_', 'val_').
            step (int, optional): Training step or epoch number for metric logging.
                                Helps track performance evolution during training.

        Returns:
            Tuple[dict[str, float], dict]: A tuple containing:
                - Dictionary of computed metrics with descriptive names
                - Dictionary of generated visualization figures

        Note:
            When MLflow tracking is active, this method automatically logs all
            computed metrics, classification reports, and generated visualizations.
            The logging includes step information for tracking performance over time.
        """
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
