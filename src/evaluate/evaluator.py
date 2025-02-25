from typing import Dict, Tuple, List

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_curve, roc_auc_score, classification_report

from src.visualizations.plotter import Visualizer


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
    def compute_metrics(y_true, y_pred, rec_errors, threshold, labels, target_names) -> Dict[str, float]:
        clf_report = classification_report(y_true=y_true,
                                           y_pred=y_pred,
                                           labels=labels,
                                           target_names=target_names,
                                           output_dict=True)
        final_report = {'threshold': threshold,
                        'roc_auc': roc_auc_score(y_true=y_true, y_score=rec_errors)}
        final_report.update(clf_report['macro avg'])
        return final_report

    def evaluate(self, model, test_dataloader) -> Tuple[dict[str, float], dict]:
        rec_errors, y_true = self.compute_reconstruction_errors(model, test_dataloader)
        threshold = self.find_optimal_threshold(rec_errors, y_true)
        y_pred = (rec_errors >= threshold).astype(int)

        target_names = ['benign', 'malicious']
        labels = [0, 1]

        final_report = self.compute_metrics(
            y_true=y_true,
            y_pred=y_pred,
            rec_errors=rec_errors,
            threshold=threshold,
            labels=labels,
            target_names=target_names)
        self.visualizer.visualize(
            y_true=y_true,
            y_pred=y_pred,
            scores=rec_errors,
            threshold=threshold,
            target_names=target_names,
            prefix='test')

        return final_report, self.visualizer.figures
