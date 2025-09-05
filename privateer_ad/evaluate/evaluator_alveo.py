# privateer_ad/evaluate/evaluate_alveo.py
import logging
from typing import Tuple, List, Dict, Optional, Any

import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import roc_curve, roc_auc_score, classification_report
from torch.utils.data import DataLoader

from privateer_ad.config import TrainingConfig
from privateer_ad.visualizations import Visualizer
from privateer_ad.alveo.alveo_runner import AlveoRunner, convert_types


class AlveoEvaluator:
    """
    Evaluation framework for anomaly detection with an Alveo FPGA runner.

    Mirrors ModelEvaluator's structure and behavior, but uses an AlveoRunner
    to obtain reconstructions instead of a PyTorch model forward pass.
    """

    def __init__(self, loss_fn: Optional[str] = None):
        """
        Initialize evaluator with loss function and visualizer.

        Args:
            loss_fn (str, optional): Name of a torch.nn loss (e.g., "L1Loss").
                                     Defaults to TrainingConfig().loss_fn_name.
        """
        logging.info("Instantiate AlveoEvaluator...")
        self.loss_fn = loss_fn or TrainingConfig().loss_fn_name
        self.visualizer = Visualizer()

    def compute_anomaly_scores(
        self, runner: AlveoRunner, dataloader: DataLoader
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run the dataset through the FPGA and compute per-sample reconstruction errors.

        Returns:
            x          : np.ndarray of inputs (for parity with ModelEvaluator)
            y_true     : np.ndarray of ground-truth labels (0/1)
            y_score    : np.ndarray of anomaly scores (mean reconstruction error)
        """
        x: List[float] = []
        y_true: List[int] = []
        y_score: List[float] = []

        # Loss with elementwise reduction to later average per-sample
        loss_fn = getattr(torch.nn, self.loss_fn)(reduction="none")

        # Expected flattened block size from the runner (seq_len * n_features)
        flat_block = runner.parameters.input_buffer_elements
        np_input_dtype = convert_types(runner.parameters.input_t)

        with torch.no_grad():
            progress_bar = tqdm(dataloader, desc='Computing reconstruction errors')
            
            for inputs in progress_bar:
                batch_input = inputs[0]["encoder_cont"]   # [B, T, F] torch.Tensor
                batch_y_true = np.squeeze(inputs[1][0])     # (B,)

                B, T, F = batch_input.shape
                print(f"{B}, {T}, {F}")

                if T * F != flat_block:
                    raise AssertionError(
                        f"Per-sample size T*F={T*F} must equal "
                        f"runner input_buffer_elements={flat_block}"
                    )

                # Convert to numpy with the dtype expected by the runner
                x_np = batch_input.cpu().numpy()

                # FPGA reconstruction (shape back to [B, T, F] for loss calc)
                out_np = runner.run_vector(
                    input_vector=x_np,
                    output_shape=(B, T, F),
                    timed=False,
                    verbose=False,
                )
                batch_output = torch.tensor(out_np, dtype=torch.float32)
                print(batch_output.shape)

                batch_score = loss_fn(batch_input, batch_output)
                batch_y_score_per_sample = batch_score.mean(dim=(1, 2))
                
                x.extend(batch_input.tolist())
                y_true.extend(batch_y_true.tolist())
                y_score.extend(batch_y_score_per_sample.cpu().tolist())

        return np.array(x, dtype=np.float32), np.array(y_true, dtype=np.int32), np.array(y_score, dtype=np.float32)

    @staticmethod
    def find_optimal_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
        """
        Same ROC-based threshold selection as ModelEvaluator.
        """
        fpr, tpr, thresholds = roc_curve(y_true=y_true, y_score=y_score)
        optimal_idx = np.argmin(np.sqrt(np.power(fpr, 2) + np.power(1 - tpr, 2)))
        return float(thresholds[optimal_idx])

    def evaluate(
        self,
        runner: AlveoRunner,
        dataloader: DataLoader,
        threshold: Optional[float] = None,
        prefix: str = "",
        step: int = 0,
    ) -> Tuple[Dict[str, float], Dict[str, Any]]:
        """
        End-to-end evaluation with metrics, visuals, and (optional) MLflow logging.
        Mirrors ModelEvaluator.evaluate signature & behavior.
        """
        x, y_true, anomaly_scores = self.compute_anomaly_scores(runner, dataloader)

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

        metrics['threshold'] = float(threshold)
        if prefix != '':
            metrics = {f'_'.join([prefix, k]): v for k, v in metrics.items()}
        self.visualizer.visualize(y_true=y_true,
                                  y_pred=y_pred,
                                  scores=anomaly_scores,
                                  threshold=threshold,
                                  target_names=target_names,
                                  prefix=prefix
                                  )

        # Optional MLflow logging (kept identical in spirit)
        try:
            import mlflow

            if mlflow.active_run():
                report_name = (
                    f"{str(step).zfill(3)}_{prefix}_classification_report.txt"
                    if prefix
                    else f"{str(step).zfill(3)}_classification_report.txt"
                )
                mlflow.log_text(
                    classification_report(
                        y_true=y_true, y_pred=y_pred, target_names=target_names
                    ),
                    report_name,
                )
                mlflow.log_metrics(metrics, step=step)
                for name, fig in self.visualizer.figures.items():
                    mlflow.log_figure(fig, f"{str(step).zfill(3)}_{name}.png")
        except Exception as e:  # pragma: no cover
            logging.warning(f"MLflow logging skipped: {e}")

        # Pretty log
        metrics_logs = "\n".join([f"{k}: {v}" for k, v in metrics.items()])
        logging.info(f"Test metrics:\n{metrics_logs}")

        return metrics, self.visualizer.figures


# --- Backwards-compatible wrapper (so marimo.py keeps working) ----------------

def evaluate_alveo(
    runner: AlveoRunner,
    dataloader: DataLoader,
    threshold: float = None,
    batch_size: Optional[int] = None,  # kept for signature compatibility; unused
    loss_fn_name: str = None,
    mlflow_prefix: str = "alveo",
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """
    Thin wrapper around AlveoEvaluator to preserve the existing import & call sites.
    """
    evaluator = AlveoEvaluator(loss_fn=loss_fn_name)
    return evaluator.evaluate(
        runner=runner,
        dataloader=dataloader,
        threshold=threshold,
        prefix=mlflow_prefix,
        step=0,
    )
