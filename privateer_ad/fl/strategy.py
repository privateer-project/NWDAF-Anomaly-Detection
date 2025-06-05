import logging
from typing import List, Tuple, Union, Optional

import numpy as np
import mlflow

from flwr.common import (ndarrays_to_parameters, parameters_to_ndarrays, Scalar, Parameters,
                         Metrics, EvaluateRes, FitRes)
from flwr.server.strategy import FedAvg
from flwr.server.client_proxy import ClientProxy

from privateer_ad.architectures import TransformerAD
from privateer_ad.config import TrainingConfig, MLFlowConfig
from privateer_ad.fl.utils import set_weights


def metrics_aggregation_fn(results: List[Tuple[int, Metrics]]):
    """Aggregate metrics across clients."""
    weighted_sums = {}
    total_num_examples = 0
    for num_examples, _metrics in results:
        total_num_examples += num_examples
        for name, value in _metrics.items():
            if name not in weighted_sums:
                weighted_sums[name] = 0
            weighted_sums[name] += (num_examples * value)
    weighted_metrics = {name: (value / total_num_examples) for name, value in weighted_sums.items()}
    return weighted_metrics


class CustomStrategy(FedAvg):
    """Custom federated learning strategy."""

    def __init__(self, training_config=None, mlflow_config=None):
        """
        Initialize the custom strategy.
        """
        logging.info("Initializing custom strategy...")

        # Store server run ID
        self.mlflow_config = mlflow_config or MLFlowConfig()
        self.training_config = training_config or TrainingConfig()
        # Create model with proper configuration
        self.model = TransformerAD()
        initial_parameters = ndarrays_to_parameters([val.cpu().numpy() for _, val in self.model.state_dict().items()])
        # Initialize parent strategy with custom config functions
        super().__init__(
            on_fit_config_fn=self._fit_config_fn,
            on_evaluate_config_fn=self._evaluate_config_fn,
            fit_metrics_aggregation_fn=metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=metrics_aggregation_fn,
            initial_parameters=initial_parameters
        )

        if self.training_config.direction == 'maximize':
            self.best_target_metric = -np.inf
        else:
            self.best_target_metric = np.inf
        self.best_metrics = {}

    def _fit_config_fn(self, server_round: int):
        """Generate fit configuration for each round."""
        return {'server_round': server_round,
                'server_run_id': self.mlflow_config.parent_run_id}

    def _evaluate_config_fn(self, server_round: int):
        """Generate evaluation configuration for each round."""
        return {'server_round': server_round}

    def _is_better_metric(self, current_value: float) -> bool:
        """Check if current metric value is better than the best value."""
        if self.training_config.direction == 'maximize':
            return current_value > self.best_target_metric
        else:  # minimize
            return current_value < self.best_target_metric

    def aggregate_fit(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, FitRes]],
            failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], dict[str, Scalar]]:
        """Aggregate fit results from clients."""
        parameters_aggregated, metrics_aggregated = super().aggregate_fit(server_round, results, failures)
        # Update best model if target metric improved
        logging.warning(f'metrics_aggregated: {metrics_aggregated}')
        if metrics_aggregated:
            logging.warning(f'self.training_config.target_metric: {self.training_config.target_metric}')
            if self.training_config.target_metric in metrics_aggregated:
                if self._is_better_metric(metrics_aggregated[self.training_config.target_metric]):
                    self.best_metrics = metrics_aggregated
                    self.best_target_metric = self.best_metrics[self.training_config.target_metric]
                    set_weights(self.model, parameters_to_ndarrays(parameters_aggregated))
                    logging.info(f"New best model at round {server_round} with "
                                 f"{self.training_config.target_metric}: {self.best_target_metric:.5f}")
        if mlflow.active_run() and metrics_aggregated:
            mlflow.log_metrics(metrics_aggregated, step=server_round)

        return parameters_aggregated, metrics_aggregated

    def aggregate_evaluate(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, EvaluateRes]],
            failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], dict[str, Scalar]]:
        """Aggregate evaluation results from clients."""
        loss_aggregated, metrics_aggregated = super().aggregate_evaluate(server_round, results, failures)
        # Log metrics to MLFlow
        if mlflow.active_run() and metrics_aggregated:
            mlflow.log_metrics(metrics_aggregated, step=server_round)
            mlflow.log_metric('test_loss', loss_aggregated, step=server_round)
        return loss_aggregated, metrics_aggregated
