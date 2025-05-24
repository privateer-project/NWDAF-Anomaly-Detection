from typing import List, Tuple, Union, Optional

import numpy as np
import mlflow

from flwr.common import (ndarrays_to_parameters, parameters_to_ndarrays, Scalar, Parameters,
                         Metrics, EvaluateRes, FitRes)
from flwr.server.strategy import FedAvg
from flwr.server.client_proxy import ClientProxy

from privateer_ad import logger
from privateer_ad.architectures import TransformerAD, TransformerADConfig
from .utils import set_weights


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

    def __init__(self, input_size, server_run_id=None):
        """
        Initialize the custom strategy.
        """
        logger.info("Initializing custom strategy...")

        # Store server run ID
        self.server_run_id = server_run_id

        # Create model with proper configuration
        self.model_config = TransformerADConfig(input_size=input_size)
        self.model = TransformerAD(config=self.model_config)
        initial_parameters = ndarrays_to_parameters([
            val.cpu().numpy() for _, val in self.model.state_dict().items()
        ])

        # Initialize parent strategy with custom config functions
        super().__init__(
            on_fit_config_fn=self._fit_config_fn,
            on_evaluate_config_fn=self._evaluate_config_fn,
            fit_metrics_aggregation_fn=metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=metrics_aggregation_fn,
            initial_parameters=initial_parameters
        )
        # Best model tracking
        self.best_loss = np.Inf

    def _fit_config_fn(self, server_round: int):
        """Generate fit configuration for each round."""
        config = {
            'server_round': server_round,
        }

        # Add server run ID if available
        if self.server_run_id:
            config['server_run_id'] = self.server_run_id

        return config

    def _evaluate_config_fn(self, server_round: int):
        """Generate evaluation configuration for each round."""
        config = {
            'server_round': server_round,
        }

        # Add server run ID if available
        if self.server_run_id:
            config['server_run_id'] = self.server_run_id

        return config

    def aggregate_fit(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, FitRes]],
            failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], dict[str, Scalar]]:
        """Aggregate fit results from clients."""
        parameters_aggregated, metrics_aggregated = super().aggregate_fit(
            server_round, results, failures
        )

        # Update best model if validation loss improved
        if metrics_aggregated and 'val_loss' in metrics_aggregated:
            if metrics_aggregated['val_loss'] < self.best_loss:
                self.best_loss = metrics_aggregated['val_loss']
                set_weights(self.model, parameters_to_ndarrays(parameters_aggregated))
                logger.info(f"New best model at round {server_round} with val_loss: {self.best_loss:.5f}")

        # Log metrics to MLFlow
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
        loss_aggregated, metrics_aggregated = super().aggregate_evaluate(
            server_round=server_round,
            results=results,
            failures=failures
        )
        # Log metrics to MLFlow
        if mlflow.active_run() and metrics_aggregated:
            mlflow.log_metrics(metrics_aggregated, step=server_round)
            mlflow.log_metric('eval_loss', loss_aggregated, step=server_round)
        return loss_aggregated, metrics_aggregated
