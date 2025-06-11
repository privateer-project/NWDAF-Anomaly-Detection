import logging
from typing import List, Tuple, Union, Optional

import numpy as np
import mlflow

from flwr.common import (ndarrays_to_parameters, parameters_to_ndarrays, Scalar, Parameters,
                         Metrics, EvaluateRes, FitRes)
from flwr.server.strategy import FedAvg
from flwr.server.client_proxy import ClientProxy

from privateer_ad.architectures import TransformerAD
from privateer_ad.config import TrainingConfig, MLFlowConfig, ModelConfig
from privateer_ad.fl.utils import set_weights


def metrics_aggregation_fn(results: List[Tuple[int, Metrics]]):
    """
    Aggregate performance metrics from multiple federated learning clients.
    Args:
    results (List[Tuple[int, Metrics]]): Collection of client results where
                                       each tuple contains the number of
                                       examples used by the client and a
                                       dictionary of computed metrics.
    Returns:
        dict: Weighted average of metrics across all participating clients,
              where each metric value represents the dataset-size-weighted
              mean of individual client contributions.
    """
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
    """
    Advanced federated learning strategy with model tracking and adaptive coordination.

    This strategy extends the standard Federated Averaging approach with
    sophisticated model state management, comprehensive experiment tracking,
    and adaptive coordination capabilities tailored for anomaly detection
    scenarios. The implementation maintains a global model state that evolves
    through collaborative learning while preserving the best-performing model
    configurations throughout the federated training process.

    The strategy incorporates intelligent model selection based on aggregated
    performance metrics, ensuring that the global model consistently represents
    the best collaborative learning outcome achieved across all federation rounds.
    This approach prevents performance regression that might occur from temporary
    training fluctuations or suboptimal client contributions in specific rounds.

    The implementation integrates comprehensive experiment tracking through MLflow,
    providing detailed visibility into federated learning dynamics, model evolution,
    and client-specific contributions. This tracking capability proves invaluable
    for understanding collaborative learning behavior and optimizing federation
    configurations for specific deployment scenarios.

    Attributes:
        mlflow_config (MLFlowConfig): Configuration for experiment tracking coordination
        training_config (TrainingConfig): Training parameters and optimization settings
        model_config (ModelConfig): Model architecture and hyperparameter specifications
        model (TransformerAD): Global model instance maintained throughout federation
        best_target_metric (float): Best achieved value for the optimization target
        best_metrics (dict): Complete metric set from the best performing round
    """

    def __init__(self, training_config=None, model_config=None, mlflow_config=None):
        """
        Initialize the federated learning strategy with comprehensive configuration.
        Args:
            training_config (TrainingConfig, optional): Training parameters including
                                                      optimization direction and target
                                                      metrics. Defaults to standard
                                                      configuration if not provided.
            model_config (ModelConfig, optional): Model architecture specifications
                                                including layer dimensions and
                                                hyperparameters. Uses default settings
                                                if not specified.
            mlflow_config (MLFlowConfig, optional): Experiment tracking configuration
                                                  for coordinating with server-side
                                                  experiment management. Defaults to
                                                  standard tracking setup.

        Note:
            The strategy automatically determines the optimal metric direction
            (maximize or minimize) and initializes tracking variables accordingly.
            This setup ensures that model selection throughout federation consistently
            improves the target performance metric.
        """
        logging.info("Initializing custom strategy...")

        # Store server run ID
        self.mlflow_config = mlflow_config or MLFlowConfig()
        self.training_config = training_config or TrainingConfig()
        self.model_config = model_config or ModelConfig()
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
        """
        Generate round-specific configuration for client training coordination.
        Args:
            server_round (int): Current federation round number for tracking and
                              coordination purposes.

        Returns:
            dict: Configuration dictionary containing round information and
                  experiment tracking identifiers for client coordination.
        """
        return {'server_round': server_round,
                'server_run_id': self.mlflow_config.parent_run_id}

    def _evaluate_config_fn(self, server_round: int):
        """
        Generate round-specific configuration for client evaluation coordination.
        Args:
            server_round (int): Current federation round number for tracking and
                              coordination purposes.

        Returns:
            dict: Configuration dictionary containing round information and
                  experiment tracking identifiers for client coordination.
        """
        return {'server_round': server_round}

    def _is_better_metric(self, current_value: float) -> bool:
        """
        Determine whether current performance represents an improvement over previous best.
        Args:
            current_value (float): Performance value from current federation round
                                 for comparison with historical best performance.

        Returns:
            bool: True if current performance represents an improvement, False
                  if performance has not exceeded previous best results.
        """

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
        """
        Aggregate training results from federated clients with intelligent model selection.

        The method implements sophisticated model tracking that evaluates aggregated
        performance metrics after each round and updates the global model state only
        when improvements are achieved. This approach prevents performance regression
        from temporary training fluctuations while ensuring that the federation
        consistently progresses toward better collaborative learning outcomes.

        Args:
            server_round (int): Current federation round number for tracking and
                              coordination purposes.
            results (List[Tuple[ClientProxy, FitRes]]): Training results from
                                                      participating clients containing
                                                      model updates and performance metrics.
            failures (List[Union[Tuple[ClientProxy, FitRes], BaseException]]):
                                                      Collection of client failures
                                                      encountered during training
                                                      for error handling and analysis.

        Returns:
            Tuple[Optional[Parameters], dict[str, Scalar]]: Aggregated model parameters
                                                          and performance metrics for
                                                          distribution to clients in
                                                          subsequent federation rounds.
        """
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
        """
        Aggregate evaluation results from federated clients with comprehensive tracking.

        This method combines evaluation results from distributed clients to assess
        overall federation performance on test data. The aggregation provides insights
        into how well the collaborative model generalizes across different data
        distributions represented by federation participants, which is crucial for
        understanding federated learning effectiveness in heterogeneous environments.
        Args:
            server_round (int): Current federation round for evaluation tracking
                              and performance analysis coordination.
            results (List[Tuple[ClientProxy, EvaluateRes]]): Evaluation results from
                                                           participating clients containing
                                                           performance metrics on local
                                                           test data.
            failures (List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]]):
                                                           Collection of evaluation
                                                           failures for error analysis
                                                           and debugging support.
        Returns:
            Tuple[Optional[float], dict[str, Scalar]]: Aggregated evaluation loss
                                                     and comprehensive performance
                                                     metrics for federation assessment
                                                     and tracking purposes.
        """
        loss_aggregated, metrics_aggregated = super().aggregate_evaluate(server_round, results, failures)
        # Log metrics to MLFlow
        if mlflow.active_run() and metrics_aggregated:
            mlflow.log_metrics(metrics_aggregated, step=server_round)
            mlflow.log_metric('test_loss', loss_aggregated, step=server_round)
        return loss_aggregated, metrics_aggregated
