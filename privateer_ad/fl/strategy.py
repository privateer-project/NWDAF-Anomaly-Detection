from typing import List, Tuple, Union, Optional

import numpy as np
import torch
import mlflow

from mlflow.entities import RunStatus
from flwr.common import (ndarrays_to_parameters, parameters_to_ndarrays, Scalar, Parameters,
                         Metrics, EvaluateRes, FitRes)
from flwr.server.strategy import FedAvg
from flwr.server.client_proxy import ClientProxy

from privateer_ad import logger
from privateer_ad.config import (
    get_mlflow_config,
    get_paths,
    get_privacy_config,
    get_model_config,
    get_training_config,
    get_data_config,
)
from privateer_ad.fl.utils import set_weights
from privateer_ad.etl.transform import DataProcessor
from privateer_ad.models import TransformerAD, TransformerADConfig
from privateer_ad.evaluate.evaluator import ModelEvaluator


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

def config_fn(server_round: int):
    """Generate evaluation configuration for each round."""
    return {'server_round': server_round}

class CustomStrategy(FedAvg):
    """Custom federated learning strategy."""

    def __init__(self, num_rounds: int, dp_enabled: bool | None):
        """
        Initialize the custom strategy.

        Args:
            num_rounds: Number of federated learning rounds
        """
        # Inject configurations
        self.paths_config = get_paths()
        self.mlflow_config = get_mlflow_config()
        self.privacy_config = get_privacy_config()
        self.dp_enabled = dp_enabled or self.privacy_config.dp_enabled
        self.model_config = get_model_config()
        self.training_config = get_training_config()
        self.data_config = get_data_config()

        self.num_rounds = num_rounds

        # Create model with proper configuration
        model_config = TransformerADConfig(
            seq_len=self.model_config.seq_len,
            input_size=self.model_config.input_size,
            num_layers=self.model_config.num_layers,
            hidden_dim=self.model_config.hidden_dim,
            latent_dim=self.model_config.latent_dim,
            num_heads=self.model_config.num_heads,
            dropout=self.model_config.dropout
        )

        self.model = TransformerAD(config=model_config)
        initial_parameters = ndarrays_to_parameters([
            val.cpu().numpy() for _, val in self.model.state_dict().items()
        ])

        # Initialize parent strategy
        super().__init__(
            on_fit_config_fn=config_fn,
            on_evaluate_config_fn=config_fn,
            fit_metrics_aggregation_fn=metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=metrics_aggregation_fn,
            initial_parameters=initial_parameters
        )

        # Setup data processing and evaluation
        self._setup_data_processing()
        self._setup_evaluation()
        self._setup_mlflow()
        # Best model tracking
        self.best_loss = np.Inf

    def _setup_data_processing(self):
        """Setup data processing components."""
        self.data_processor = DataProcessor(partition=False)
        self.test_dl = self.data_processor.get_dataloader(
            'test',
            batch_size=self.training_config.batch_size,
            seq_len=self.model_config.seq_len,
            only_benign=False
        )
        self.sample = next(iter(self.test_dl))[0]['encoder_cont'][:1].to('cpu')

    def _setup_evaluation(self):
        """Setup evaluation components."""
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.evaluator = ModelEvaluator(
            criterion=self.training_config.loss_function,
            device=self.device
        )

    def _setup_mlflow(self):
        """Setup MLFlow tracking if enabled."""
        if not self.mlflow_config.enabled:
            return

        server_run_name = self.mlflow_config.server_run_name
        server_run_id = None

        mlflow.set_tracking_uri(self.mlflow_config.server_address)
        mlflow.set_experiment(self.mlflow_config.experiment_name)

        if not server_run_name:
            server_run_name = 'federated_learning'

        if self.dp_enabled:
            server_run_name += '-dp'

        if mlflow.active_run():
            logger.info(
                f"Found active run: {mlflow.active_run().info.run_name} - "
                f"{mlflow.active_run().info.run_id}, ending it"
            )
            mlflow.end_run()

        # Search for existing run
        runs = mlflow.search_runs(
            experiment_names=[self.mlflow_config.experiment_name],
            filter_string=f'tags.mlflow.runName = \'{server_run_name}\'',
            max_results=1
        )

        if len(runs) > 0:
            server_run_id = runs.iloc[0].run_id
            logger.info(f'Found existing run: {server_run_name} - {server_run_id}')
            if not RunStatus.is_terminated(mlflow.get_run(server_run_id).info.status):
                mlflow.MlflowClient().set_terminated(run_id=server_run_id)

        mlflow.start_run(run_id=server_run_id, run_name=server_run_name)
        server_run_name = mlflow.active_run().info.run_name

        # Setup trial directory
        self.trial_path = self.paths_config.experiments_dir.joinpath(server_run_name)
        self.trial_path.mkdir(exist_ok=True, parents=True)

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

        # Final evaluation on server round completion
        if server_round == self.num_rounds:
            self._final_evaluation(server_round)

        return loss_aggregated, metrics_aggregated

    def _final_evaluation(self, server_round: int):
        """Perform final evaluation on the best model."""
        logger.info("Performing final evaluation on the global test set...")

        self.model.to(self.device)
        metrics, figures = self.evaluator.evaluate(
            self.model,
            self.test_dl,
            prefix='final_global',
            step=server_round
        )

        logger.info(f"Final global evaluation metrics: {metrics}")

        if mlflow.active_run():
            # Log final metrics
            mlflow.log_metrics(metrics, step=server_round)

            # Log figures
            for name, fig in figures.items():
                mlflow.log_figure(fig, f'{name}_final.png')

            # Save figures locally
            if hasattr(self, 'trial_path'):
                for name, fig in figures.items():
                    fig.savefig(self.trial_path.joinpath(f'{name}_final.png'))

            # Log the final model
            self._log_final_model()

    def _log_final_model(self):
        """Log the final model to MLFlow."""
        try:
            self.model.to('cpu')
            _input = self.sample.to('cpu')
            _output = self.model(_input)

            if isinstance(_output, dict):
                _output = {key: val.detach().numpy() for key, val in _output.items()}
            else:
                _output = _output.detach().numpy()

            mlflow.pytorch.log_model(
                pytorch_model=self.model,
                artifact_path='final_global_model',
                registered_model_name='privateer_global_model',
                signature=mlflow.models.infer_signature(
                    model_input=_input.detach().numpy(),
                    model_output=_output
                ),
                pip_requirements=str(self.paths_config.root_dir.joinpath('requirements.txt'))
            )

            logger.info("Final model logged to MLFlow successfully")

        except Exception as e:
            logger.error(f"Failed to log final model to MLFlow: {e}")

    def get_strategy_summary(self) -> dict:
        """Get a summary of the strategy configuration."""
        return {
            'strategy_type': 'CustomStrategy',
            'num_rounds': self.num_rounds,
            'best_loss': self.best_loss,
            'model_config': self.model_config.model_dump(),
            'training_config': self.training_config.model_dump(),
            'privacy_enabled': self.privacy_config.dp_enabled,
            'mlflow_enabled': self.mlflow_config.enabled
        }
