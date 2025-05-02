from collections import OrderedDict
from datetime import datetime
from typing import Tuple, List, Union, Optional

import mlflow

import numpy as np
import torch


from flwr.common import Context, Metrics, ndarrays_to_parameters, FitRes, Parameters, Scalar, parameters_to_ndarrays
from flwr.server import ServerApp, ServerConfig, ClientManager, LegacyContext, Driver
from flwr.server.strategy import FedAvg, DifferentialPrivacyClientSideFixedClipping
from flwr.server.workflow import SecAggPlusWorkflow, DefaultWorkflow
from flwr.server.client_proxy import ClientProxy

from flwr.client.client import FitIns

from privateer_ad.config import DifferentialPrivacyConfig, SecureAggregationConfig, HParams, MLFlowConfig, \
    AttentionAutoencoderConfig, PathsConf, logger
from privateer_ad.models import AttentionAutoencoder
from privateer_ad.utils import set_config

def make_config():
    config = {}
    config.update(HParams().__dict__)
    config.update(AttentionAutoencoderConfig().__dict__)
    config.update(DifferentialPrivacyConfig().__dict__)
    config.update(SecureAggregationConfig().__dict__)
    config.update(MLFlowConfig().__dict__)
    # config.update(PathsConf().__dict__)
    return config


config = make_config()
dp_config = set_config(DifferentialPrivacyConfig, config)
secagg_config = set_config(SecureAggregationConfig, config)
mlflow_config = set_config(MLFlowConfig, config)
model_config = set_config(AttentionAutoencoderConfig, config)
paths = set_config(PathsConf, config)

trial_id = datetime.now().strftime("%Y%m%d-%H%M%S")
trial_path = paths.experiments_dir.joinpath('fl_train').joinpath(trial_id)
trial_path.mkdir(exist_ok=True, parents=True)
model = AttentionAutoencoder(model_config)


class CustomStrategy(FedAvg):
    def __init__(self,
                 on_fit_config_fn,
                 on_evaluate_config_fn,
                 evaluate_metrics_aggregation_fn,
                 fit_metrics_aggregation_fn,
                 initial_parameters):

        super().__init__(on_fit_config_fn=on_fit_config_fn,
                         on_evaluate_config_fn=on_evaluate_config_fn,
                         evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
                         fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
                         initial_parameters=initial_parameters)
        self.best_loss = float('inf')
        self.mlflow_config = mlflow_config
        self.mlflow_parent_run_id = None

        # Initialize MLflow parent run if tracking is enabled
        if self.mlflow_config and self.mlflow_config.track:
            mlflow.set_tracking_uri(self.mlflow_config.server_address)
            mlflow.set_experiment(self.mlflow_config.experiment_name)

            # Start the parent run
            with mlflow.start_run(run_name="federated_learning") as run:
                self.mlflow_parent_run_id = run.info.run_id
                logger.info(f"Created parent run for federated learning: {self.mlflow_parent_run_id}")

                # Log initial parameters
                mlflow.log_param("num_clients", on_fit_config_fn(1).get('num_partitions', 1))
                mlflow.log_param("model", on_fit_config_fn(1).get('model', 'Unknown'))
                # Log other relevant parameters

    def aggregate_fit(self,
                      server_round: int,
                      results: list[tuple[ClientProxy, FitRes]],
                      failures: list[Union[tuple[ClientProxy, FitRes], BaseException]],
                      ) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        """Aggregate model weights using weighted average and store checkpoint"""

        # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
        parameters_aggregated, metrics_aggregated = super().aggregate_fit(server_round, results, failures)

        if parameters_aggregated is not None:
            if metrics_aggregated['val_loss'] < self.best_loss:
                self.best_loss = metrics_aggregated['val_loss']
                print(f"Saving round {server_round} aggregated_parameters...")

            # Convert `Parameters` to `list[np.ndarray]`
            aggregated_ndarrays: list[np.ndarray] = parameters_to_ndarrays(parameters_aggregated)

            # Convert to PyTorch `state_dict`
            params_dict = zip(model.state_dict().keys(), aggregated_ndarrays)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            model.load_state_dict(state_dict, strict=True)
            # Save the model to disk
            model_path = trial_path.joinpath(f"model.pt")
            torch.save(model.state_dict(), model_path)

        # Log metrics to MLflow parent run
        if self.mlflow_config and self.mlflow_config.track and self.mlflow_parent_run_id:
            # Log aggregated metrics
            with mlflow.start_run(run_id=self.mlflow_parent_run_id):
                for key, value in metrics_aggregated.items():
                    mlflow.log_metric(key=f"aggregated_{key}",
                                      value=value,
                                      step=server_round)
        return parameters_aggregated, metrics_aggregated

    def configure_fit(
            self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:

        """Configure the next round of training."""
        # Get FitIns from base class
        fit_ins = super().configure_fit(server_round, parameters, client_manager)
        print('fit_ins', '*' * 100,  fit_ins)
        # Add the parent run ID to the config for each client
        if self.mlflow_parent_run_id:
            for _, ins in fit_ins:
                ins.config["mlflow_parent_run_id"] = self.mlflow_parent_run_id
        return fit_ins

def on_fit_config_fn(server_round: int):
    """Generate training configuration for each round."""
    # Create the configuration dictionary
    config["server_round"] = server_round
    return config

def on_evaluate_config_fn(server_round: int):
    """Generate evaluation configuration for each round."""
    # Create the configuration dictionary
    config["server_round"] = server_round
    return config

def get_model_weights(model):
    """Get model weights as NDArrays."""
    return ndarrays_to_parameters([val.cpu().numpy() for _, val in model.state_dict().items()])

def weighted_average(results: List[Tuple[int, Metrics]]) -> Metrics:
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

# Flower ServerApp
app = ServerApp()

@app.main()
def main(driver: Driver, context: Context) -> None:
    server_address: str = "[::]:8081"
    logger.info(f"MAIN CONTEXT: {context}")
    n_clients = context.run_config['n_clients']

    # Set up MLflow - first check if there's an active run and end it
    if mlflow_config.track:
        mlflow.set_tracking_uri(mlflow_config.server_address)
        mlflow.set_experiment(mlflow_config.experiment_name)
        # End any active runs to avoid errors
        active_run = mlflow.active_run()
        if active_run:
            logger.warning(f"Found active run {active_run.info.run_id}, ending it")
            mlflow.end_run()

        # Now create a new run
        parent_run = mlflow.start_run(run_name=f"fl_{trial_id}")
        parent_run_id = parent_run.info.run_id
        mlflow.end_run()
        logger.info(f"Created parent run: {parent_run_id}")
        # Log parameters
        mlflow.log_param("num_clients", n_clients)
        mlflow.log_param("num_rounds", context.run_config["num-server-rounds"])
        context.node_config["mlflow_parent_run_id"] = parent_run_id

    # Initialize model and get weights
    initial_parameters = get_model_weights(model)
    # Create strategy with the modified config function
    strategy = CustomStrategy(on_fit_config_fn=on_fit_config_fn,
                              on_evaluate_config_fn=on_evaluate_config_fn,
                              evaluate_metrics_aggregation_fn=weighted_average,
                              fit_metrics_aggregation_fn=weighted_average,
                              initial_parameters=initial_parameters)


    # Set up server context
    context = LegacyContext(
        context=context,
        config=ServerConfig(num_rounds=context.run_config['num-server-rounds']),
        strategy=DifferentialPrivacyClientSideFixedClipping(
            strategy=strategy,
            noise_multiplier=dp_config.noise_multiplier,
            clipping_norm=dp_config.max_grad_norm,
            num_sampled_clients=n_clients)
    )
    # Configure SecAgg+ workflow
    fit_workflow = SecAggPlusWorkflow(
        num_shares=secagg_config.num_shares,
        reconstruction_threshold=secagg_config.reconstruction_threshold)

    workflow = DefaultWorkflow(fit_workflow=fit_workflow)
    # Execute workflow
    logger.info(f'Starting Flower server with SecAgg+ at {server_address}')
    logger.info(f'Server will run for {context.run_config["num-server-rounds"]} rounds')
    logger.info(f'SecAgg+ config: {secagg_config.num_shares} shares\n'
                f'{secagg_config.reconstruction_threshold} reconstruction threshold')

    try:
        workflow(driver, context)
    finally:
        # Make sure to end the MLflow run when done
        if mlflow_config.track and mlflow.active_run():
            mlflow.end_run()
