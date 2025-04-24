"""privateer_ad: A Flower / PyTorch app."""
from collections import OrderedDict
from datetime import datetime
from typing import Tuple, List, Union, Optional
import json

import flwr.common.logger
import mlflow
import numpy as np
import torch
from flwr.common import Context, Metrics, ndarrays_to_parameters, FitRes, Parameters, Scalar, parameters_to_ndarrays
from flwr.server import ServerApp, ServerConfig, LegacyContext, Driver
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg, DifferentialPrivacyClientSideFixedClipping
from flwr.server.workflow import SecAggPlusWorkflow, DefaultWorkflow

from privateer_ad.config import DifferentialPrivacyConfig, SecureAggregationConfig, HParams, MLFlowConfig, \
    AttentionAutoencoderConfig, PathsConf, logger
from privateer_ad.models import AttentionAutoencoder
from privateer_ad.utils import set_config


from privateer_ad.save_model import FedAvgWithModelSaving


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

# class CustomStrategy(FedAvg):
#     def __init__(self,
#                  on_fit_config_fn,
#                  on_evaluate_config_fn,
#                  evaluate_metrics_aggregation_fn,
#                  fit_metrics_aggregation_fn,
#                  initial_parameters):
#         super().__init__(on_fit_config_fn=on_fit_config_fn,
#                          on_evaluate_config_fn=on_evaluate_config_fn,
#                          evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
#                          fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
#                          initial_parameters=initial_parameters)
#         self.best_loss = float('inf')

#     def aggregate_fit(
#         self,
#         server_round: int,
#         results: list[tuple[ClientProxy, FitRes]],
#         failures: list[Union[tuple[ClientProxy, FitRes], BaseException]],
#     ) -> tuple[Optional[Parameters], dict[str, Scalar]]:
#         """Aggregate model weights using weighted average and store checkpoint"""

#         # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
#         parameters_aggregated, metrics_aggregated = super().aggregate_fit(
#             server_round, results, failures
#         )
#         if parameters_aggregated is not None and (metrics_aggregated['val_loss'] < self.best_loss):
#             print(f"Saving round {server_round} aggregated_parameters...")

#             # Convert `Parameters` to `list[np.ndarray]`
#             aggregated_ndarrays: list[np.ndarray] = parameters_to_ndarrays(
#                 parameters_aggregated
#             )

#             # Convert `list[np.ndarray]` to PyTorch `state_dict`
#             params_dict = zip(model.state_dict().keys(), aggregated_ndarrays)
#             state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
#             model.load_state_dict(state_dict, strict=True)
#             # Save the model to disk
#             model_path = trial_path.joinpath(f"model.pt")
#             torch.save(model.state_dict(), model_path)
#         return parameters_aggregated, metrics_aggregated

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
    with open(f"results/dp_level_{dp_config.noise_multiplier}_max_grad_norm_{dp_config.max_grad_norm}_target_epsilon_10_scores.txt", "a") as f: 
        print(weighted_metrics, file=f)
        f.close()
    return weighted_metrics

# Flower ServerApp
app = ServerApp()

@app.main()
def main(driver: Driver, context: Context) -> None:

    dp_config.noise_multiplier = context.run_config['noise_multiplier']
    dp_config.max_grad_norm = context.run_config['max_grad_norm']

    trial_id = datetime.now().strftime("%Y%m%d")
    trial_path = paths.experiments_dir.joinpath(f'fl_train_noise_{dp_config.noise_multiplier}_max_grad_norm_{dp_config.max_grad_norm}').joinpath(trial_id)
    trial_path.mkdir(exist_ok=True, parents=True)
    model = AttentionAutoencoder(model_config)
    model_keys = model.state_dict().keys()
    server_address: str = "[::]:8081"
    logger.info(f"MAIN CONTEXT: {context}")
    n_clients = context.run_config['n_clients']
    initial_parameters = get_model_weights(model)
    # strategy = CustomStrategy(
    #     on_fit_config_fn=on_fit_config_fn,
    #     on_evaluate_config_fn=on_evaluate_config_fn,
    #     evaluate_metrics_aggregation_fn=weighted_average,
    #     fit_metrics_aggregation_fn=weighted_average,
    #     initial_parameters=initial_parameters)

    strategy = FedAvgWithModelSaving(
        save_path=trial_path,
        model_keys=model_keys,
        on_fit_config_fn=on_fit_config_fn,
        on_evaluate_config_fn=on_evaluate_config_fn,
        min_fit_clients=n_clients,
        min_evaluate_clients=n_clients,
        min_available_clients=n_clients,
        evaluate_metrics_aggregation_fn=weighted_average,
        initial_parameters=initial_parameters)


    # Configure MLflow if enabled
    if mlflow_config.track:
        mlflow.set_tracking_uri(mlflow_config.server_address)
        mlflow.set_experiment(mlflow_config.experiment_name)
        mlflow.start_run(run_name=f"federated_{trial_id}_{context.run_config['num-server-rounds']}_rounds")

    # Get initial parameters from the model
    # Try to update here
    # context.node_config = node_config
    # context.run_config = run_config

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
    flwr.common.logger.FLOWER_LOGGER.info(f'Starting Flower server with SecAgg+ at {server_address}')
    flwr.common.logger.FLOWER_LOGGER.info(f'Server will run for {context.run_config["num-server-rounds"]} rounds')
    flwr.common.logger.FLOWER_LOGGER.info(f'SecAgg+ config: {secagg_config.num_shares} shares\n'
                                          f'{secagg_config.reconstruction_threshold} reconstruction threshold')
    workflow(driver, context)