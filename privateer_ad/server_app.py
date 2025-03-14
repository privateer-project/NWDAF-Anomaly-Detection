"""privateer_ad: A Flower / PyTorch app."""
from typing import Tuple, List

import flwr.common.logger
import mlflow
from flwr.common import Context, Metrics, ndarrays_to_parameters
from flwr.server import ServerApp, ServerConfig, LegacyContext, Driver
from flwr.server.strategy import FedAvg, DifferentialPrivacyClientSideFixedClipping
from flwr.server.workflow import SecAggPlusWorkflow, DefaultWorkflow

from privateer_ad.config import DifferentialPrivacyConfig, SecureAggregationConfig, HParams, MLFlowConfig, \
    AttentionAutoencoderConfig
from privateer_ad.models import AttentionAutoencoder
from privateer_ad.utils import set_config


def make_config():
    config = {}
    config.update(HParams().__dict__)
    config.update(AttentionAutoencoderConfig().__dict__)
    config.update(DifferentialPrivacyConfig().__dict__)
    config.update(SecureAggregationConfig().__dict__)
    config.update(MLFlowConfig().__dict__)
    return config

def on_fit_config(server_round: int):
    """Generate training configuration for each round."""
    # Create the configuration dictionary
    config = make_config()
    config["current_round"] = server_round
    return config

def on_evaluate_config(server_round: int):
    """Generate evaluation configuration for each round."""
    # Create the configuration dictionary
    config = make_config()
    config["current_round"] = server_round
    return config

def get_model_weights(model):
    """Get model weights as NDArrays."""
    return ndarrays_to_parameters([val.cpu().numpy() for _, val in model.state_dict().items()])

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    weighted_sums = {}
    total_examples = 0
    for num_examples, m in metrics:
        total_examples += num_examples
        for key, val in m.items():
            if key not in weighted_sums:
                weighted_sums[key] = 0
            weighted_sums[key] += val * num_examples
    weighted_metrics = {k: v / total_examples for k, v in weighted_sums.items()}
    return weighted_metrics

# Flower ServerApp
app = ServerApp()

@app.main()
def main(driver: Driver, context: Context) -> None:
    server_address: str = "[::]:8081"
    n_clients = context.run_config['n_clients']
    config = make_config()
    model_config = set_config(AttentionAutoencoderConfig, config)
    dp_config = set_config(DifferentialPrivacyConfig, config)
    secagg_config = set_config(SecureAggregationConfig, config)
    mlflow_config = set_config(MLFlowConfig, config)
    # Configure MLflow if enabled
    if mlflow_config.track:
        mlflow.set_tracking_uri(mlflow_config.server_address)
        mlflow.set_experiment(mlflow_config.experiment_name)
        mlflow.start_run(run_name=f"federated_{context.run_config['num-server-rounds']}_rounds")

    # Get initial parameters from the model
    model = AttentionAutoencoder(model_config)
    initial_parameters = get_model_weights(model)

    # Define strategy
    strategy = FedAvg(
        on_fit_config_fn=on_fit_config,
        on_evaluate_config_fn=on_evaluate_config,
        min_fit_clients=n_clients,
        min_evaluate_clients=n_clients,
        min_available_clients=n_clients,
        evaluate_metrics_aggregation_fn=weighted_average,
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

    flwr.common.logger.FLOWER_LOGGER.info(f'Starting Flower server with SecAgg+ at {server_address}')
    flwr.common.logger.FLOWER_LOGGER.info(f'Server will run for {context.run_config['num-server-rounds']} rounds')
    flwr.common.logger.FLOWER_LOGGER.info(f'Minimum clients per round: {n_clients}')
    flwr.common.logger.FLOWER_LOGGER.info(f'SecAgg+ config: {secagg_config.num_shares} shares\n'
                                          f'{secagg_config.reconstruction_threshold} reconstruction threshold')
    workflow(driver, context)
