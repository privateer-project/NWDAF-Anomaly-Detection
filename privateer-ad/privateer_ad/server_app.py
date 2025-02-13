"""privateer-ad: A Flower / PyTorch app."""
from dataclasses import dataclass
from typing import Tuple, List

from flwr.common import Context, Metrics
from flwr.server import ServerApp, ServerConfig, LegacyContext, Driver
from flwr.server.strategy import FedAvg, DifferentialPrivacyClientSideFixedClipping
from flwr.server.workflow import SecAggPlusWorkflow, DefaultWorkflow

from src.config import DifferentialPrivacyConfig


def get_model_weights(model):
    """Get model weights as NDArrays."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]

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

# Create ServerApp
app = ServerApp()

@app.main()
def main(driver: Driver, context: Context) -> None:
    flwr_conf = FlowerConfig()
    # Get configuration values for server
    # Define strategy
    strategy = FedAvg(
        fraction_fit=context.run_config["fraction-fit"],
        evaluate_metrics_aggregation_fn=weighted_average)
    dp_conf = DifferentialPrivacyConfig()
    strategy = DifferentialPrivacyClientSideFixedClipping(
        strategy,
        noise_multiplier=dp_conf.noise_multiplier,
        clipping_norm=dp_conf.max_grad_norm,
        num_sampled_clients=flwr_conf.n_clients,
    )
    context.node_config['num-partitions'] = flwr_conf.n_clients

    # Create server configuration
    server_config = ServerConfig(num_rounds=5)
    context = LegacyContext(context=context,
                            config=server_config,
                            strategy=strategy)
    fit_workflow = SecAggPlusWorkflow(**SecureAggregationConfig().__dict__)
    workflow = DefaultWorkflow(fit_workflow=fit_workflow)
    workflow(driver, context)
