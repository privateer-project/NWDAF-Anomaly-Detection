"""privateer-ad: A Flower / PyTorch app."""
from dataclasses import dataclass
from typing import Tuple, List

from flwr.common import Context, Metrics
from flwr.server import ServerApp, ServerConfig, LegacyContext, Driver
from flwr.server.strategy import FedAvg, DifferentialPrivacyClientSideFixedClipping
from flwr.server.workflow import SecAggPlusWorkflow, DefaultWorkflow

from src.config import DifferentialPrivacyConfig


@dataclass
class SecureAggregationConfig:
   num_shares: int = 3
   reconstruction_threshold: int = 2
   max_weight: float = 1000.0
   clipping_range: float = 8.0
   quantization_range: int = 4194304
   modulus_range: int = 4294967296
   timeout: float = 600.0

@dataclass
class FlowerConfig:
   enable: bool = False
   mode: str = "client"
   num_classes_per_partition: int = 9
   client_id: int = 0
   n_clients: int = 1
   server_address: str = "[::]:8081"
   num_rounds: int = 3
   num_clients_per_round: int = 2
   min_clients_per_round: int = 2
   fraction_fit: float = 1.0
   fraction_evaluate: float = 1.0

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
