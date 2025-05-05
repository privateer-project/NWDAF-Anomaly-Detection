import logging
import mlflow

from flwr.common.logger import FLOWER_LOGGER
from flwr.common import Context
from flwr.server import Driver, LegacyContext, ServerApp, ServerConfig
from flwr.server.workflow import SecAggPlusWorkflow, DefaultWorkflow

from privateer_ad.config import SecureAggregationConfig, DifferentialPrivacyConfig
from privateer_ad.fl.strategy import CustomStrategy
import ray


ray.logger.setLevel(logging.WARNING)
FLOWER_LOGGER.setLevel(logging.WARNING)

# Flower ServerApp
app = ServerApp()

@app.main()
def main(driver: Driver, context: Context) -> None:


    secaggr_cfg = SecureAggregationConfig()
    # todo set configs to context
    dp_cfg = DifferentialPrivacyConfig()
    n_clients = context.run_config.get('n_clients')
    num_rounds = context.run_config.get('num-server-rounds')
    run_name = context.run_config.get('run-name')

    strategy = CustomStrategy(noise_multiplier=dp_cfg.noise_multiplier,
                              clipping_norm=dp_cfg.max_grad_norm,
                              num_sampled_clients=n_clients,
                              run_name=run_name)
    if strategy.mlflow_config.track:
        with mlflow.start_run(run_id=strategy.parent_run_id):
            mlflow.log_param('num_clients', n_clients)
            mlflow.log_param('num_rounds', num_rounds)


    workflow = DefaultWorkflow(fit_workflow=SecAggPlusWorkflow(num_shares=secaggr_cfg.num_shares,
                                                               reconstruction_threshold=secaggr_cfg.reconstruction_threshold))
    # Execute workflow
    FLOWER_LOGGER.info(f'Server will run for {num_rounds} rounds')
    FLOWER_LOGGER.info(f'SecAgg+ config: num_shares: {secaggr_cfg.num_shares} reconstruction threshold: {secaggr_cfg.reconstruction_threshold}')

    context = LegacyContext(context=context,
                            config=ServerConfig(num_rounds=num_rounds),
                            strategy=strategy)
    FLOWER_LOGGER.info(f'Main Context: {context}')
    workflow(driver, context)
