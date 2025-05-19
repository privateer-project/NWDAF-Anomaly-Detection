import mlflow

from flwr.common import Context
from flwr.server import Driver, LegacyContext, ServerApp, ServerConfig
from flwr.server.workflow import SecAggPlusWorkflow, DefaultWorkflow

from privateer_ad.config import SecureAggregationConfig, DifferentialPrivacyConfig
from privateer_ad.fl.strategy import CustomStrategy

# Flower ServerApp
app = ServerApp()

@app.main()
def main(driver: Driver, context: Context) -> None:
    secaggr_cfg = SecureAggregationConfig()
    # todo set configs to context
    dp_cfg = DifferentialPrivacyConfig()
    n_clients = context.run_config.get('n-clients')
    num_rounds = context.run_config.get('num-server-rounds')

    strategy = CustomStrategy(noise_multiplier=dp_cfg.noise_multiplier,
                              clipping_norm=dp_cfg.server_clipping_norm,
                              num_sampled_clients=n_clients)
    if strategy.mlflow_config.track:
        with mlflow.start_run(run_id=strategy.run_id):
            mlflow.log_params({'num_clients': n_clients,
                               'num_rounds': num_rounds})

    workflow = DefaultWorkflow(fit_workflow=SecAggPlusWorkflow(num_shares=secaggr_cfg.num_shares,
                                                               reconstruction_threshold=secaggr_cfg.reconstruction_threshold,
                                                               max_weight=200000))
    # Execute workflow
    strategy.logger.info(f'Server will run for {num_rounds} rounds')
    strategy.logger.info(f'SecAgg+ config: num_shares: {secaggr_cfg.num_shares} reconstruction threshold: {secaggr_cfg.reconstruction_threshold}')

    context = LegacyContext(context=context,
                            config=ServerConfig(num_rounds=num_rounds),
                            strategy=strategy)
    strategy.logger.info(f'Main Context: {context}')
    workflow(driver, context)
