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
    n_clients = context.run_config.get('n_clients')
    num_rounds = context.run_config.get('num-server-rounds')
    run_name = context.run_config.get('run-name')

    strategy = CustomStrategy(noise_multiplier=dp_cfg.noise_multiplier,
                              clipping_norm=dp_cfg.max_grad_norm,
                              num_sampled_clients=n_clients,
                              run_name=run_name)
    if strategy.parent_run_id:
        mlflow.log_params({'num_clients': n_clients,
                           'num_rounds': num_rounds},
                          run_id=strategy.parent_run_id)

    workflow = DefaultWorkflow(fit_workflow=SecAggPlusWorkflow(num_shares=secaggr_cfg.num_shares,
                                                               reconstruction_threshold=secaggr_cfg.reconstruction_threshold))
    # Execute workflow
    strategy.logger.info(f'Server will run for {num_rounds} rounds')
    strategy.logger.info(f'SecAgg+ config: num_shares: {secaggr_cfg.num_shares} reconstruction threshold: {secaggr_cfg.reconstruction_threshold}')

    context = LegacyContext(context=context,
                            config=ServerConfig(num_rounds=num_rounds),
                            strategy=strategy)
    strategy.logger.info(f'Main Context: {context}')
    workflow(driver, context)
