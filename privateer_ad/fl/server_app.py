import mlflow

from flwr.common import Context
from flwr.server import Driver, LegacyContext, ServerApp, ServerConfig
from flwr.server.workflow import SecAggPlusWorkflow, DefaultWorkflow

from privateer_ad import logger
from privateer_ad.config import SecureAggregationConfig
from privateer_ad.fl.strategy import CustomStrategy

# Flower ServerApp
app = ServerApp()

@app.main()
def main(driver: Driver, context: Context) -> None:
    n_clients = context.run_config.get('n-clients')
    num_rounds = context.run_config.get('num-server-rounds')
    logger.info(f'FL will run for {num_rounds} rounds on {n_clients} clients')

    secaggr_cfg = SecureAggregationConfig().__dict__

    strategy = CustomStrategy(num_rounds=num_rounds)
    if mlflow.active_run():
        mlflow.log_params({'num_clients': n_clients,
                           'num_rounds': num_rounds})

    # Execute workflow
    context = LegacyContext(context=context,
                            config=ServerConfig(num_rounds=num_rounds),
                            strategy=strategy)
    logger.info(f'Main Context: {context}')
    workflow = DefaultWorkflow(fit_workflow=SecAggPlusWorkflow(max_weight=200000, **secaggr_cfg,
                                                               ))
    workflow(driver, context)
