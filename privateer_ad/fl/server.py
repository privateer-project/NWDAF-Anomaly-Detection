import mlflow

from flwr.common import Context
from flwr.server import Driver, LegacyContext, ServerApp, ServerConfig
from flwr.server.workflow import SecAggPlusWorkflow, DefaultWorkflow

from privateer_ad import logger
from privateer_ad.config import get_fl_config, get_mlflow_config
from .strategy import CustomStrategy

# Flower ServerApp
app = ServerApp()

@app.main()
def main(driver: Driver, context: Context) -> None:
    """
    Main server application.

    Args:
        driver: Flower driver instance
        context: Flower context containing run configuration
    """
    fl_config = get_fl_config()
    mlflow_config = get_mlflow_config()

    # Get run parameters from context
    n_clients = context.run_config.get('n-clients', fl_config.min_clients)
    num_rounds = context.run_config.get('num-server-rounds', fl_config.num_rounds)
    dp_enabled = context.run_config.get('dp-enabled', None)

    logger.info(f'FL will run for {num_rounds} rounds on {n_clients} clients')
    logger.info(f'Secure aggregation: {fl_config.secure_aggregation_enabled}')

    strategy = CustomStrategy(num_rounds=num_rounds, dp_enabled=dp_enabled)

    # Log federated learning parameters to MLFlow if enabled
    if mlflow.active_run() and mlflow_config.enabled:
        mlflow.log_params({
            'num_clients': n_clients,
            'num_rounds': num_rounds,
            'secure_aggregation': fl_config.secure_aggregation_enabled,
            'fraction_fit': fl_config.fraction_fit,
            'fraction_evaluate': fl_config.fraction_evaluate,
            'min_clients': fl_config.min_clients,
            'epochs_per_round': fl_config.epochs_per_round,
            'partition_data': fl_config.partition_data
        })

    # Setup server context
    server_context = LegacyContext(
        context=context,
        config=ServerConfig(num_rounds=num_rounds),
        strategy=strategy
    )

    logger.info(f'Server Context: {server_context}')

    # Setup workflow based on secure aggregation setting
    if fl_config.secure_aggregation_enabled:
        logger.info('Using SecAggPlus workflow for secure aggregation')

        # SecAgg+ configuration
        secagg_config = {
            'num_shares': fl_config.num_shares,
            'reconstruction_threshold': fl_config.reconstruction_threshold,
            'timeout': fl_config.secagg_timeout,
            'max_weight': fl_config.secagg_max_weight,
        }

        # Log SecAgg parameters
        if mlflow.active_run() and mlflow_config.enabled:
            mlflow.log_params(secagg_config)

        workflow = DefaultWorkflow(
            fit_workflow=SecAggPlusWorkflow(**secagg_config)
        )
    else:
        logger.info('Using default workflow (no secure aggregation)')
        workflow = DefaultWorkflow()

    # Log strategy summary
    strategy_summary = strategy.get_strategy_summary()
    logger.info(f'Strategy summary: {strategy_summary}')

    if mlflow.active_run() and mlflow_config.enabled:
        mlflow.log_params({
            'strategy_type': strategy_summary['strategy_type'],
            'model_architecture': str(strategy_summary['model_config']),
            'training_params': str(strategy_summary['training_config'])
        })

    # Execute federated learning workflow
    try:
        logger.info('Starting federated learning workflow...')
        workflow(driver, server_context)
        logger.info('Federated learning workflow completed successfully')

    except Exception as e:
        logger.error(f'Federated learning workflow failed: {e}')
        raise

    finally:
        # Ensure proper cleanup
        if mlflow.active_run() and mlflow_config.enabled:
            logger.info('Ending MLFlow run')
            mlflow.end_run()
