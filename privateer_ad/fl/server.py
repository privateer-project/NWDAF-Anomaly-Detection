import mlflow
import torch

from flwr.common import Context
from flwr.server import Driver, LegacyContext, ServerApp, ServerConfig
from flwr.server.workflow import SecAggPlusWorkflow, DefaultWorkflow

from privateer_ad import logger
from privateer_ad.config import FederatedLearningConfig, MLFlowConfig, PathConfig, TrainingConfig
from privateer_ad.etl import DataProcessor
from privateer_ad.evaluate import ModelEvaluator
from privateer_ad.fl.strategy import CustomStrategy
from privateer_ad.utils import log_model

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
    mlflow_config = MLFlowConfig()
    _, server_run_id = setup_mlflow(experiment_name=mlflow_config.experiment_name,
                                    server_address=mlflow_config.server_address)
    mlflow_config.server_run_id = server_run_id

    fl_config = FederatedLearningConfig()
    training_config = TrainingConfig()

    # Get run parameters from context
    fl_config.n_clients = context.run_config.get('n-clients', fl_config.n_clients)
    fl_config.num_rounds = context.run_config.get('num-server-rounds', fl_config.num_rounds)

    logger.info(f'Federated Learning will run for {fl_config.num_rounds} rounds on {fl_config.n_clients} clients')
    logger.info(f'Secure aggregation: {fl_config.secure_aggregation_enabled}')
    logger.info(f'Server MLflow run ID: {server_run_id}')

    # Log federated learning parameters to parent run
    mlflow.log_params(fl_config.__dict__)
    mlflow.log_params({'session_type': 'federated_learning'})

    data_processor = DataProcessor()
    test_dl = data_processor.get_dataloader('test', only_benign=False)

    strategy = CustomStrategy(training_config=training_config, mlflow_config=mlflow_config)

    # Setup and run FL
    server_context = LegacyContext(
        context=context,
        config=ServerConfig(num_rounds=fl_config.num_rounds),
        strategy=strategy
    )

    if fl_config.secure_aggregation_enabled:
        workflow = DefaultWorkflow(
            fit_workflow=SecAggPlusWorkflow(
                num_shares=fl_config.num_shares,
                reconstruction_threshold=fl_config.reconstruction_threshold,
                timeout=fl_config.secagg_timeout,
                max_weight=fl_config.secagg_max_weight,
            )
        )
    else:
        workflow = DefaultWorkflow()
    try:
        logger.info('Starting federated learning...')
        logger.info(f'Server Context:\n{server_context}')
        workflow(driver, server_context)
        logger.info('Federated learning completed')
    finally:
        logger.info("Performing final evaluation on the global test set...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        evaluator = ModelEvaluator(device=device)

        metrics, _ = evaluator.evaluate(strategy.model, test_dl, prefix='global_test', step=fl_config.num_rounds)
        logger.info(f"Final global evaluation metrics: {metrics}")

        log_model(model=strategy.model,
                  model_name='global_TransformerAD',
                  dataloader=test_dl,
                  direction=strategy.training_config.direction,
                  target_metric=strategy.training_config.target_metric,
                  current_target_metric=metrics[strategy.training_config.target_metric],
                  experiment_id=mlflow.get_experiment_by_name(mlflow_config.experiment_name).experiment_id,
                  pip_requirements=PathConfig().requirements_file.as_posix())

        logger.info("Final model logged to MLFlow successfully")
        # End parent run
        if mlflow.active_run():
            mlflow.end_run()


def setup_mlflow(experiment_name, server_address):
    """Setup MLFlow tracking if enabled."""
    try:
        mlflow.set_tracking_uri(server_address)
        mlflow.set_experiment(experiment_name)
        if mlflow.active_run():
            logger.info(
                f"Found active run: {mlflow.active_run().info.run_name} - "
                f"{mlflow.active_run().info.run_id}, ending it"
            )
            mlflow.end_run()

        mlflow.start_run()
        run_name = mlflow.active_run().info.run_name
        run_id = mlflow.active_run().info.run_id
        logger.info(f"MLFlow run started: name: {run_name} (ID: {run_id})")
        return run_name, run_id
    except Exception as e:
        logger.error(f"Failed to setup MLFlow: {e}")
        return None, None
