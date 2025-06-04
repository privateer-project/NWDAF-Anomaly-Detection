import logging

import mlflow
import torch

from flwr.common import Context
from flwr.server import Grid, LegacyContext, ServerApp, ServerConfig
from flwr.server.workflow import SecAggPlusWorkflow, DefaultWorkflow

from privateer_ad.config import FederatedLearningConfig, MLFlowConfig, PathConfig, TrainingConfig
from privateer_ad.etl import DataProcessor
from privateer_ad.evaluate import ModelEvaluator
from privateer_ad.fl.strategy import CustomStrategy
from privateer_ad.utils import log_model

# Flower ServerApp
app = ServerApp()

@app.main()
def main(grid: Grid, context: Context) -> None:
    """
    Main server application.

    Args:
        grid: Flower Grid instance
        context: Flower context containing run configuration
    """
    mlflow_config = MLFlowConfig()
    _, server_run_id = setup_mlflow(experiment_name=mlflow_config.experiment_name,
                                    tracking_uri=mlflow_config.tracking_uri)
    mlflow_config.parent_run_id = server_run_id

    fl_config = FederatedLearningConfig()
    training_config = TrainingConfig()

    # Get run parameters from context
    fl_config.n_clients = context.run_config.get('n-clients', fl_config.n_clients)
    fl_config.num_rounds = context.run_config.get('num-server-rounds', fl_config.num_rounds)

    logging.info(f'Federated Learning will run for {fl_config.num_rounds} rounds on {fl_config.n_clients} clients')
    logging.info(f'Secure aggregation: {fl_config.secure_aggregation_enabled}')
    logging.info(f'Server MLflow run ID: {server_run_id}')

    # Log federated learning parameters to parent run
    mlflow.log_params(fl_config.__dict__)
    mlflow.log_params({'session_type': 'federated_learning'})

    dp = DataProcessor()

    test_dl = dp.get_dataloader('test', only_benign=False, train=False)
    sample = next(iter(test_dl))[0]['encoder_cont'][:1].to('cpu')
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
        logging.info('Starting federated learning...')
        logging.info(f'Server Context:\n{server_context}')
        workflow(grid, server_context)
        logging.info('Federated learning completed')
    finally:
        logging.info("Performing final evaluation on the global test set...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        evaluator = ModelEvaluator(device=device)

        metrics, _ = evaluator.evaluate(strategy.model, test_dl, prefix='global_test', step=fl_config.num_rounds)
        logging.info(f"Final global evaluation metrics: {metrics}")

        log_model(model=strategy.model,
                  model_name='global_TransformerAD',
                  sample=sample,
                  direction=strategy.training_config.direction,
                  target_metric=strategy.training_config.target_metric,
                  current_target_metric=metrics[strategy.training_config.target_metric],
                  experiment_id=mlflow.get_experiment_by_name(mlflow_config.experiment_name).experiment_id,
                  pip_requirements=PathConfig().requirements_file.as_posix())

        logging.info("Final model logged to MLFlow successfully")
        # End parent run
        if mlflow.active_run():
            mlflow.end_run()


def setup_mlflow(experiment_name, tracking_uri):
    """Setup MLFlow tracking if enabled."""
    try:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        if mlflow.active_run():
            logging.info(
                f"Found active run: {mlflow.active_run().info.run_name} - "
                f"{mlflow.active_run().info.run_id}, ending it"
            )
            mlflow.end_run()

        mlflow.start_run()
        run_name = mlflow.active_run().info.run_name
        run_id = mlflow.active_run().info.run_id
        logging.info(f"MLFlow run started: name: {run_name} (ID: {run_id})")
        return run_name, run_id
    except Exception as e:
        logging.error(f"Failed to setup MLFlow: {e}")
        return None, None
