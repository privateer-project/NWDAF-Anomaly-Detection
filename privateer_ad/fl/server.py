import mlflow
import torch

from flwr.common import Context
from flwr.server import Driver, LegacyContext, ServerApp, ServerConfig
from flwr.server.workflow import SecAggPlusWorkflow, DefaultWorkflow

from privateer_ad import logger
from privateer_ad.config import get_fl_config, get_mlflow_config, get_training_config, get_model_config, get_paths
from privateer_ad.etl import DataProcessor
from privateer_ad.evaluate import ModelEvaluator
from privateer_ad.fl.strategy import CustomStrategy


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
    _, server_run_id = _setup_mlflow()
    fl_config = get_fl_config()
    # Get run parameters from context
    fl_config.n_clients = context.run_config.get('n-clients', fl_config.n_clients)
    fl_config.num_rounds = context.run_config.get('num-server-rounds', fl_config.num_rounds)
    dp_enabled = context.run_config.get('dp-enabled', None)
    context.run_config['server_run_id'] =  server_run_id

    logger.info(f'Federated Learning will run for {fl_config.num_rounds} rounds on {fl_config.n_clients} clients')
    logger.info(f'Secure aggregation: {fl_config.secure_aggregation_enabled}')
    logger.info(f'Server MLflow run ID: {server_run_id}')

    # Log federated learning parameters to parent run
    mlflow.log_params(fl_config.__dict__)
    logger.info({
        'dp_enabled': dp_enabled,
        'session_type': 'federated_learning'
    })
    mlflow.log_params({
        'dp_enabled': dp_enabled,
        'session_type': 'federated_learning'
    })

    data_processor = DataProcessor(partition=False)
    test_dl = data_processor.get_dataloader('test',
                                            batch_size=1024,
                                            seq_len=get_model_config().seq_len,
                                            only_benign=False
                                            )
    sample = next(iter(test_dl))[0]['encoder_cont'][:1].to('cpu')

    strategy = CustomStrategy(
        input_size=sample.shape[-1],
        server_run_id=server_run_id
    )

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
        _final_evaluation(model=strategy.model, dataloader=test_dl, server_round=fl_config.num_rounds)
        # End parent run
        if mlflow.active_run():
            mlflow.end_run()


def _setup_mlflow():
    """Setup MLFlow tracking if enabled."""
    try:
        mlflow_config = get_mlflow_config()
        mlflow.set_tracking_uri(mlflow_config.server_address)
        mlflow.set_experiment(mlflow_config.experiment_name)
        if mlflow.active_run():
            logger.info(
                f"Found active run: {mlflow.active_run().info.run_name} - "
                f"{mlflow.active_run().info.run_id}, ending it"
            )
            mlflow.end_run()

        mlflow.start_run()
        server_run_name = mlflow.active_run().info.run_name
        server_run_id = mlflow.active_run().info.run_id
        logger.info(f"MLFlow run started: name: {server_run_name} (ID: {server_run_id})")
        return server_run_name, server_run_id
    except Exception as e:
        logger.error(f"Failed to setup MLFlow: {e}")
        return None, None

def _final_evaluation(model, dataloader, server_round: int):
    """Perform final evaluation on the best model."""
    logger.info("Performing final evaluation on the global test set...")
    training_config = get_training_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    evaluator = ModelEvaluator(
        criterion=training_config.loss_function,
        device=device
    )

    model.to('cpu')

    metrics, figures = evaluator.evaluate(model, dataloader, prefix='global_test', step=server_round)
    logger.info(f"Final global evaluation metrics: {metrics}")

    sample_tensor = next(iter(dataloader))[0]['encoder_cont'][:1].to('cpu')

    model.to('cpu')
    _output = model(sample_tensor)

    # Convert to numpy AFTER model call
    _input_np = sample_tensor.detach().numpy()
    _output_np = _output.detach().numpy() if not isinstance(_output, dict) else {k: v.detach().numpy() for k, v in
                                                                                 _output.items()}
    pip_requirements = str(get_paths().root_dir.joinpath('requirements.txt'))

    mlflow.pytorch.log_model(pytorch_model=model,
                             artifact_path='final_global_model',
                             registered_model_name='privateer_global_model',
                             signature=mlflow.models.infer_signature(_input_np, _output_np),
                             pip_requirements=pip_requirements
                             )
    logger.info("Final model logged to MLFlow successfully")
