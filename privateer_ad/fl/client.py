import mlflow
from flwr.common import Context
from flwr.client import ClientApp, NumPyClient
from flwr.client.mod import secaggplus_mod

from privateer_ad import logger
from privateer_ad.config import get_privacy_config, get_fl_config
from privateer_ad.fl.utils import set_weights, get_weights
from privateer_ad.train import TrainPipeline
from mlflow.entities import RunStatus

class SecAggFlowerClient(NumPyClient):
    """Flower client implementation with secure aggregation for anomaly detection."""

    def __init__(self, train_pln: TrainPipeline):
        """Initialize client."""
        self.privacy_config = get_privacy_config()
        self.fl_config = get_fl_config()
        self.train_pln = train_pln
        self.partition_id = self.train_pln.partition_id

        logger.info(f'Client {self.partition_id} initialized')

    def fit(self, parameters, config):
        """Train model."""
        server_round = config.get('server_round', 1)
        logger.info(f'Client {self.partition_id} - Round {server_round}')

        set_weights(self.train_pln.model, parameters)
        start_epoch = (server_round - 1) * self.train_pln.training_config.epochs
        best_ckpt = self.train_pln.train_model(start_epoch=start_epoch)

        # Get results
        weights = get_weights(self.train_pln.model)
        num_examples = sum(len(batch[0]['encoder_cont']) for batch in self.train_pln.train_dl)
        metrics = best_ckpt['metrics'].copy()

        return weights, num_examples, metrics

    def evaluate(self, parameters, config):
        """Evaluate model."""
        server_round = config.get('server_round', 0)
        logger.info(f'Client {self.partition_id} - Evaluation Round {server_round}')

        # Evaluate (TrainingPipeline handles MLFlow)
        set_weights(self.train_pln.model, parameters)
        metrics = self.train_pln.evaluate_model(step=server_round)
        num_examples = sum(len(batch[0]['encoder_cont']) for batch in self.train_pln.test_dl)

        loss = metrics.pop('eval_loss', 0.0)
        return float(loss), num_examples, metrics



def client_fn(context: Context):
    """Create client."""
    logger.info(f'Client Context:\n{context}.')
    if not context.run_config.get('client_run_id', None):
        logger.warning('Client run ID not found, starting new run.')
        if RunStatus.is_terminated(mlflow.get_run(context.run_config.get('server_run_id')).info.status):
            mlflow.start_run(parent_run_id=context.run_config.get('server_run_id'))
            context.run_config['client_run_id'] = mlflow.active_run().info.run_id
    else:
        logger.info(f'Client run ID found: {context.run_config.get("client_run_id")}')
        mlflow.start_run(run_id=context.run_config.get('client_run_id'))

    train_pln = TrainPipeline(
        partition_id=int(context.node_config.get('partition-id')),
        partition=context.run_config.get('partition'),
        dp_enabled=context.run_config.get('dp-enabled'),
        run_id=context.run_config['client_run_id'],
        config_overrides={'training.epochs': int(context.run_config.get('epochs'))}
    )
    return SecAggFlowerClient(train_pln).to_client()


app = ClientApp(client_fn, mods=[secaggplus_mod])