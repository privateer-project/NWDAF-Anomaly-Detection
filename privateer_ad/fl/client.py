from flwr.common import Context
from flwr.common.config import ConfigsRecord
from flwr.client import ClientApp, NumPyClient
from flwr.client.mod import secaggplus_mod

from privateer_ad import logger
from privateer_ad.fl.utils import set_weights, get_weights
from privateer_ad.train import TrainPipeline



class SecAggFlowerClient(NumPyClient):
    """Flower client implementation with secure aggregation for anomaly detection."""

    def __init__(self,context: Context):
        """Initialize client."""
        self.client_state = context.state
        if 'run_ids' not in self.client_state.configs_records:
            self.client_state.configs_records['run_ids'] = ConfigsRecord()
        else:
            logger.info(f'Found existing run IDs in client state: {self.client_state.configs_records['run_ids']}')
        self.partition_id = context.node_config['partition-id']
        self.dp_enabled = context.run_config['dp-enabled']
        self.epochs = int(context.run_config.get('epochs'))


        logger.info(f'Client {self.partition_id} initialized')

    def fit(self, parameters, config):
        """Train model."""
        server_round = config.get('server_round')
        server_run_id = config.get('server_run_id')
        logger.info(f'Server run id {server_run_id}')
        run_ids = self.client_state.configs_records['run_ids']
        if 'server_run_id' not in run_ids:
            run_ids['server_run_id'] = server_run_id

        train_pln = TrainPipeline(
            partition_id=self.partition_id,
            partition=True,
            dp_enabled=self.dp_enabled,
            run_id=run_ids.get('run_id', None),
            parent_run_id=run_ids['server_run_id'],
            config_overrides={'training.epochs': self.epochs}
        )
        if 'run_id' not in run_ids:
            run_ids['run_id'] = train_pln.run_id

        logger.info(f'Client {self.partition_id} - Round {server_round} - Server Run ID: {server_run_id}')


        set_weights(train_pln.model, parameters)
        start_epoch = (server_round - 1) * train_pln.training_config.epochs
        best_ckpt = train_pln.train_model(start_epoch=start_epoch)

        # Get results
        weights = get_weights(train_pln.model)
        num_examples = sum(len(batch[0]['encoder_cont']) for batch in train_pln.train_dl)
        metrics = best_ckpt['metrics'].copy()

        return weights, num_examples, metrics

    def evaluate(self, parameters, config):
        """Evaluate model."""
        server_round = config.get('server_round')
        server_run_id = config.get('server_run_id')
        logger.info(f'Server run id {server_run_id}')
        run_ids = self.client_state.configs_records['run_ids']
        if 'server_run_id' not in run_ids:
            run_ids['server_run_id'] = server_run_id

        logger.info(f'Client {self.partition_id} - Evaluation Round {server_round}')
        train_pln = TrainPipeline(
            partition_id=self.partition_id,
            partition=True,
            dp_enabled=self.dp_enabled,
            run_id=run_ids.get('run_id', None),
            parent_run_id=run_ids['server_run_id'],
            config_overrides={'training.epochs': self.epochs}
        )
        if 'run_id' not in run_ids:
            run_ids['run_id'] = train_pln.run_id

        # Evaluate (TrainingPipeline handles MLFlow)
        set_weights(train_pln.model, parameters)
        metrics = train_pln.evaluate_model(step=server_round)
        num_examples = sum(len(batch[0]['encoder_cont']) for batch in train_pln.test_dl)

        loss = metrics.pop('eval_loss', 0.0)
        return float(loss), num_examples, metrics

def client_fn(context: Context):
    """Create client."""
    logger.info(f'Client Context:\n{context}.')
    return SecAggFlowerClient(context).to_client()


app = ClientApp(client_fn, mods=[secaggplus_mod])