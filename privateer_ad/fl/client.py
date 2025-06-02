from flwr.common import Context
from flwr.common.config import ConfigsRecord
from flwr.client import ClientApp, NumPyClient
from flwr.client.mod import secaggplus_mod

from privateer_ad import logger
from privateer_ad.config import DataConfig, TrainingConfig, MLFlowConfig
from privateer_ad.fl.utils import set_weights, get_weights
from privateer_ad.train import TrainPipeline


class SecAggFlowerClient(NumPyClient):
    """Flower client implementation with secure aggregation for anomaly detection."""

    def __init__(self,context: Context):
        """Initialize client."""
        self.client_state = context.state
        self.mlflow_conf = MLFlowConfig()
        self.train_conf = TrainingConfig()
        self.data_conf = DataConfig()

        if 'run_ids' not in self.client_state.configs_records:
            self.client_state.configs_records['run_ids'] = ConfigsRecord()

        self.train_conf.epochs = int(context.run_config.get('epochs'))
        self.data_conf.partition_id = context.node_config['partition-id']
        logger.info(f'Client {self.data_conf.partition_id} initialized')

    def fit(self, parameters, config):
        """Train model."""
        server_round = config.get('server_round')
        self.client_state.configs_records['run_ids'].update({'server_run_id': config.get('server_run_id', None)})
        self.mlflow_conf.server_run_id = self.client_state.configs_records['run_ids'].get('server_run_id', None)
        self.mlflow_conf.client_run_id = self.client_state.configs_records['run_ids'].get('client_run_id', None)
        logger.info(f'Server run id {self.mlflow_conf.server_run_id}')

        train_pln = TrainPipeline(data_config=self.data_conf, training_config=self.train_conf, mlflow_config=self.mlflow_conf)
        self.client_state.configs_records['run_ids'].update({'client_run_id': train_pln.mlflow_config.client_run_id})

        logger.info(f'Round {server_round} '
                    f'- Client {train_pln.data_config.partition_id} '
                    f'- Server Run ID: {train_pln.mlflow_config.server_run_id} '
                    f'- Client Run ID: {train_pln.mlflow_config.client_run_id}')

        set_weights(train_pln.model, parameters)
        start_epoch = (server_round - 1) * train_pln.training_config.epochs

        # Get results
        best_ckpt = train_pln.train_model(start_epoch=start_epoch)
        weights = get_weights(train_pln.model)
        num_examples = sum(len(batch[0]['encoder_cont']) for batch in train_pln.train_dl)
        return weights, num_examples, best_ckpt['metrics']

    def evaluate(self, parameters, config):
        """Evaluate model."""
        server_round = config.get('server_round')
        server_run_id = config.get('server_run_id')
        run_ids = self.client_state.configs_records['run_ids']
        if 'server_run_id' not in run_ids:
            run_ids['server_run_id'] = server_run_id

        logger.info(f'Client {self.data_conf.partition_id} - Evaluation Round {server_round}')
        self.client_state.configs_records['run_ids'].update({'server_run_id': config.get('server_run_id', None)})
        self.mlflow_conf.server_run_id = self.client_state.configs_records['run_ids'].get('server_run_id', None)
        self.mlflow_conf.client_run_id = self.client_state.configs_records['run_ids'].get('client_run_id', None)

        train_pln = TrainPipeline(data_config=self.data_conf, training_config=self.train_conf, mlflow_config=self.mlflow_conf)
        self.client_state.configs_records['run_ids'].update({'client_run_id': train_pln.mlflow_config.client_run_id})

        set_weights(train_pln.model, parameters)
        metrics, figs = train_pln.evaluate_model(step=server_round)
        num_examples = sum(len(batch[0]['encoder_cont']) for batch in train_pln.test_dl)

        loss = metrics.pop('test_loss')
        return float(loss), num_examples, metrics

def client_fn(context: Context):
    """Create client."""
    logger.info(f'Client Context:\n{context}.')
    return SecAggFlowerClient(context).to_client()


app = ClientApp(client_fn, mods=[secaggplus_mod])