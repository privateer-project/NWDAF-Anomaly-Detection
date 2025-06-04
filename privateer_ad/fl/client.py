import logging

from flwr.common import Context
from flwr.common.config import ConfigRecord
from flwr.client import ClientApp, NumPyClient
from flwr.client.mod import secaggplus_mod

from privateer_ad.config import DataConfig, TrainingConfig, MLFlowConfig
from privateer_ad.fl.utils import set_weights, get_weights
from privateer_ad.train import TrainPipeline


class SecAggFlowerClient(NumPyClient):
    """Flower client implementation with secure aggregation for anomaly detection."""

    def __init__(self,context: Context):
        """Initialize client."""
        self.client_state = context.state
        self.mlflow_config = MLFlowConfig()
        self.training_config = TrainingConfig()
        self.data_config = DataConfig()
        if 'run_ids' not in self.client_state.config_records:
            self.client_state.config_records['run_ids'] = ConfigRecord()

        self.mlflow_config.parent_run_id = self.client_state.config_records['run_ids'].get('parent_run_id', None)
        self.mlflow_config.child_run_id = self.client_state.config_records['run_ids'].get('child_run_id', None)
        self.training_config.epochs = int(context.run_config.get('epochs'))
        self.data_config.partition_id = context.node_config['partition-id']
        self.data_config.num_partitions = context.node_config['num-partitions']
        logging.info(f'Client {self.data_config.partition_id} initialized')


    def fit(self, parameters, config):
        """Train model."""
        server_round = config.get('server_round')

        if self.mlflow_config.parent_run_id is None:
            self.mlflow_config.parent_run_id = config.get('server_run_id', None)
            self.client_state.config_records['run_ids'].update({'parent_run_id': self.mlflow_config.parent_run_id})
            logging.info(f'Server run id {self.mlflow_config.parent_run_id}')

        training_pipeline = TrainPipeline(data_config=self.data_config,
                                          training_config=self.training_config,
                                          mlflow_config=self.mlflow_config)
        if self.client_state.config_records['run_ids'].get('child_run_id', None) is None:
            self.mlflow_config.child_run_id = training_pipeline.mlflow_config.child_run_id
            self.client_state.config_records['run_ids']['child_run_id'] = self.mlflow_config.child_run_id

        logging.info(f'Round {server_round} '
                    f'- Client {self.data_config.partition_id} '
                    f'- Server Run ID: {self.mlflow_config.parent_run_id} '
                    f'- Client Run ID: {self.mlflow_config.child_run_id}')

        set_weights(training_pipeline.model, parameters)

        # Get results
        best_checkpoint = training_pipeline.train_model(start_epoch=self.training_config.epochs * (server_round - 1))

        weights = get_weights(training_pipeline.model)

        num_examples = sum(len(batch[0]['encoder_cont']) for batch in training_pipeline.val_dl)
        return weights, num_examples, best_checkpoint['metrics']

    def evaluate(self, parameters, config):
        """Evaluate model."""
        server_round = config.get('server_round')
        logging.info(f'Client {self.data_config.partition_id} - Evaluation Round {server_round}')
        training_pipeline = TrainPipeline(data_config=self.data_config,
                                          training_config=self.training_config,
                                          mlflow_config=self.mlflow_config)

        set_weights(training_pipeline.model, parameters)
        metrics, figs = training_pipeline.evaluate_model(step=server_round)
        num_examples = sum(len(batch[0]['encoder_cont']) for batch in training_pipeline.test_dl)
        print('eval metrics', metrics)
        loss = metrics.pop('test_loss')
        return float(loss), num_examples, metrics

def client_fn(context: Context):
    """Create client."""
    logging.info(f'Client run config:\n{context.run_config}.')
    return SecAggFlowerClient(context).to_client()


app = ClientApp(client_fn, mods=[secaggplus_mod])