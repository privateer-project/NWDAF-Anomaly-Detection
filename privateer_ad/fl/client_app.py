from flwr.common import Context
from flwr.client import ClientApp, NumPyClient
from flwr.client.mod import secaggplus_mod

from privateer_ad import logger
from privateer_ad.config import DPConfig
from privateer_ad.fl.utils import set_weights, get_weights
from privateer_ad.train.train import TrainPipeline

class SecAggFlowerClient(NumPyClient):
    """Flower client implementation with secure aggregation for anomaly detection."""

    def __init__(self, epochs, partition_id, partition=False, dp=False):
        self.dp_config = DPConfig()
        self.partition_id = partition_id
        self.epochs = epochs
        self.train_pln = TrainPipeline(run_name=f'client-{partition_id}',
                                       partition_id=self.partition_id,
                                       partition=partition,
                                       dp=dp,
                                       nested=True)
        self.train_pln.hparams.epochs = self.epochs
        logger.info(f'Client - {self.partition_id}')

    def fit(self, parameters, config):
        """Train model on local data with secure aggregation."""
        logger.info(f'\nClient {self.partition_id}')
        # Set received parameters
        set_weights(self.train_pln.model, parameters)

        # Perform training
        best_ckpt = self.train_pln.train_model(start_epoch=(config['server_round'] - 1) * self.epochs)

        weights = get_weights(self.train_pln.model)
        num_examples = sum(len(batch[0]['encoder_cont']) for batch in self.train_pln.train_dl)
        metrics = best_ckpt['metrics']

        # Log
        logger.info(f'\nClient {self.partition_id}\n')
        [logger.info(f'{k}: {v}') for k, v in metrics.items()]
        logger.info(f'Number of examples: {num_examples}')
        return weights, num_examples, metrics

    def evaluate(self, parameters, config):
        """Evaluate model on local test data."""
        logger.info(f'Client {self.partition_id}\n'
                                   f'evaluate config: {config}')

        set_weights(self.train_pln.model, parameters)
        # Evaluate the model
        metrics = self.train_pln.evaluate_model(step=config['server_round'])
        # Calculate number of evaluation samples
        num_examples = sum(len(batch[0]['encoder_cont']) for batch in self.train_pln.test_dl)

        logger.info(f'Client {self.partition_id}: Evaluation completed with metrics: {metrics}')

        loss = metrics.pop('eval_loss')
        return float(loss), num_examples, metrics

def client_fn(context: Context):
    """Create and return a Flower client."""
    # Initialize SecAgg client
    try:
        partition_id = int(context.node_config['partition-id'])
    except KeyError:
        partition_id = 0
    return SecAggFlowerClient(epochs=int(context.run_config['epochs']),
                              partition_id=partition_id,
                              partition =context.run_config['partition'],
                              dp =DPConfig().enable).to_client()

# Create Flower ClientApp
app = ClientApp(client_fn, mods=[secaggplus_mod])
