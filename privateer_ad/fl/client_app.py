from collections import OrderedDict

import flwr.common.logger
import torch
from flwr.common.context import Context
from flwr.client import ClientApp, NumPyClient
from flwr.client.mod import secaggplus_mod, fixedclipping_mod
from privateer_ad.train.train import TrainPipeline


def set_weights(net, parameters):
    """Set model parameters from a list of NumPy arrays."""
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)

def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

class SecAggFlowerClient(NumPyClient):
    """Flower client implementation with secure aggregation for anomaly detection."""

    def __init__(self, context):
        self.context = context
        self.partition_id = int(self.context.node_config['partition-id'])
        self.num_partitions = int(self.context.node_config['num-partitions'])

        flwr.common.logger.FLOWER_LOGGER.info(f'Client - {self.partition_id} of {self.num_partitions}')

        # Create a unique run name for this client

        self.train_pln = TrainPipeline(run_name=f'client-{self.partition_id}',
                                       partition_id=self.partition_id,
                                       num_partitions=self.num_partitions,
                                       nested=True)
        self.train_pln.hparams.epochs = int(self.context.run_config['epochs'])

    def fit(self, parameters, config):
        """Train model on local data with secure aggregation."""
        self.train_pln.logger.info(f'\n'
                                   f'Client {self.partition_id}\n'
                                   f'fit config: {config}')
        # Set received parameters
        set_weights(self.train_pln.model, parameters)

        # Perform training
        best_ckpt = self.train_pln.train_model(start_epoch=(config['server_round'] - 1) * self.context.run_config['epochs'])
        num_examples = sum(len(batch[0]['encoder_cont']) for batch in self.train_pln.train_dl)

        self.train_pln.logger.info(f'\n'
                                   f'Client {self.partition_id}\n'
                                   f'Training completed: {best_ckpt['metrics']}\n'
                                   f'Number of examples: {num_examples}')

        return get_weights(self.train_pln.model), num_examples, best_ckpt['metrics']

    def evaluate(self, parameters, config):
        """Evaluate model on local test data."""
        self.train_pln.logger.info(f'\n'
                                   f'Client {self.partition_id}\n'
                                   f'evaluate config: {config}')

        set_weights(self.train_pln.model, parameters)
        # Evaluate the model
        report = self.train_pln.evaluate_model(step=config['server_round'])
        loss = report.pop('eval_loss')
        # Calculate number of evaluation samples
        num_examples = sum(len(batch[0]['encoder_cont']) for batch in self.train_pln.test_dl)

        self.train_pln.logger.info(f'Client {self.partition_id}: Evaluation completed with metrics: {report}')
        return float(loss), num_examples, report

def client_fn(context: Context):
    """Create and return a Flower client."""
    # Initialize SecAgg client
    return SecAggFlowerClient(context=context).to_client()

# Create Flower ClientApp
app = ClientApp(client_fn, mods=[secaggplus_mod, fixedclipping_mod])
