from flwr.common import Context
from flwr.client import ClientApp, NumPyClient
from flwr.client.mod import secaggplus_mod

from privateer_ad import logger
from privateer_ad.config import (get_privacy_config,
                                 get_fl_config
                                 )
from privateer_ad.fl.utils import set_weights, get_weights
from privateer_ad.train import TrainPipeline


class SecAggFlowerClient(NumPyClient):
    """Flower client implementation with secure aggregation for anomaly detection."""

    def __init__(self, epochs: int, partition_id: int, partition: bool = False, dp_enabled: bool = False):
        """
        Initialize the Flower client.

        Args:
            epochs: Number of training epochs per round
            partition_id: Client partition identifier
            partition: Whether to use data partitioning
            dp_enabled: Whether to enable differential privacy
        """

        self.privacy_config = get_privacy_config()
        self.fl_config = get_fl_config()

        # Override DP setting if explicitly provided
        self.dp_enabled = dp_enabled or self.privacy_config.dp_enabled

        # Store client parameters
        self.partition_id = partition_id
        self.partition = partition

        _overrides = None
        if epochs:
            _overrides = {'training.epochs': epochs} # Override epochs per round

        # Create training pipeline with dependency injection
        self.train_pln = TrainPipeline(
            run_name=f'client-{partition_id}',
            partition_id=self.partition_id,
            partition=partition,
            dp_enabled=self.dp_enabled,
            nested=True,
            config_overrides=_overrides
        )

        logger.info(f'Client {self.partition_id} initialized')
        logger.info(f'  - Epochs per round: {self.train_pln.training_config.epochs}')
        logger.info(f'  - Data partitioning: {self.partition}')
        logger.info(f'  - Differential Privacy: {self.dp_enabled}')

    def fit(self, parameters, config):
        """Train model on local data with secure aggregation."""
        logger.info(f'\n=== Client {self.partition_id} - Round {config.get("server_round", "Unknown")} ===')

        # Set received global model parameters
        set_weights(self.train_pln.model, parameters)

        # Calculate start epoch based on server round and epochs per round
        server_round = config.get('server_round', 1)
        start_epoch = (server_round - 1) * self.train_pln.training_config.epochs

        logger.info(f'Starting training from epoch {start_epoch} for'
                    f' {self.train_pln.training_config.epochs} epochs')

        try:
            # Perform local training
            best_ckpt = self.train_pln.train_model(start_epoch=start_epoch)

            # Get updated model weights
            weights = get_weights(self.train_pln.model)

            # Calculate number of training examples
            num_examples = sum(
                len(batch[0]['encoder_cont']) for batch in self.train_pln.train_dl
            )

            # Extract metrics from best checkpoint
            metrics = best_ckpt['metrics'].copy()

            # Log training results
            logger.info(f'Client {self.partition_id} training completed:')
            for k, v in metrics.items():
                logger.info(f'  {k}: {v:.5f}')
            logger.info(f'  Number of training examples: {num_examples}')

            # Log privacy metrics if DP is enabled
            if self.dp_enabled and hasattr(self.train_pln, 'privacy_engine'):
                epsilon = self.train_pln.privacy_engine.get_epsilon(
                    self.privacy_config.target_delta
                )
                metrics['privacy_epsilon'] = epsilon
                logger.info(f'  Privacy epsilon: {epsilon:.5f}')

            return weights, num_examples, metrics

        except Exception as e:
            logger.error(f'Client {self.partition_id} training failed: {e}')
            raise

    def evaluate(self, parameters, config):
        """Evaluate model on local test data."""
        logger.info(f'\n=== Client {self.partition_id} - Evaluation ===')
        logger.info(f'Evaluation config: {config}')

        # Set received global model parameters
        set_weights(self.train_pln.model, parameters)

        try:
            # Evaluate the model on local test data
            server_round = config.get('server_round', 0)
            metrics = self.train_pln.evaluate_model(step=server_round)

            # Calculate number of evaluation samples
            num_examples = sum(
                len(batch[0]['encoder_cont']) for batch in self.train_pln.test_dl
            )

            logger.info(f'Client {self.partition_id} evaluation completed:')
            for k, v in metrics.items():
                logger.info(f'  {k}: {v:.5f}')
            logger.info(f'  Number of test examples: {num_examples}')

            # Extract loss for Flower (required return format)
            loss = metrics.pop('eval_loss', 0.0)

            return float(loss), num_examples, metrics

        except Exception as e:
            logger.error(f'Client {self.partition_id} evaluation failed: {e}')
            raise

    def get_client_info(self) -> dict:
        """Get client information summary."""
        return {
            'partition_id': self.partition_id,
            'epochs_per_round': self.train_pln.training_config.epochs,
            'partition_enabled': self.partition,
            'dp_enabled': self.dp_enabled,
            'privacy_config': self.privacy_config.model_dump() if self.dp_enabled else None,
            'train_samples': sum(len(batch[0]['encoder_cont']) for batch in self.train_pln.train_dl),
            'test_samples': sum(len(batch[0]['encoder_cont']) for batch in self.train_pln.test_dl),
        }


def client_fn(context: Context):
    """
    Create and return a Flower client.

    Args:
        context: Flower context containing node and run configuration

    Returns:
        Configured Flower client
    """
    # Extract configuration from context
    try:
        partition_id = int(context.node_config.get('partition-id', 0))
    except (KeyError, ValueError):
        partition_id = 0
        logger.warning(f'Could not parse partition-id from node_config, using default: {partition_id}')

    # Get run configuration
    epochs = int(context.run_config.get('epochs', None))
    partition_enabled = context.run_config.get('partition', True)
    dp_enabled = context.run_config.get('dp-enabled', None)

    logger.info(f'Creating client with configuration:')
    logger.info(f'  - Partition ID: {partition_id}')
    logger.info(f'  - Epochs per round: {epochs}')
    logger.info(f'  - Partitioning enabled: {partition_enabled}')
    logger.info(f'  - Differential Privacy: {dp_enabled}')

    # Create and return client
    client = SecAggFlowerClient(
        epochs=epochs,
        partition_id=partition_id,
        partition=partition_enabled,
        dp_enabled=dp_enabled
    )

    # Log client info
    client_info = client.get_client_info()
    logger.info(f'Client info: {client_info}')

    return client.to_client()

# Create Flower ClientApp with SecAgg+ modification
app = ClientApp(client_fn, mods=[secaggplus_mod])

