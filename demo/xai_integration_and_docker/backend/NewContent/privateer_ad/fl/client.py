import logging

from flwr.common import Context
from flwr.common.config import ConfigRecord
from flwr.client import ClientApp, NumPyClient
from flwr.client.mod import secaggplus_mod

from privateer_ad.config import DataConfig, TrainingConfig, MLFlowConfig
from privateer_ad.fl.utils import set_weights, get_weights
from privateer_ad.train import TrainPipeline


class SecAggFlowerClient(NumPyClient):
    """
    Secure federated learning client for privacy-preserving anomaly detection.

    This class implements a Flower federated learning client that incorporates
    secure aggregation protocol to protect model updates during the federation
    process. The client is specifically designed for anomaly detection scenarios
    where maintaining privacy of local data patterns is paramount.

    The implementation integrates SecAgg+ protocol to ensure that individual
    client model updates remain encrypted during transmission and aggregation.
    This approach allows multiple organizations or network segments to collaborate
    in training robust anomaly detection models without exposing their specific
    data distributions or learned patterns to other participants.

    The client manages its own data partition and training pipeline while
    coordinating with the federated server for model synchronization. It maintains
    proper experiment tracking through MLflow integration, allowing for detailed
    analysis of federated training dynamics and client-specific performance
    characteristics.

    Each client operates on a designated data partition, simulating realistic
    federated scenarios where different network segments or organizations contribute
    their local data to the collaborative learning process. The partitioning ensures
    that clients see distinct data distributions, which is crucial for evaluating
    federated learning effectiveness in heterogeneous environments.

    Attributes:
        client_state: Flower context state management for persistent client information
        mlflow_config (MLFlowConfig): Configuration for experiment tracking and run coordination
        training_config (TrainingConfig): Training parameters including epochs and optimization settings
        data_config (DataConfig): Data handling configuration including partition assignments
    """

    def __init__(self,context: Context):
        """
        Initialize the federated learning client with secure aggregation capabilities.

        Sets up the client environment by extracting configuration from the Flower
        context and establishing connections to the experiment tracking infrastructure.
        The initialization process configures data partitioning, training parameters,
        and MLflow integration for comprehensive experiment management.

        Args:
            context (Context): Flower context object containing run configuration,
                             node assignments, and state management capabilities.
                             This context provides the client with its identity
                             and operational parameters within the federation.
        """
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
        """
        Perform local model training with received global model parameters.

        This method constitutes the core of the federated learning process from
        the client perspective. It receives the current global model parameters
        from the server, applies them to the local model, conducts training on
        the client's data partition, and returns the updated model weights along
        with training metrics.

        The training process maintains full integration with the experiment tracking
        infrastructure, ensuring that each client's contribution to the federated
        learning process is properly documented. This detailed tracking proves
        invaluable for understanding client-specific learning dynamics and
        identifying potential issues in heterogeneous federated environments.

        The method handles MLflow run coordination carefully, establishing parent-child
        relationships between the server's experiment run and the client's local
        training session. This hierarchical structure enables comprehensive analysis
        of federated learning experiments while maintaining clear organization of
        results across multiple participants.

        Args:
            parameters: Global model parameters received from the federated server,
                       typically numpy arrays representing the current state of
                       the collaborative model.
            config: Round-specific configuration dictionary containing server round
                   number and other coordination information needed for this
                   training iteration.

        Returns:
            tuple: A three-element tuple containing:
                - Updated model weights as numpy arrays
                - Number of training examples used in this round
                - Dictionary of training metrics for server aggregation

        Note:
            The training continues from the epoch corresponding to the current
            federated round, maintaining proper training progression across
            multiple federation cycles.
        """

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
        """
        Evaluate the global model on local test data for federated assessment.

        This method performs local evaluation of the current global model state
        using the client's test data partition. The evaluation provides insights
        into how well the federated model performs across different data
        distributions, which is crucial for understanding the effectiveness of
        the collaborative learning process.

        The evaluation process maintains the same experimental rigor as local
        training, with full integration into the tracking infrastructure. This
        consistency enables comprehensive analysis of model performance evolution
        throughout the federated learning process, both from individual client
        perspectives and in aggregate across the entire federation.

        Unlike the training phase, evaluation does not modify the model parameters.
        Instead, it applies the received global model to local test data and
        computes performance metrics that help assess generalization across
        different data distributions represented by various federation participants.

        Args:
            parameters: Current global model parameters received from the server
                       for evaluation on local test data.
            config: Evaluation round configuration containing coordination
                   information and round-specific settings.

        Returns:
            tuple: A three-element tuple containing:
                - Primary evaluation loss as a float value
                - Number of test examples evaluated
                - Dictionary of comprehensive evaluation metrics
        """

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
    """
    Factory function for creating federated learning client instances.

    Args:
        context (Context): Flower execution context containing run configuration,
                         node assignments, and other parameters needed for client
                         initialization and operation.

    Returns:
        Client: Configured Flower client instance ready for federated learning
               participation with secure aggregation capabilities enabled.
    """
    logging.info(f'Client run config:\n{context.run_config}.')
    return SecAggFlowerClient(context).to_client()


app = ClientApp(client_fn, mods=[secaggplus_mod])