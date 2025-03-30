from collections import OrderedDict

import torch

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from flwr.client.mod import secaggplus_mod, fixedclipping_mod

from privateer_ad import config, models
from privateer_ad.config import HParams, logger, SecureAggregationConfig, OptimizerConfig
from privateer_ad.data_utils.transform import DataProcessor
from privateer_ad.evaluate.evaluator import ModelEvaluator

from privateer_ad.train.trainer import ModelTrainer
from privateer_ad.utils import set_config


class SecAggFlowerClient(NumPyClient):
    """Flower client implementation with secure aggregation for anomaly detection."""

    def __init__(
            self,
            trainer,
            evaluator,
            train_dl,
            val_dl,
            test_dl,
            device,
            client_id,
            secagg_config
    ):
        self.trainer = trainer
        self.evaluator = evaluator
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.test_dl = test_dl
        self.device = device
        self.client_id = client_id
        self.secagg_config = secagg_config
        logger.info(f"SecAggFlowerClient {client_id} initialized with SecAgg+ config")

    def get_parameters(self, config):
        """Get model parameters as a list of NumPy arrays."""
        return [val.cpu().numpy() for _, val in self.trainer.model.state_dict().items()]

    def set_parameters(self, parameters):
        """Set model parameters from a list of NumPy arrays."""
        params_dict = zip(self.trainer.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.trainer.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        """Train model on local data with secure aggregation."""
        server_round = config.get('server_round', 0)
        logger.info(f"Client {self.client_id}: Starting round {server_round} training with SecAgg+")

        # Set received parameters
        self.set_parameters(parameters)

        # Perform training
        best_checkpoint = self.trainer.training(self.train_dl, self.val_dl)

        # Load best model from checkpoint
        self.trainer.model.load_state_dict(best_checkpoint['model_state_dict'])

        # Calculate metrics
        metrics = {k: v for k, v in best_checkpoint['metrics'].items()}
        logger.info(f"Client {self.client_id}: Training completed with metrics: {metrics}")

        # Get updated model parameters
        updated_parameters = self.get_parameters(config)

        # Calculate number of training samples
        num_examples = sum(len(batch[0]['encoder_cont']) for batch in self.train_dl)

        logger.info(f"Client {self.client_id}: Fit completed with {num_examples} examples")
        return updated_parameters, num_examples, metrics

    def evaluate(self, parameters, config):
        """Evaluate model on local test data."""
        server_round = config.get("server_round", 0)
        logger.info(f"Client {self.client_id}: Starting round {server_round} evaluation")
        self.set_parameters(parameters)

        # Evaluate the model
        metrics, figs = self.evaluator.evaluate(self.trainer.model, self.test_dl)

        # Get loss (or a default value if not available)
        loss = metrics.get("val_loss", 0.0)

        # Calculate number of evaluation samples
        num_examples = sum(len(batch[0]['encoder_cont']) for batch in self.test_dl)

        logger.info(f"Client {self.client_id}: Evaluation completed with metrics: {metrics}")
        return float(loss), num_examples, metrics


def client_fn(context: Context):
    """Create and return a Flower client."""
    # train(**context.run_config)
    # Create all components
    hparams = set_config(HParams, context.run_config)

    secagg_config = set_config(SecureAggregationConfig, context.run_config)
    logger.info(f"CLIENT_FN CONTEXT: {context}")
    client_id = context.node_config.get("partition-id", 0)
    num_partitions = context.node_config.get('num-partitions', 1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Client {client_id}: Using device: {device}")

    # Initialize data processor
    data_processor = DataProcessor()
    # Load and preprocess data
    train_dl = data_processor.get_dataloader('train',
                                             use_pca=hparams.use_pca,
                                             batch_size=hparams.batch_size,
                                             partition_id=client_id,
                                             num_partitions=num_partitions,
                                             seq_len=hparams.seq_len,
                                             only_benign=True)
    val_dl = data_processor.get_dataloader('val',
                                           use_pca=hparams.use_pca,
                                           batch_size=hparams.batch_size,
                                           partition_id=client_id,
                                           num_partitions=num_partitions,
                                           seq_len=hparams.seq_len,
                                           only_benign=True)
    test_dl = data_processor.get_dataloader('test',
                                            use_pca=hparams.use_pca,
                                            batch_size=hparams.batch_size,
                                            partition_id=client_id,
                                            num_partitions=num_partitions,
                                            seq_len=hparams.seq_len,
                                            only_benign=False)

    # Get a sample to determine input dimensions
    sample = next(iter(train_dl))[0]['encoder_cont'][:1].to('cpu')

    # Initialize model
    model_config = set_config(getattr(config, f"{hparams.model}Config", None), context.run_config)
    model_config.seq_len = hparams.seq_len
    model_config.input_size = sample.shape[-1]
    model = getattr(models, hparams.model)(model_config)
    model = model.to(device)

    # Initialize optimizer
    optimizer_config = set_config(OptimizerConfig, context.run_config)
    optimizer = getattr(torch.optim, optimizer_config.name)(
        model.parameters(),
        lr=hparams.learning_rate,
        **optimizer_config.params
    )

    # Initialize trainer and evaluator
    trainer = ModelTrainer(
        model=model,
        optimizer=optimizer,
        criterion=hparams.loss,
        device=device,
        hparams=hparams
    )

    evaluator = ModelEvaluator(
        criterion=hparams.loss,
        device=device
    )

    # Initialize SecAgg client
    client = SecAggFlowerClient(
        trainer=trainer,
        evaluator=evaluator,
        train_dl=train_dl,
        val_dl=val_dl,
        test_dl=test_dl,
        device=device,
        client_id=client_id,
        secagg_config=secagg_config
    )

    logger.info(f"Client {client_id}: Initialized with SecAgg+")
    return client.to_client()

# Create Flower ClientApp
app = ClientApp(client_fn=client_fn,
                mods=[secaggplus_mod,
                      fixedclipping_mod
                      ])
