import os
from collections import OrderedDict

import torch
import mlflow
import matplotlib.pyplot as plt

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from flwr.client.mod import secaggplus_mod, fixedclipping_mod

from privateer_ad import config, models
from privateer_ad.config import HParams, logger, SecureAggregationConfig, OptimizerConfig, MLFlowConfig
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
            secagg_config,
            mlflow_config,
            parent_run_id=None
    ):
        self.trainer = trainer
        self.evaluator = evaluator
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.test_dl = test_dl
        self.device = device
        self.client_id = client_id
        self.secagg_config = secagg_config
        self.mlflow_config = mlflow_config
        self.mlflow_run_id = None
        self.parent_run_id = parent_run_id

        logger.info(f"SecAggFlowerClient {client_id} initialized with SecAgg+ config")

        # Setup MLflow tracking - create a single run for this client
        if self.mlflow_config.track:
            mlflow.set_tracking_uri(self.mlflow_config.server_address)
            mlflow.set_experiment(self.mlflow_config.experiment_name)

            # Get the experiment ID
            experiment = mlflow.get_experiment_by_name(self.mlflow_config.experiment_name)

            # Check if we already have a run for this client
            client_runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string=f"tags.mlflow.runName = 'client_{self.client_id}'",
                max_results=1
            )

            if len(client_runs) > 0:
                # Use existing run
                self.mlflow_run_id = client_runs.iloc[0].run_id
                logger.info(f"Found existing run for client {self.client_id}: {self.mlflow_run_id}")
            else:
                # Create a new run for this client as a child of the parent run
                if self.parent_run_id:
                    # Create nested run under the parent
                    with mlflow.start_run(run_id=self.parent_run_id):
                        with mlflow.start_run(run_name=f"client_{self.client_id}", nested=True) as child_run:
                            self.mlflow_run_id = child_run.info.run_id
                            # Log client ID as a parameter
                            mlflow.log_param("client_id", self.client_id)
                            logger.info(f"Created new nested run for client {self.client_id}: {self.mlflow_run_id}")
                else:
                    # Create standalone run if no parent
                    with mlflow.start_run(run_name=f"client_{self.client_id}") as run:
                        self.mlflow_run_id = run.info.run_id
                        # Log client ID as a parameter
                        mlflow.log_param("client_id", self.client_id)
                        logger.info(f"Created new standalone run for client {self.client_id}: {self.mlflow_run_id}")

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

        # Log metrics to MLflow with round as step
        if self.mlflow_config.track and self.mlflow_run_id:
            # Log to existing run without creating a new nested run
            for key, value in metrics.items():
                mlflow.log_metric(key=key, value=value, step=server_round, run_id=self.mlflow_run_id)

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

        # Log metrics to MLflow with the existing client run
        if self.mlflow_config.track and self.mlflow_run_id:
            # Add prefix to distinguish evaluation metrics
            eval_metrics = {f"eval_{k}": v for k, v in metrics.items()}

            # Log evaluation metrics with round as step
            for key, value in eval_metrics.items():
                mlflow.log_metric(
                    key=key,
                    value=value,
                    step=server_round,
                    run_id=self.mlflow_run_id
                )

            # Log figures if available - with proper handling to avoid file errors
            if figs:
                try:
                    mlflow_client = mlflow.tracking.MlflowClient()
                    for name, fig in figs.items():
                        # Use a unique path with client ID and round number to avoid conflicts
                        fig_name = f"client_{self.client_id}_{name}_round_{server_round}.png"

                        # Save the figure to a temporary file
                        fig.savefig(fig_name)

                        # Log the figure as an artifact
                        mlflow_client.log_artifact(self.mlflow_run_id, fig_name)

                        # Close the figure to release memory and avoid figure limit warning
                        plt.close(fig)

                        # Clean up the file after logging
                        if os.path.exists(fig_name):
                            os.remove(fig_name)

                    logger.info(
                        f"Client {self.client_id}: Successfully logged {len(figs)} figures for round {server_round}")
                except Exception as e:
                    # If anything goes wrong with figure logging, log the error but continue execution
                    logger.error(f"Client {self.client_id}: Error logging figures: {e}")
                    # Still try to log the count as a metric
                    mlflow.log_metric(
                        key="eval_figures_count",
                        value=len(figs),
                        step=server_round,
                        run_id=self.mlflow_run_id
                    )

        logger.info(f"Client {self.client_id}: Evaluation completed with metrics: {metrics}")
        return float(loss), num_examples, metrics


def client_fn(context: Context):
    """Create and return a Flower client."""
    # Get configurations from context
    hparams = set_config(HParams, context.run_config)
    secagg_config = set_config(SecureAggregationConfig, context.run_config)
    mlflow_config = set_config(MLFlowConfig, context.run_config)

    logger.info(f"CLIENT_FN CONTEXT: {context}")
    client_id = context.node_config.get("partition-id", 0)
    num_partitions = context.node_config.get('num-partitions', 1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Client {client_id}: Using device: {device}")

    # Get the parent run ID from the config - this should be set by the server
    parent_run_id = context.run_config.get('mlflow_parent_run_id', None)
    if parent_run_id:
        logger.info(f"Client {client_id}: Using parent run ID: {parent_run_id}")
    else:
        logger.warning(f"Client {client_id}: No parent run ID provided in config")

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
        secagg_config=secagg_config,
        mlflow_config=mlflow_config,
        parent_run_id=parent_run_id
    )

    logger.info(f"Client {client_id}: Initialized with SecAgg+ and MLflow tracking")
    return client.to_client()

# Create Flower ClientApp
app = ClientApp(client_fn=client_fn,
                mods=[secaggplus_mod,
                      fixedclipping_mod
                      ])