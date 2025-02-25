"""privateer-ad: A Flower / PyTorch app."""
import torch
import torch.nn as nn

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from flwr.client.mod import secaggplus_mod, fixedclipping_mod

from src import config

from src.train.trainer import ModelTrainer

class FlowerClient(NumPyClient):
    """Flower client implementing federated learning client functionality."""

    def __init__(
        self, trainer: ModelTrainer):
        self.trainer = trainer

    @staticmethod
    def get_model_weights(model: nn.Module):
        """Get model weights as NDArrays."""
        return [val.cpu().numpy() for _, val in model.state_dict().items()]

    @staticmethod
    def set_model_weights(model: nn.Module, weights):
        """Set model weights from NDArrays."""
        params_dict = zip(model.state_dict().keys(), weights)
        state_dict = {k: torch.from_numpy(v) for k, v in params_dict}
        model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        """Train model on local data."""
        # Update model with received parameters
        self.set_model_weights(self.trainer.model, parameters)

        # Train model
        best_checkpoint = self.trainer.training()
        best_checkpoint['train_loss'] = float("inf")
        return (
            self.get_model_weights(self.trainer.model),
            len(self.trainer.train_dl.dataset),
            {"loss": best_checkpoint['train_loss'],
             'val_loss': best_checkpoint['val_loss']}
        )

    def evaluate(self, parameters, config):
        """Evaluate model on local data."""
        self.set_model_weights(self.trainer.model, parameters)

        # Evaluate model
        eval_metrics = self.trainer.evaluate_model()

        return (
            eval_metrics['ae_roc_auc_score'],
            len(self.trainer.test_dl.dataset),
            eval_metrics
        )

def create_client_components(context: Context):
    """Create all necessary components for the client."""
    # Initialize configs from context
    paths = config.PathsConf()
    metadata = config.MetaData()
    hparams = config.HParams()
    partition_config = PartitionConfig(partition_id=context.node_config['partition-id'],
                                       num_partitions=context.node_config['num-partitions'])

    model_config = getattr(config, hparams.model + 'Config')()
    if hparams.model == 'TransformerAD':
        model_config.seq_len = hparams.seq_len

    # Initialize trainer
    trainer = ModelTrainer(
        paths=paths,
        metadata=metadata,
        hparams=hparams,
        model_config=model_config,
        optimizer_config=OptimizerConfig(),  # Use default optimizer config
        diff_privacy_config=DifferentialPrivacyConfig(enable=False),  # No DP in federated setting
        mlflow_config=MLFlowConfig(track=False),  # No MLFlow in client
        partition_config=partition_config)

    return  trainer

def client_fn(context: Context):
    """Create and return a Flower client."""
    # Create all components
    trainer = create_client_components(context)
    # Create and return client
    return FlowerClient(trainer=trainer).to_client()

# Create Flower ClientApp
app = ClientApp(client_fn=client_fn,
                mods=[secaggplus_mod,
                      fixedclipping_mod
                      ])
