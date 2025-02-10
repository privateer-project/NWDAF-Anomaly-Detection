import os
from pathlib import Path
from typing import Any, Dict

from tqdm import tqdm
import mlflow
import torch
import torch.nn as nn
from torch.optim import Adam
from torchinfo import summary

from opacus import PrivacyEngine
from opacus.validators import ModuleValidator

from .. import architectures
from src.config import *
from src.data_utils.data_loading.dataloader import DataLoaderFactory
from .evaluator import ModelEvaluator


class ModelTrainer:
    """Model trainer with optional differential privacy."""

    def __init__(
            self,
            paths: Paths,
            metadata: MetaData,
            hparams: HParams,
            model_config: TransformerADConfig | LSTMAutoencoderConfig,
            optimizer_config: OptimizerConfig,
            diff_privacy_config: DifferentialPrivacyConfig | None,
            mlflow_config: MLFlowConfig,
            partition_config: PartitionConfig,
    ):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.hparams = hparams
        self.model_config = model_config
        self.optimizer_config = optimizer_config
        self.diff_privacy = diff_privacy_config
        self.mlflow_config = mlflow_config
        self.paths = paths
        self.partition_config = partition_config
        # Initialize data loaders
        self._init_data_loaders(self.paths, metadata, hparams)

        # Initialize model and training components
        self._init_model()
        self._init_optimizer(self.optimizer_config)
        self._init_criterion()

        # Initialize tracking
        self._init_tracking(self.paths)

        # Setup differential privacy if enabled
        if self.diff_privacy.enable:
            self._setup_privacy_engine()

        # Log parameters
        self._log_parameters()

    def _init_data_loaders(self, paths: Paths, metadata: MetaData, hparams: HParams):
        """Initialize data loaders."""
        loader_factory = DataLoaderFactory(metadata, paths, hparams)
        dataloader_params = {
            'num_workers': os.cpu_count(),
            'pin_memory': True,
            'prefetch_factor': hparams.batch_size * 100,
            'persistent_workers': True
        }

        self.train_dl = loader_factory.get_single_dataloader(train=True,
                                                             split='train',
                                                             window_size=hparams.window_size,
                                                             partition_config=self.partition_config,
                                                             **dataloader_params)

        self.val_dl = loader_factory.get_single_dataloader(train=False,
                                                           split='val',
                                                           window_size=hparams.window_size,
                                                           partition_config=self.partition_config,
                                                           **dataloader_params)

        self.test_dl = loader_factory.get_single_dataloader(train=False,
                                                            split='test',
                                                            window_size=hparams.window_size,
                                                            partition_config=self.partition_config,
                                                            **dataloader_params)

        # Get sample for model initialization
        self.sample = next(iter(self.train_dl))[0]['encoder_cont'][:1].to('cpu')

    def _init_model(self):
        """Initialize the LSTM Autoencoder model."""
        if self.hparams.model == 'TransformerAD':
            self.model_config.d_input = self.sample.shape[-1]
        if self.hparams.model == 'LSTMAutoencoder':
            self.model_config.input_size = self.sample.shape[-1]
        self.model_class = getattr(architectures, self.hparams.model)
        self.model = self.model_class(**self.model_config.__dict__)
        _sum = summary(self.model,
                       input_data=self.sample,
                       col_names=('input_size', 'output_size', 'num_params', 'params_percent', 'trainable'))
        mlflow.log_text(str(_sum), 'model_summary.txt')

    def _init_optimizer(self, optimizer_config: OptimizerConfig):
        """Initialize the optimizer."""

        self.optimizer = getattr(torch.optim, optimizer_config.type)
        self.optimizer = Adam(
            self.model.parameters(),
            lr=self.hparams.learning_rate,
            **optimizer_config.params
        )

    def _init_criterion(self):
        """Initialize the loss criterion."""
        self.criterion = getattr(nn, self.hparams.loss)(reduction='mean')

    def _init_tracking(self, paths: Paths):
        """Initialize training tracking and checkpoints."""
        paths.models.mkdir(parents=True, exist_ok=True)

        self.metrics = {'loss': float('inf'), 'val_loss': float('inf')}
        self.best_checkpoint: Dict[str, Any] = {
            'epoch': 0,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }
        self.best_checkpoint.update(self.metrics)

        # Early stopping parameters
        self.es_not_improved_epochs = 0
        self.es_patience_epochs = 10
        self.es_warmup_epochs = 10
        self.es_improvement_threshold = 0.005

    def _setup_privacy_engine(self):
        """Setup differential privacy if enabled."""
        if not self.diff_privacy.enable:
            return
        self.privacy_engine = PrivacyEngine(accountant="rdp", secure_mode=self.diff_privacy.secure_mode)
        self.model = ModuleValidator.fix(self.model)

        self._init_optimizer(self.optimizer_config)
        self.model, self.optimizer, self.train_dl = (
            self.privacy_engine.make_private_with_epsilon(
                module=self.model,
                optimizer=self.optimizer,
                data_loader=self.train_dl,
                epochs=self.hparams.epochs,
                target_epsilon=self.diff_privacy.target_epsilon,
                target_delta=self.diff_privacy.target_delta,
                max_grad_norm=self.diff_privacy.max_grad_norm,
                # secure_mode=self.diff_privacy.secure_mode,
                )
        )

    def _log_parameters(self):
        """Log parameters to MLFlow."""
        if not self.mlflow_config.track:
            return
        mlflow.log_params(self.hparams.__dict__)
        mlflow.log_params(self.model_config.__dict__)
        mlflow.log_params(self.optimizer_config.__dict__)
        mlflow.log_params({'model_type': self.model.__class__.__name__,
                           'device': self.device})
        if self.diff_privacy.enable:
            mlflow.log_params(self.diff_privacy.__dict__)

    def training(self):
        """Main training loop."""
        print(f'Using {self.device}')
        self.model = self.model.to(self.device)

        for epoch in range(1, self.hparams.epochs + 1):
            # Training and validation
            train_metrics = self._training_loop(epoch)
            val_metrics = self._evaluate_loop()

            # Update and log metrics
            self._update_metrics(epoch, train_metrics, val_metrics)

            # Early stopping check
            if self._check_early_stopping(epoch):
                break

        # Load best model state
        self.model.load_state_dict(self.best_checkpoint['model_state_dict'])

        # Log model if tracking enabled
        if self.mlflow_config.track:
            mlflow.pytorch.log_model(pytorch_model=self.model.to('cpu'),
                                     artifact_path='best_model.pt',
                                     input_example=self.sample.to('cpu').numpy())
            self.model.to(self.device)
        return self.best_checkpoint

    def _training_loop(self, epoch: int) -> Dict[str, float]:
        """Single epoch training loop."""
        self.model.train()
        loss = 0.0

        for inputs in tqdm(self.train_dl, desc=f"Epoch {epoch}/{self.hparams.epochs}"):
            x = inputs[0]['encoder_cont'].to(self.device)

            self.optimizer.zero_grad()
            output = self.model(x)
            rec_error = self.criterion(x, output)
            rec_error.backward()
            self.optimizer.step()
            loss += rec_error.item()
        loss /= len(self.train_dl)
        return {'loss': loss}

    def _evaluate_loop(self) -> Dict[str, float]:
        """Evaluation loop."""
        self.model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for inputs in tqdm(self.val_dl, desc="Validation"):
                x = inputs[0]['encoder_cont'].to(self.device)
                output = self.model(x)
                val_loss += self.criterion(x, output).item()
        val_loss /= len(self.val_dl)
        return {'val_loss': val_loss}

    def evaluate_model(self, save_path: Path = None):
        """Evaluate the trained model."""
        evaluator = ModelEvaluator(
            model=self.model,
            hparams=self.hparams,
            device=self.device,
            paths=self.paths,
            mlflow_config=self.mlflow_config
        )
        return evaluator.evaluate(self.test_dl, save_path=save_path)

    def _update_metrics(self, epoch: int, train_metrics: Dict[str, float],
                        val_metrics: Dict[str, float]):
        """Update and log metrics."""
        self.metrics.update(train_metrics)
        self.metrics.update(val_metrics)

        if self.diff_privacy.enable:
            self.metrics.update({
                'epsilon': self.privacy_engine.get_epsilon(self.diff_privacy.target_delta)
            })

        if self.mlflow_config.track:
            mlflow.log_metrics(self.metrics, step=epoch)

        print(f"epoch:{epoch} loss:{self.metrics['loss']:.4f} val_loss:{self.metrics['val_loss']:.4f}")

        if self.diff_privacy.enable:
            print(f"Privacy Budget (ε, δ): ({self.metrics['epsilon']:.4f}, {self.diff_privacy.target_delta})")

    def _check_early_stopping(self, epoch: int) -> bool:
        """Check early stopping conditions."""
        if self.metrics['val_loss'] < self.best_checkpoint['val_loss']:
            self.best_checkpoint.update({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict()
            })
            self.best_checkpoint.update(self.metrics)
            self.es_not_improved_epochs = 0
            return False

        if epoch > self.es_warmup_epochs:
            self.es_not_improved_epochs += 1

        if self.es_not_improved_epochs > self.es_patience_epochs:
            print(f'Early stopping. No improvement for {self.es_patience_epochs} epochs. '
                  f'Current epoch {epoch}')
            return True
        if epoch > self.es_warmup_epochs:
            print(f'\nval_loss have not increased for {self.es_not_improved_epochs} epochs.')
            print(f"val_loss={self.metrics['val_loss']:.4f}\n"
                  f"best val_loss={self.best_checkpoint['val_loss']:.4f}")
        return False
