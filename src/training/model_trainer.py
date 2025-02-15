import logging
from typing import Dict, Any

import mlflow
import torch
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
from torch import nn
from torchinfo import summary
from tqdm import tqdm

from src import architectures

from src.config import *


class ModelTrainer:
    """Model trainer with optional differential privacy."""

    def __init__(self, train_dl, val_dl, device,
                 hparams: HParams,
                 model_config: TransformerADConfig | LSTMAutoencoderConfig,
                 optimizer_config: OptimizerConfig,
                 diff_privacy_config: DifferentialPrivacyConfig | None):
        self.hparams = hparams
        self.diff_privacy = diff_privacy_config
        self.model_config = model_config
        self.optimizer_config = optimizer_config

        self.train_dl = train_dl
        self.val_dl = val_dl
        self.device = device
        self.model = None

        # Initialize model and training components
        self.sample = next(iter(self.train_dl))[0]['encoder_cont'][:1].to('cpu')
        model_sum = self._init_model()


        self._init_optimizer()
        self._init_criterion()
        self._init_early_stopping()
        self._init_tracking()
        self._setup_privacy_engine()
        if mlflow.active_run():
            mlflow.log_params({'es_not_improved_epochs': self.es_not_improved_epochs,
                               'es_patience_epochs': self.es_patience_epochs,
                               'es_warmup_epochs': self.es_warmup_epochs,
                               'es_improvement_threshold': self.es_improvement_threshold})
            mlflow.log_text(str(model_sum), 'model_summary.txt')
            mlflow.log_params(model_config.__dict__)
            mlflow.log_params({'device': self.device})
            mlflow.log_params(self.hparams.__dict__)
            mlflow.log_params(self.diff_privacy.__dict__)
            mlflow.log_params(self.optimizer_config.__dict__)

    def _init_model(self):
        """Initialize model."""
        if self.hparams.model == 'TransformerAD':
            self.model_config.d_input = self.sample.shape[-1]
        if self.hparams.model == 'LSTMAutoencoder':
            self.model_config.input_size = self.sample.shape[-1]

        model_class = getattr(architectures, self.hparams.model)
        self.model = model_class(**self.model_config.__dict__)
        return summary(self.model,
                       input_data=self.sample,
                       col_names=('input_size', 'output_size', 'num_params', 'params_percent', 'trainable'))

    def _init_optimizer(self):
        """Initialize the optimizer."""
        optimizer_class = getattr(torch.optim, self.optimizer_config.type)
        self.optimizer = optimizer_class(self.model.parameters(),
                                         lr=self.hparams.learning_rate,
                                         **self.optimizer_config.params)

    def _init_criterion(self):
        """Initialize the loss criterion."""
        self.criterion = getattr(nn, self.hparams.loss)(reduction='mean')

    def _init_early_stopping(self):
        self.es_not_improved_epochs = 0
        self.es_patience_epochs = 10
        self.es_warmup_epochs = 10
        self.es_improvement_threshold = 0.005

    def _init_tracking(self):
        """Initialize training tracking and checkpoints."""
        metrics = {'loss': float('inf'), 'val_loss': float('inf')}
        self.metrics = metrics.copy()
        self.best_checkpoint: Dict[str, Any] = {
            'epoch': 0,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics.copy()
        }

    def _setup_privacy_engine(self):
        """Setup differential privacy if enabled."""
        if not self.diff_privacy.enable:
            logging.warning('Differential Privacy disabled.')
            return
        self.privacy_engine = PrivacyEngine(accountant="rdp", secure_mode=self.diff_privacy.secure_mode)
        self.model = ModuleValidator.fix(self.model)
        self.model, self.optimizer, self.train_dl = (
            self.privacy_engine.make_private_with_epsilon(
                module=self.model,
                optimizer=self.optimizer,
                data_loader=self.train_dl,
                epochs=self.hparams.epochs,
                target_epsilon=self.diff_privacy.target_epsilon,
                target_delta=self.diff_privacy.target_delta,
                max_grad_norm=self.diff_privacy.max_grad_norm,
                secure_mode=self.diff_privacy.secure_mode,
            )
        )

    def training(self):
        """Main training loop."""
        print(f'Using {self.device}')
        self.model = self.model.to(self.device)
        for epoch in range(1, self.hparams.epochs + 1):
            # Training and validation
            train_metrics = self._training_loop(epoch)
            val_metrics =self._validation_loop()
            epoch_metrics = train_metrics | val_metrics
            self.log_metrics(epoch_metrics, epoch)

            # Early stopping check
            if self._check_early_stopping(epoch, target='val_loss'):
                break
        # Load best model state
        self.model.load_state_dict(self.best_checkpoint['model_state_dict'])
        return self.best_checkpoint['metrics']

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
        return {'loss': loss}  #extend to return n metrics too

    def _validation_loop(self) -> Dict[str, float]:
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

    def log_metrics(self, metrics, epoch: int):
        """Update and log metrics."""
        self.metrics.update(metrics)
        if self.diff_privacy.enable:
            self.metrics.update({
                'epsilon': self.privacy_engine.get_epsilon(self.diff_privacy.target_delta)
            })

        if mlflow.active_run():
            mlflow.log_metrics(metrics, step=epoch)
        _prnt = [f'{key}: {str(round(value, 5))}' for key, value in metrics.items()]
        print(f"\nMetrics: {' '.join(_prnt)}")
        if self.diff_privacy.enable:
            print(f"Privacy Budget (ε, δ): ({self.metrics['epsilon']:.4f}, {self.diff_privacy.target_delta})")

    def _check_early_stopping(self, epoch: int, target='val_loss') -> bool:
        """Check early stopping conditions."""
        if epoch < self.es_warmup_epochs:
            return False

        if self.metrics[target] < self.best_checkpoint['metrics'][target]:
            self.es_not_improved_epochs = 0
            self.best_checkpoint.update({
                'metrics': self.metrics.copy(),
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict()
            })
            return False
        else:
            self.es_not_improved_epochs += 1
            print(f'{target} have not increased for {self.es_not_improved_epochs} epochs.')
            print(f"{target}: "
                  f"best= {self.best_checkpoint['metrics'][target]:.5f} - "
                  f"current= {self.metrics[target]:.5f}\n"
                  f"")
        if self.es_not_improved_epochs > self.es_patience_epochs:
            print(f'Early stopping. No improvement for {self.es_not_improved_epochs} epochs.')
            return True

        return False
