from typing import Dict, Any
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
from torch import nn
from tqdm import tqdm
import mlflow
import torch

from src.config import *
from src.config import EarlyStoppingConfig
from src.utils import set_config

class ModelTrainer:
    """Model trainer with optional differential privacy."""

    def __init__(self,
                 train_dl,
                 val_dl,
                 model,
                 optimizer,
                 criterion,
                 device,
                 hparams: HParams,
                 **kwargs):

        self.paths = PathsConf()
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = getattr(nn, criterion)(reduction='mean')
        self.device = device
        self.hparams = hparams
        self.early_stopping = self.hparams.early_stopping
        if self.hparams.apply_dp:
            self.dp_config = set_config(DifferentialPrivacyConfig, kwargs)

        if self.early_stopping:
            logger.info("Early stopping ")
            self.es_not_improved_epochs = 0
            self.es_conf = EarlyStoppingConfig()
            self.es_patience_epochs = self.es_conf.es_patience_epochs
            self.es_warmup_epochs = self.es_conf.es_warmup_epochs
            self.es_improvement_threshold = self.es_conf.es_improvement_threshold

            if mlflow.active_run():  # log early stopping params
                mlflow.log_params({'es_not_improved_epochs': self.es_not_improved_epochs,
                                   'es_patience_epochs': self.es_patience_epochs,
                                   'es_warmup_epochs': self.es_warmup_epochs,
                                   'es_improvement_threshold': self.es_improvement_threshold
                                   })

        # Initialize metrics dict and best_checkpoint dict
        metrics = {'loss': float('inf'), 'val_loss': float('inf')}
        self.metrics = metrics.copy()
        self.best_checkpoint: Dict[str, Any] = {'epoch': 0,
                                                'model_state_dict': self.model.state_dict(),
                                                'optimizer_state_dict': self.optimizer.state_dict(),
                                                'metrics': {'best_' + k: v for k, v in metrics.copy().items()}
                                                }

        # Initialize differential privacy
        if self.hparams.apply_dp:
            logger.info('Differential Privacy enabled.')
            self.privacy_engine = PrivacyEngine(accountant="rdp", secure_mode=self.dp_config.secure_mode)
            self.model = ModuleValidator.fix(self.model)
            self.model, self.optimizer, self.train_dl = self.privacy_engine.make_private_with_epsilon(
                module=self.model,
                optimizer=self.optimizer,
                data_loader=self.train_dl,
                epochs=hparams.epochs,
                target_epsilon=self.dp_config.target_epsilon,
                target_delta=self.dp_config.target_delta,
                max_grad_norm=self.dp_config.max_grad_norm,
                secure_mode=self.dp_config.secure_mode,
            )
        if self.hparams.apply_dp and mlflow.active_run():
            mlflow.log_params(self.dp_config.__dict__)

    def training(self):
        """Main training loop."""
        try:
            for epoch in range(1, self.hparams.epochs + 1):
                # Training and validation
                train_metrics = self._training_loop(epoch)
                val_metrics = self._validation_loop()
                self.log_metrics(train_metrics | val_metrics, epoch)
                # Early stopping check

                if self.early_stopping and epoch > self.es_warmup_epochs:
                    if self.metrics[self.hparams.target] <= self.best_checkpoint['metrics']['best_' + self.hparams.target]:
                        best_metrics = {'best_' + k: v for k, v in self.metrics.copy().items()}
                        self.best_checkpoint.update({
                            'metrics': best_metrics,
                            'epoch': epoch,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict()
                        })
                    if self._check_early_stopping():
                        break

        except KeyboardInterrupt:
            print("\nTraining interrupted by user. Proceeding to evaluation...")
            # Save the current state as the best checkpoint if we don't have one
            if self.best_checkpoint['epoch'] == 0:
                self.best_checkpoint.update({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict()
                })
        return self.best_checkpoint

    def _training_loop(self, epoch: int) -> Dict[str, float]:
        """Single epoch handlers loop."""
        self.model.train()
        loss = 0.0

        for inputs in tqdm(self.train_dl, desc=f"Epoch {epoch}/{self.hparams.epochs} - Train:"):
            x = inputs[0]['encoder_cont'].to(self.device)
            self.optimizer.zero_grad()
            # Compute autoencoder loss
            ae_output = self.model(x)
            batch_loss = self.loss_fn(x, ae_output)
            batch_loss.backward()
            self.optimizer.step()

            loss += batch_loss.item()
        loss /= len(self.train_dl)
        return {'loss': loss}

    def _validation_loop(self) -> Dict[str, float]:
        """Evaluation loop."""
        self.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs in tqdm(self.val_dl, desc=f"Validation:"):
                x = inputs[0]['encoder_cont'].to(self.device)
                ae_output = self.model(x)
                val_loss += self.loss_fn(x, ae_output).item()
            val_loss /= len(self.val_dl)
        return {'val_loss': val_loss}

    def log_metrics(self, metrics, epoch: int):
        """Update and log metrics."""
        self.metrics.update(metrics)
        if self.hparams.apply_dp:
            self.metrics.update({
                'epsilon': self.privacy_engine.get_epsilon(self.dp_config.target_delta)
            })

        if mlflow.active_run():
            mlflow.log_metrics(self.metrics, step=epoch)

        _prnt = [f'{key}: {str(round(value, 5))}' for key, value in self.metrics.items()]
        print(f"Metrics: {' '.join(_prnt)}")

    def _check_early_stopping(self) -> bool:
        """Check early stopping conditions."""
        if self.metrics[self.hparams.target] <= self.best_checkpoint['metrics']['best_' + self.hparams.target]:
            self.es_not_improved_epochs = 0
            return False
        else:
            self.es_not_improved_epochs += 1
            print(f'{self.hparams.target} have not increased for {self.es_not_improved_epochs} epochs.')
            print(f"{self.hparams.target}: "
                  f"best= {self.best_checkpoint['metrics']['best_' + self.hparams.target]:.5f} - "
                  f"current= {self.metrics[self.hparams.target]:.5f}\n"
                  f"")
        if self.es_not_improved_epochs >= self.es_patience_epochs:
            print(f'Early stopping. No improvement for {self.es_not_improved_epochs} epochs.')
            return True
        return False
