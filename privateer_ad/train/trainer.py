from typing import Dict, Any
from torch import nn
from tqdm import tqdm
import mlflow
import torch

from privateer_ad.config import HParams, EarlyStoppingConfig, logger

class ModelTrainer:
    """Trains a PyTorch model with early stopping capability.

     This class handles the training workflow for autoencoder models, including
     training loop, validation, metrics tracking, and early stopping functionality.
     It also provides integration with MLflow for experiment tracking.

     Attributes:
         model: The PyTorch model to be trained.
         optimizer: The optimizer used for training.
         loss_fn: The loss function used for training.
         device: The device (CPU/GPU) where training will be performed.
         hparams (HParams): Hyperparameters for training.
         early_stopping (bool): Whether to use early stopping.
         metrics (Dict[str, float]): Dictionary to track training metrics.
         best_checkpoint (Dict[str, Any]): Dictionary to store the best model checkpoint.
     """
    def __init__(self,
                 model,
                 optimizer,
                 criterion,
                 device,
                 hparams: HParams):
        """Initializes the ModelTrainer with model, optimizer, and training configuration.

                Args:
                    model: The PyTorch model to be trained.
                    optimizer: The optimizer used for training.
                    criterion: The name of the loss function as defined in torch.nn.
                    device: The device (CPU/GPU) where training will be performed.
                    hparams (HParams): Hyperparameters for training including epochs,
                                      early stopping flag, and target metric.
        """

        self.model = model
        self.optimizer = optimizer
        self.loss_fn = getattr(nn, criterion)(reduction='mean')
        self.device = device
        self.hparams = hparams
        self.early_stopping = self.hparams.early_stopping

        if self.early_stopping:
            logger.info("Early stopping ")
            self.es_not_improved_epochs = 0
            es_conf = EarlyStoppingConfig()
            self.es_patience_epochs = es_conf.es_patience_epochs
            self.es_warmup_epochs = es_conf.es_warmup_epochs
            self.es_improvement_threshold = es_conf.es_improvement_threshold
            if self.hparams.direction not in ('maximize', 'minimize'):
                raise ValueError(
                "direction must be 'maximize' or 'minimize'. Current value: {}".format(self.hparams.direction))

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

    def training(self, train_dl, val_dl):
        """Executes the full training process with validation and early stopping.

        Trains the model for the specified number of epochs or until early stopping
        criteria are met. For each epoch, runs a training loop followed by a validation
        loop, logs metrics, and tracks the best model state. Supports both maximization
        and minimization objectives based on the direction specified in hyperparameters.

        The training can be manually interrupted with Ctrl+C, in which case the best
        model state until interruption is preserved for evaluation.

        Args:
            train_dl: DataLoader for training data.
            val_dl: DataLoader for validation data.

        Returns:
            Dict[str, Any]: The best checkpoint dictionary containing:
                - model_state_dict: State dict of the best model
                - optimizer_state_dict: State dict of the optimizer at best point
                - metrics: Dictionary of metrics at best point
                - epoch: Epoch number when best model was achieved
        """
        try:
            for epoch in range(1, self.hparams.epochs + 1): # Training and validation loop
                train_metrics = self._training_loop(epoch, train_dl)
                val_metrics = self._validation_loop(val_dl)
                self.log_metrics(train_metrics | val_metrics, epoch)
                current_val = self.metrics[self.hparams.target]
                best_val = self.best_checkpoint['metrics'][self.hparams.target]
                if self.hparams.direction == 'maximize':
                    is_best = current_val >= best_val
                else:
                    is_best = current_val <= best_val
                if is_best:
                    self.best_checkpoint.update({
                        'metrics': self.metrics.copy(),
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict()
                    })
                if self._check_early_stopping(epoch=epoch, is_best=is_best): # Early stopping check
                    break

        except KeyboardInterrupt: # Break training loop when Ctrl+C pressed (Manual early stopping)
            print("\nTraining interrupted by user. Proceeding to evaluation...")
            # Save the current state as the best checkpoint if we don't have one
            if self.best_checkpoint['epoch'] == 0:
                self.best_checkpoint.update({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict()
                })
        return self.best_checkpoint

    def _training_loop(self, epoch: int, train_dl) -> Dict[str, float]:
        """Executes one training epoch.

        Puts the model in training mode and processes all batches in the training
        dataloader. For each batch, performs forward pass, loss calculation,
        backward pass, and optimizer step.

        Args:
            epoch (int): Current epoch number.
            train_dl: DataLoader for training data.

        Returns:
            Dict[str, float]: Dictionary containing training metrics (loss).
        """
        self.model.train()
        loss = 0.0

        for inputs in tqdm(train_dl, desc=f"Epoch {epoch}/{self.hparams.epochs} - Train:"):
            x = inputs[0]['encoder_cont'].to(self.device)
            self.optimizer.zero_grad()
            # Compute autoencoder loss
            ae_output = self.model(x)
            batch_loss = self.loss_fn(x, ae_output)
            batch_loss.backward()
            self.optimizer.step()

            loss += batch_loss.item()
        loss /= len(train_dl)
        return {'loss': loss}

    def _validation_loop(self, val_dl) -> Dict[str, float]:
        """Executes validation after a training epoch.

        Puts the model in evaluation mode and processes all batches in the
        validation dataloader without computing gradients. Calculates the
        validation loss.

        Args:
            val_dl: DataLoader for validation data.

        Returns:
            Dict[str, float]: Dictionary containing validation metrics (val_loss).
        """
        self.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs in tqdm(val_dl, desc=f"Validation:"):
                x = inputs[0]['encoder_cont'].to(self.device)
                ae_output = self.model(x)
                val_loss += self.loss_fn(x, ae_output).item()
            val_loss /= len(val_dl)
        return {'val_loss': val_loss}

    def log_metrics(self, metrics, epoch: int):
        """Updates and logs metrics.

        Updates the internal metrics dictionary with new values, logs them to MLflow
        if an active run exists, and prints them to console.

        Args:
            metrics (Dict[str, float]): Dictionary of metrics to log.
            epoch (int): Current epoch number for MLflow logging.
        """
        self.metrics.update(metrics)
        if mlflow.active_run():
            mlflow.log_metrics(self.metrics, step=epoch)

        _prnt = [f'{key}: {str(round(value, 5))}' for key, value in self.metrics.items()]
        print(f"Metrics: {' '.join(_prnt)}")

    def _check_early_stopping(self, epoch, is_best) -> bool:
        """Checks if early stopping criteria are met based on current performance.

        Evaluates whether training should be stopped early by tracking improvement
        in the target metric. If no improvement is seen for a number of epochs
        exceeding the patience threshold, early stopping is triggered.

        Args:
            epoch (int): Current training epoch number.
            is_best (bool): Whether the current checkpoint is the best so far
                according to the target metric.

        Returns:
            bool: True if training should be stopped, False otherwise.
        """
        if self.early_stopping and epoch > self.es_warmup_epochs:  # Early stopping check
            if is_best:
                self.es_not_improved_epochs = 0
                return False
            else:
                self.es_not_improved_epochs += 1
                print(f'{self.hparams.target} have not increased for {self.es_not_improved_epochs} epochs.')
                print(f"{self.hparams.target}: "
                      f"best= {self.best_checkpoint['metrics'][self.hparams.target]:.5f} - "
                      f"current= {self.metrics[self.hparams.target]:.5f}\n"
                      f"")
            if self.es_not_improved_epochs >= self.es_patience_epochs:
                print(f'Early stopping. No improvement for {self.es_not_improved_epochs} epochs.')
                return True
        return False
