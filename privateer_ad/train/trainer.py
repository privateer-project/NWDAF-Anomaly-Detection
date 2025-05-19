from copy import deepcopy
from typing import Dict, Any

from tqdm import tqdm
import mlflow
import torch

from mlflow.models import infer_signature

from privateer_ad.config import HParams, EarlyStoppingConfig, setup_logger, PathsConf

logger = setup_logger('trainer')

class ModelTrainer:
    """Trains a PyTorch model with early stopping capability.

     This class handles the training workflow for autoencoder models, including
     training loop, validation, metrics tracking, and early stopping functionality.
     It also provides integration with MLFlow for experiment tracking.

     Attributes:
         model: The PyTorch model to be trained.
         optimizer: The optimizer used for training.
         loss_fn: The loss function used for training.
         device: The device (CPU/GPU) where training will be performed.
         hparams (HParams): Hyperparameters for training.
         metrics (Dict[str, float]): Dictionary to track training metrics.
         best_checkpoint (Dict[str, Any]): Dictionary to store the best model checkpoint.
     """
    es_conf = EarlyStoppingConfig()

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
        self.device = device
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.loss_fn = getattr(torch.nn, criterion)(reduction='mean')
        self.hparams = hparams
        self.paths = PathsConf()
        if self.hparams.early_stopping:
            logger.info('Early stopping enabled.')
            self.es_not_improved_epochs = 0

            if self.hparams.direction not in ('maximize', 'minimize'):
                raise ValueError(f'direction must be `maximize` or `minimize`. Current value: {self.hparams.direction}')
            if mlflow.active_run():
                mlflow.log_params(self.es_conf.__dict__)

        # Initialize metrics dict and best_checkpoint dict
        metrics = {'loss': float('inf'), 'val_loss': float('inf')}
        self.metrics = metrics.copy()
        self.best_checkpoint: Dict[str, Any] = {'epoch': 0,
                                                'model_state_dict': self.model.state_dict(),
                                                'optimizer_state_dict': self.optimizer.state_dict(),
                                                'metrics': {k: v for k, v in metrics.copy().items()}
                                                }

    def training(self, train_dl, val_dl, start_epoch=0):
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
            start_epoch:

        Returns:
            Dict[str, Any]: The best checkpoint dictionary containing:
                - model_state_dict: State dict of the best model
                - optimizer_state_dict: State dict of the optimizer at best point
                - metrics: Dictionary of metrics at best point
                - epoch: Epoch number when best model was achieved
        """
        try:
            for epoch in range(start_epoch, start_epoch + self.hparams.epochs):
                local_epoch = epoch - start_epoch
                train_metrics = self._training_loop(epoch=local_epoch, train_dl=train_dl)
                val_metrics = self._validation_loop(val_dl)
                self.log_metrics(train_metrics | val_metrics, epoch)
                current_value = self.metrics[self.hparams.target]
                best_value = self.best_checkpoint['metrics'][self.hparams.target]
                if self.hparams.direction == 'maximize':
                    is_best = current_value >= best_value
                else:
                    is_best = current_value <= best_value
                if is_best:
                    self.best_checkpoint.update({
                        'metrics': deepcopy(self.metrics),
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict()
                    })
                if self._check_early_stopping(epoch=local_epoch, is_best=is_best): # Early stopping check
                    break
        except KeyboardInterrupt: # Break training loop when Ctrl+C pressed (Manual early stopping)
            logger.warning('Training interrupted by user...')

        if mlflow.active_run():
            # log model with signature to mlflow
            self.model.to('cpu')
            sample = next(iter(train_dl))[0]['encoder_cont'][:1].to('cpu')

            _input = sample.to('cpu')
            _output = self.model(_input)
            if isinstance(_output, dict):
                _output = {key: val.detach().numpy() for key, val in _output.items()}
            else:
                _output = _output.detach().numpy()

            mlflow.pytorch.log_model(pytorch_model=self.model,
                                     artifact_path='model',
                                     signature=infer_signature(model_input=_input.detach().numpy(),
                                                               model_output=_output),
                                     pip_requirements=self.paths.root.joinpath('requirements.txt').as_posix())

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

        for inputs in tqdm(train_dl, desc=f'Epoch {epoch}/{self.hparams.epochs} - Train:'):
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
            for inputs in tqdm(val_dl, desc='Validation:'):
                x = inputs[0]['encoder_cont'].to(self.device)
                ae_output = self.model(x)
                val_loss += self.loss_fn(x, ae_output).item()
            val_loss /= len(val_dl)
        return {'val_loss': val_loss}

    def log_metrics(self, metrics, epoch: int):
        """Updates and logs metrics.

        Updates the internal metrics dictionary with new values, logs them to MLFlow
        if an active run exists, and prints them to console.

        Args:
            metrics (Dict[str, float]): Dictionary of metrics to log.
            epoch (int): Current epoch number for MLFlow logging.
        """
        self.metrics.update(metrics)
        if mlflow.active_run():
            mlflow.log_metrics(self.metrics, step=epoch)
        _prnt = [f'{key}: {str(round(value, 5))}' for key, value in self.metrics.items()]
        logger.info(f'Metrics: {" ".join(_prnt)}')

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
        if self.hparams.early_stopping and epoch > self.es_conf.es_warmup_epochs:  # Early stopping check
            if is_best:
                self.es_not_improved_epochs = 0
                return False
            else:
                self.es_not_improved_epochs += 1
                logger.warning(f'{self.hparams.target} have not increased for {self.es_not_improved_epochs} epochs.')
                logger.warning(f'{self.hparams.target}: '
                               f'best= {self.best_checkpoint["metrics"][self.hparams.target]:.5f} - '
                               f'current= {self.metrics[self.hparams.target]:.5f}\n')
            if self.es_not_improved_epochs >= self.es_conf.es_patience_epochs:
                logger.warning(f'Early stopping. No improvement for {self.es_not_improved_epochs} epochs.')
                return True
        return False
