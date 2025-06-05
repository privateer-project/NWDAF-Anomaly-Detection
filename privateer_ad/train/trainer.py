import logging

from copy import deepcopy
from pprint import pprint
from typing import List, Dict, Any

import mlflow
import numpy as np
import torch

from tqdm import tqdm
from sklearn.metrics import classification_report, roc_curve, roc_auc_score

from privateer_ad.config import TrainingConfig


class ModelTrainer:
    """
    Trains a PyTorch model with early stopping capability.

    This class handles the training workflow for models, including
    training loop, validation, metrics tracking, and early stopping functionality.
    It integrates with MLFlow for experiment tracking and uses proper configuration
    dependency injection.
    """

    def __init__(
            self,
            model: torch.nn.Module,
            optimizer: torch.optim.Optimizer,
            device: torch.device,
            training_config: TrainingConfig = None):
        """
        Initialize the ModelTrainer.

        Args:
            model: The PyTorch model to be trained
            optimizer: The optimizer used for training
            device: The device (CPU/GPU) where training will be performed
        """
        logging.info('Instantiate ModelTrainer...')

        self.model = model
        self.optimizer = optimizer
        self.device = device

        # Inject dependencies
        self.training_config = training_config or TrainingConfig()

        self.model = self.model.to(self.device)
        self.loss_fn = getattr(torch.nn, self.training_config.loss_fn_name)(reduction='none')

        # Initialize early stopping tracking
        if self.training_config.es_enabled:
            self.es_not_improved_epochs = 0
            valid_directions = ('maximize', 'minimize')
            if self.training_config.direction not in valid_directions:
                raise ValueError(f'optimization_direction must be one of {valid_directions}. '
                                 f'Current value: {self.training_config.direction}')

        # Initialize metrics and best checkpoint
        self.metrics = {}
        self.best_checkpoint: Dict[str, Any] = {
            'epoch': 0,
            'model_state_dict': deepcopy(self.model.state_dict()),
            'optimizer_state_dict': deepcopy(self.optimizer.state_dict()),
            'metrics': {}
        }

    def training(self, train_dl, val_dl, start_epoch: int = 0) -> Dict[str, Any]:
        """
        Execute the full training process with validation and early stopping.

        Trains the model for the specified number of epochs or until early stopping
        criteria are met. For each epoch, runs a training loop followed by a validation
        loop, logs metrics, and tracks the best model state.

        Args:
            train_dl: DataLoader for training data
            val_dl: DataLoader for validation data
            start_epoch: Starting epoch number

        Returns:
            Dict[str, Any]: The best checkpoint dictionary containing:
                - model_state_dict: State dict of the best model
                - optimizer_state_dict: State dict of the optimizer at best point
                - metrics: Dictionary of metrics at best point
                - epoch: Epoch number when best model was achieved
        """
        try:
            for epoch in range(start_epoch, start_epoch + self.training_config.epochs):
                local_epoch = epoch - start_epoch

                # Training phase
                train_report = self._training_loop(epoch=epoch, train_dl=train_dl)

                # Validation phase
                val_report = self._validation_loop(val_dl=val_dl)

                # Log metrics
                self.log_metrics(train_report | val_report, epoch)

                # Check if this is the best model so far
                is_best = self._is_best_checkpoint()

                if is_best:
                    self.best_checkpoint.update({
                        'metrics': deepcopy(self.metrics),
                        'epoch': epoch,
                        'model_state_dict': deepcopy(self.model.state_dict()),
                        'optimizer_state_dict': deepcopy(self.optimizer.state_dict())
                    })

                # Early stopping check
                if self._check_early_stopping(epoch=local_epoch, is_best=is_best):
                    break
        except KeyboardInterrupt:
            logging.warning('Training interrupted by user...')
        pprint(self.get_training_summary())
        return self.best_checkpoint

    def _training_loop(self, epoch: int, train_dl) -> Dict[str, float]:
        self.model.train()
        self.model.to(self.device)

        total_loss = 0.0
        progress_bar = tqdm(train_dl, desc=f'Epoch {epoch} - Train')

        for inputs in progress_bar:
            x = inputs[0]['encoder_cont'].to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            output = self.model(x)

            # Compute loss
            batch_loss = torch.mean(self.loss_fn(x, output))

            # Backward pass
            batch_loss.backward()
            self.optimizer.step()
            total_loss += batch_loss.item()

            # Update progress bar
            progress_bar.set_postfix({'loss': batch_loss.item()})
        avg_loss = total_loss / len(train_dl)
        return {'loss': avg_loss}

    def _validation_loop(self, val_dl) -> Dict[str, float]:
        self.model.eval()

        with torch.no_grad():
            progress_bar = tqdm(val_dl, desc=' ' * 5 + 'Validation')
            losses: List[torch.Tensor] | torch.Tensor = []
            y_true: List[torch.Tensor] | torch.Tensor = []

            for inputs in progress_bar:
                x = inputs[0]['encoder_cont'].to(self.device)
                targets = np.squeeze(inputs[1][0])
                output = self.model(x)  # all samples
                batch_losses = self.loss_fn(x, output)  # reconstruction loss
                batch_loss_per_sample = torch.mean(input=batch_losses, dim=(1, 2))
                y_true.append(targets)
                losses.append(batch_loss_per_sample)  # reconstruction error per sample
                progress_bar.set_postfix({'val_loss': torch.mean(batch_loss_per_sample).item()})

            losses = torch.concatenate(losses)
            y_true = torch.concatenate(y_true)

        # Split benign and malicious samples
        benign_rec_errors = losses[y_true == 0]
        malicious_rec_errors = losses[y_true == 1]

        # get min number of samples per class
        n_samples_per_class = min(map(len, [benign_rec_errors, malicious_rec_errors]))

        # shuffle samples
        benign_rec_errors = benign_rec_errors[torch.randperm(benign_rec_errors.size()[0])]
        malicious_rec_errors = malicious_rec_errors[torch.randperm(malicious_rec_errors.size()[0])]

        # select min number of samples per class
        benign_rec_errors = benign_rec_errors[:n_samples_per_class]
        malicious_rec_errors = malicious_rec_errors[:n_samples_per_class]

        # concatenate benign and malicious samples to get balanced dataset
        balanced_rec_errors = torch.concatenate([benign_rec_errors, malicious_rec_errors])
        balanced_y_true = torch.concatenate([torch.zeros(n_samples_per_class, dtype=torch.int32),
                                             torch.ones(n_samples_per_class, dtype=torch.int32)])

        # Offload to CPU
        losses = losses.cpu()
        y_true = y_true.cpu()
        balanced_y_true = balanced_y_true.cpu()
        balanced_rec_errors = balanced_rec_errors.cpu()

        # Compute threshold
        fpr, tpr, thresholds = roc_curve(y_true=balanced_y_true, y_score=balanced_rec_errors)
        optimal_idx = np.argmin(np.sqrt(np.power(fpr, 2) + np.power(1 - tpr, 2)))
        threshold = thresholds[optimal_idx]

        # Compute metrics
        balanced_y_pred = torch.where(balanced_rec_errors >= threshold, 1, 0)
        y_pred = torch.where(losses >= threshold, 1, 0)

        target_names = ['benign', 'malicious']
        balanced_metrics = classification_report(y_true=balanced_y_true,
                                                 y_pred=balanced_y_pred,
                                                 target_names=target_names,
                                                 output_dict=True)['macro avg']

        unbalanced_metrics = classification_report(y_true=y_true,
                                                   y_pred=y_pred,
                                                   target_names=target_names,
                                                   output_dict=True)['macro avg']

        balanced_roc = roc_auc_score(y_true=balanced_y_true, y_score=balanced_rec_errors)
        roc = roc_auc_score(y_true=y_true, y_score=losses)

        report_dict = {'loss': torch.mean(losses).item(),
                       'threshold': float(threshold)}

        balanced_metrics = {'balanced_' + k: v for k, v in balanced_metrics.items()}
        balanced_metrics['balanced_roc'] = balanced_roc

        unbalanced_metrics = {'unbalanced_' + k: v for k, v in unbalanced_metrics.items()}
        unbalanced_metrics['unbalanced_roc'] = roc

        report_dict |= balanced_metrics | unbalanced_metrics
        report_dict = {f'val_' + k: v for k, v in report_dict.items()}

        return report_dict

    def _is_best_checkpoint(self) -> bool:
        """
        Determine if the current metrics represent the best checkpoint so far.

        Returns:
            bool: True if current checkpoint is the best 'metrics'
        """
        current_value = self.metrics[self.training_config.target_metric]
        best_value = self.best_checkpoint['metrics'].get(self.training_config.target_metric,
                                                         -np.inf if self.training_config.direction == 'maximize'
                                                         else np.inf)

        if self.training_config.direction == 'maximize':
            return current_value > best_value
        else:  # minimize
            return current_value < best_value

    def log_metrics(self, metrics: Dict[str, float], epoch: int):
        """
        Update and log metrics to various tracking systems.

        Args:
            metrics: Dictionary of metrics to log
            epoch: Current epoch number
        """
        # Update internal metrics
        self.metrics.update(metrics)

        # Log to MLFlow
        mlflow.log_metrics(self.metrics, step=epoch)

        # Format and log to console
        formatted_metrics = [f'{key}: {str(round(value, 5))}' for key, value in self.metrics.items()]
        logging.info(f'Metrics: {" ".join(formatted_metrics)}')

    def _check_early_stopping(self, epoch: int, is_best: bool) -> bool:
        """
        Check if early stopping criteria are met.

        Args:
            epoch: Current training epoch number
            is_best: Whether the current checkpoint is the best so far

        Returns:
            bool: True if training should be stopped, False otherwise
        """
        if not self.training_config.es_enabled:
            return False

        # Skip early stopping during warmup period
        if epoch <= self.training_config.es_warmup:
            return False

        if is_best:
            self.es_not_improved_epochs = 0
            return False
        else:
            self.es_not_improved_epochs += 1

            # Log early stopping status
            logging.warning(
                f'{self.training_config.target_metric} has not improved for '
                f'{self.es_not_improved_epochs} epochs.'
            )
            logging.warning(
                f"{self.training_config.target_metric}: "
                f"best={self.best_checkpoint['metrics'][self.training_config.target_metric]:.5f}"
                f" - current={self.metrics[self.training_config.target_metric]:.5f}\n"
            )

        # Check if patience limit reached
        if self.es_not_improved_epochs >= self.training_config.es_patience:
            logging.warning(f'Early stopping triggered. No improvement for {self.es_not_improved_epochs} epochs.')
            return True

        return False

    def get_training_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the training process.

        Returns:
            Dict containing training summary information
        """
        return {'best_epoch': self.best_checkpoint['epoch'],
                'best_metrics': self.best_checkpoint['metrics'],
                'total_epochs_trained': self.best_checkpoint['epoch'],
                'early_stopping_triggered': (self.training_config.es_enabled and
                                             self.es_not_improved_epochs >= self.training_config.es_patience),
                'configuration': self.training_config.model_dump()}
