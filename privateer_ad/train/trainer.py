from copy import deepcopy
from typing import Dict, Any, Optional

import numpy as np
import torch
import mlflow
from tqdm import tqdm
from opacus.privacy_engine import PrivacyEngine
from sklearn.metrics import classification_report, roc_curve, roc_auc_score
from privateer_ad import logger
from privateer_ad.config import TrainingConfig, get_paths, get_privacy_config


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
            training_config: TrainingConfig,
            privacy_engine: Optional[PrivacyEngine] = None,
    ):
        """
        Initialize the ModelTrainer.

        Args:
            model: The PyTorch model to be trained
            optimizer: The optimizer used for training
            device: The device (CPU/GPU) where training will be performed
            training_config: Training configuration object with all parameters
        """
        logger.info('Instantiate ModelTrainer...')

        # Inject dependencies
        self.device = device
        self.privacy_config = get_privacy_config()
        self.paths_config = get_paths()
        self.training_config = training_config

        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.privacy_engine = privacy_engine

        # Initialize early stopping tracking
        if self.training_config.early_stopping_enabled:
            self.es_not_improved_epochs = 0
            self._validate_optimization_direction()

            if mlflow.active_run():
                mlflow.log_params({
                    'early_stopping_patience': self.training_config.early_stopping_patience,
                    'early_stopping_warmup': self.training_config.early_stopping_warmup,
                    'target_metric': self.training_config.target_metric,
                    'optimization_direction': self.training_config.optimization_direction
                })

        # Initialize metrics and best checkpoint
        self._initialize_tracking()

    def _validate_optimization_direction(self):
        """Validate the optimization direction setting."""
        valid_directions = ('maximize', 'minimize')
        if self.training_config.optimization_direction not in valid_directions:
            raise ValueError(
                f'optimization_direction must be one of {valid_directions}. '
                f'Current value: {self.training_config.optimization_direction}'
            )

    def _initialize_tracking(self):
        """Initialize metrics tracking and best checkpoint storage."""
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

        for epoch in range(start_epoch + 1, start_epoch + 1 + self.training_config.epochs):
            local_epoch = epoch - start_epoch

            # Training phase
            train_metrics = self._training_loop(epoch=epoch, train_dl=train_dl)

            # Validation phase
            val_metrics = self._validation_loop(val_dl)

            # Log metrics
            self.log_metrics(train_metrics | val_metrics, epoch)

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

        return self.best_checkpoint

    def _training_loop(self, epoch: int, train_dl) -> Dict[str, float]:
        """
        Execute one training epoch.

        Args:
            epoch: Current epoch number
            train_dl: DataLoader for training data

        Returns:
            Dict[str, float]: Dictionary containing training metrics (loss)
        """
        self.model.train()
        total_loss = 0.0
        self.model.to(self.device)
        progress_bar = tqdm(
            train_dl,
            desc=f'Epoch {epoch} - Train'
        )
        loss_fn = getattr(torch.nn, self.training_config.loss_function)(reduction='mean')
        for inputs in progress_bar:
            x = inputs[0]['encoder_cont'].to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            output = self.model(x)

            # Compute loss
            batch_loss = loss_fn(x, output)

            # Backward pass
            batch_loss.backward()
            self.optimizer.step()
            total_loss += batch_loss.item()

            # Update progress bar
            progress_bar.set_postfix({'loss': batch_loss.item()})
        avg_loss = total_loss / len(train_dl)
        return {'loss': avg_loss
                }

    def _validation_loop(self, val_dl) -> Dict[str, float]:
        """
        Execute validation after a training epoch.

        Args:
            val_dl: DataLoader for validation data

        Returns:
            Dict[str, float]: Dictionary containing validation metrics (val_loss)
        """
        self.model.eval()
        total_loss = 0.0
        metrics_results = {}

        with torch.no_grad():
            progress_bar = tqdm(val_dl, desc=' ' * 9 + 'Validation')
            loss_fn = getattr(torch.nn, self.training_config.loss_function)(reduction='none')
            rec_errors = []
            y_true = []
            for inputs in progress_bar:
                x = inputs[0]['encoder_cont'].to(self.device)
                targets = np.squeeze(inputs[1][0])
                output = self.model(x)  # all samples
                loss = loss_fn(x, output)  # reconstruction loss
                loss_per_sample = loss.mean(dim=(1, 2))
                batch_loss = loss_per_sample.mean().item()
                total_loss += batch_loss
                y_true.extend(targets.tolist())
                rec_errors.extend(loss.mean(dim=(1, 2)).tolist())  # reconstruction error per sample
                progress_bar.set_postfix({'val_loss': batch_loss})

        rec_errors = np.array(rec_errors)
        y_true = np.array(y_true, dtype=int)

        benign_rec_errors = rec_errors[y_true == 0]
        malicious_rec_errors = rec_errors[y_true == 1]
        if len(malicious_rec_errors) <= len(benign_rec_errors):
            n_samples_per_class = len(malicious_rec_errors)
            np.random.shuffle(benign_rec_errors)
            benign_rec_errors = benign_rec_errors[:n_samples_per_class]
        else:
            n_samples_per_class = len(benign_rec_errors)
            np.random.shuffle(malicious_rec_errors)
            malicious_rec_errors = malicious_rec_errors[:n_samples_per_class]

        balanced_rec_errors = np.concatenate([benign_rec_errors, malicious_rec_errors])
        balanced_y_true = np.concatenate([np.zeros_like(benign_rec_errors), np.ones_like(malicious_rec_errors)])

        fpr, tpr, thresholds = roc_curve(y_true=balanced_y_true, y_score=balanced_rec_errors)
        optimal_idx = np.argmin(np.sqrt(np.power(fpr, 2) + np.power(1 - tpr, 2)))
        threshold = thresholds[optimal_idx]

        balanceed_y_pred = (balanced_rec_errors >= threshold).astype(int)
        y_pred = (rec_errors >= threshold).astype(int)
        target_names = ['benign', 'malicious']
        print('Balanced', classification_report(y_true=balanced_y_true,
                                                y_pred=balanceed_y_pred,
                                                target_names=target_names))
        print('All data', classification_report(y_true=y_true,
                                                y_pred=y_pred,
                                                target_names=target_names))

        metrics_results.update(classification_report(y_true=balanced_y_true,
                                                     y_pred=balanceed_y_pred,
                                                     target_names=target_names,
                                                     output_dict=True)['macro avg'])

        metrics_results.update({'loss': np.mean(rec_errors),
                                'roc_auc': roc_auc_score(y_true=balanced_y_true, y_score=balanced_rec_errors)})
        print('Unbalanced roc', roc_auc_score(y_true=y_true, y_score=rec_errors))
        metrics_results = {f'val_' + k: v for k, v in metrics_results.items()}
        metrics_results['threshold'] = threshold
        return metrics_results

    def _is_best_checkpoint(self) -> bool:
        """
        Determine if the current metrics represent the best checkpoint so far.

        Returns:
            bool: True if current checkpoint is the best
        """
        current_value = self.metrics[self.training_config.target_metric]
        direction = self.training_config.optimization_direction
        best_value = self.best_checkpoint['metrics'].get(self.training_config.target_metric,
                                                         np.inf if direction == 'minimize' else -np.inf)

        if self.training_config.optimization_direction == 'maximize':
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

        # Log to MLFlow if available
        if mlflow.active_run():
            mlflow.log_metrics(self.metrics, step=epoch)
            # Log privacy metrics if DP is enabled
            if self.privacy_engine:
                mlflow.log_metrics(
                    {'epsilon': self.privacy_engine.get_epsilon(self.privacy_config.target_delta)},
                    step=epoch
                )

        # Format and log to console
        formatted_metrics = [
            f'{key}: {str(round(value, 5))}'
            for key, value in self.metrics.items()
        ]
        logger.info(f'Metrics: {" ".join(formatted_metrics)}')

    def _check_early_stopping(self, epoch: int, is_best: bool) -> bool:
        """
        Check if early stopping criteria are met.

        Args:
            epoch: Current training epoch number
            is_best: Whether the current checkpoint is the best so far

        Returns:
            bool: True if training should be stopped, False otherwise
        """
        if not self.training_config.early_stopping_enabled:
            return False

        # Skip early stopping during warmup period
        if epoch <= self.training_config.early_stopping_warmup:
            return False

        if is_best:
            self.es_not_improved_epochs = 0
            return False
        else:
            self.es_not_improved_epochs += 1

            # Log early stopping status
            logger.warning(
                f'{self.training_config.target_metric} has not improved for '
                f'{self.es_not_improved_epochs} epochs.'
            )
            logger.warning(
                f'{self.training_config.target_metric}: '
                f'best={self.best_checkpoint['metrics'][self.training_config.target_metric]:.5f}'
                f' - current={self.metrics[self.training_config.target_metric]:.5f}\n'
            )

        # Check if patience limit reached
        if self.es_not_improved_epochs >= self.training_config.early_stopping_patience:
            logger.warning(
                f'Early stopping triggered. No improvement for '
                f'{self.es_not_improved_epochs} epochs.'
            )
            return True

        return False

    def get_training_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the training process.

        Returns:
            Dict containing training summary information
        """
        return {
            'best_epoch': self.best_checkpoint['epoch'],
            'best_metrics': self.best_checkpoint['metrics'],
            'total_epochs_trained': self.best_checkpoint['epoch'],
            'early_stopping_triggered': (
                    self.training_config.early_stopping_enabled and
                    self.es_not_improved_epochs >= self.training_config.early_stopping_patience
            ),
            'configuration': self.training_config.model_dump()
        }