from copy import deepcopy
from typing import Optional, Dict, Any
from pathlib import Path

import mlflow
import torch
from torchinfo import summary

from privateer_ad import logger
from privateer_ad.config import (get_paths,
                                 get_model_config,
                                 get_training_config,
                                 get_data_config,
                                 get_privacy_config,
                                 get_mlflow_config
                                 )

from privateer_ad.etl.transform import DataProcessor
from privateer_ad.architectures import TransformerAD, TransformerADConfig
from privateer_ad.train.trainer import ModelTrainer
from privateer_ad.evaluate.evaluator import ModelEvaluator


class TrainPipeline:
    """
    Training pipeline.

    This class provides an interface for training models.
    """

    def __init__(
            self,
            partition_id: int = 0,
            partition: bool = False,
            dp_enabled: Optional[bool] = None,
            run_id: Optional[str] = None,
            parent_run_id: Optional[str] = None,
            config_overrides: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the training pipeline.

        Args:
            partition_id: ID for data partitioning in federated learning
            partition: Whether to enable data partitioning
            dp_enabled: Override for differential privacy setting
            run_id: MLFlow run ID
            parent_run_id: MLFlow parent run ID
            config_overrides: Dictionary of configuration overrides for testing
        """

        # Initialize instance variables
        self.partition_id = partition_id
        self.partition = partition
        self.config_overrides = config_overrides or {}

        # Inject configurations
        self._inject_configurations(dp_enabled)

        # Setup device
        self._setup_device()

        # Setup MLFlow
        self.run_name, self.run_id = self._setup_mlflow(run_id=run_id, parent_run_id=parent_run_id)
        mlflow.log_params({
            'client_id': self.partition_id,
            'partition_enabled': self.partition,
            'dp_enabled': self.privacy_config.dp_enabled if hasattr(self, 'privacy_config') else False
        })

        # Setup data processing
        self._setup_data_processing()

        # Setup model and optimizer
        self._setup_model()

        # Setup privacy if enabled
        self._setup_privacy()

        # Setup trainer components
        self._setup_trainer_components()

        # Log configuration summary
        self._log_configuration()

    def _inject_configurations(self, dp_enabled: Optional[bool] = None):
        """Inject all required configurations with optional overrides."""
        # Get all configurations
        self.paths_config = get_paths()
        self.model_config = get_model_config()
        self.training_config = get_training_config()
        self.data_config = get_data_config()
        self.privacy_config = get_privacy_config()
        self.mlflow_config = get_mlflow_config()

        # Apply configuration overrides
        if self.config_overrides:
            self._apply_config_overrides()

        # Override DP setting if explicitly provided
        if dp_enabled is not None:
            self.privacy_config.dp_enabled = dp_enabled

    def _apply_config_overrides(self):
        """Apply configuration overrides for testing or special scenarios."""
        for config_path, value in self.config_overrides.items():
            config_section, field_name = config_path.split('.')
            config_obj = getattr(self, f"{config_section}_config")
            setattr(config_obj, field_name, value)

    def _setup_device(self):
        """Setup compute device based on hardware configuration."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f'Using device: {self.device}')

    def _setup_mlflow(self,run_id=None, parent_run_id=None):
        """Setup MLFlow tracking"""
        mlflow.set_tracking_uri(self.mlflow_config.server_address)
        mlflow.set_experiment(self.mlflow_config.experiment_name)
        if mlflow.active_run():
            mlflow.end_run()
        mlflow.start_run(run_id=run_id, parent_run_id=parent_run_id)
        run_name = mlflow.active_run().info.run_name
        run_id = mlflow.active_run().info.run_id
        logger.info(f'Started MLFlow run: {run_name} (ID: {parent_run_id})')
        return run_name, run_id

    def _setup_data_processing(self):
        """Setup data processing components."""
        logger.info('Setup dataloaders.')
        self.data_processor = DataProcessor(partition=self.partition)

        # Create dataloaders with configuration
        self.train_dl = self.data_processor.get_dataloader(
            'train',
            batch_size=self.training_config.batch_size,
            seq_len=self.model_config.seq_len,
            partition_id=self.partition_id,
            only_benign=self.data_config.only_benign_for_training
        )

        self.val_dl = self.data_processor.get_dataloader(
            'val',
            batch_size=self.training_config.batch_size,
            partition_id=self.partition_id,
            seq_len=self.model_config.seq_len,
            only_benign=self.data_config.only_benign_for_training
        )

        self.test_dl = self.data_processor.get_dataloader(
            'test',
            batch_size=self.training_config.batch_size,
            partition_id=self.partition_id,
            seq_len=self.model_config.seq_len,
            only_benign=False
        )

        # Get sample for model initialization
        self.sample = next(iter(self.train_dl))[0]['encoder_cont'][:1].to('cpu')

    def _setup_model(self):
        """Setup model with proper configuration."""
        # Create model configuration
        model_config = TransformerADConfig(
            seq_len=self.model_config.seq_len,
            input_size=self.sample.shape[-1],
            num_layers=self.model_config.num_layers,
            hidden_dim=self.model_config.hidden_dim,
            latent_dim=self.model_config.latent_dim,
            num_heads=self.model_config.num_heads,
            dropout=self.model_config.dropout
        )

        # Ensure model class is safe for serialization
        torch.serialization.add_safe_globals([TransformerAD])

        # Create model instance
        self.model = TransformerAD(model_config)
        if self.privacy_config.dp_enabled:
            from opacus.validators import ModuleValidator
            ModuleValidator.validate(self.model, strict=True)
            self.model = ModuleValidator.fix(self.model)

        # Create optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.training_config.learning_rate
        )

        # Log model configuration if MLFlow is enabled
        if mlflow.active_run():
            mlflow.log_params(model_config.__dict__)
            mlflow.log_params(self.training_config.model_dump())

    def _setup_privacy(self):
        """Setup differential privacy if enabled."""
        if not self.privacy_config.dp_enabled:
            logger.info('Differential Privacy disabled.')
            return

        from opacus import PrivacyEngine

        logger.info('Differential Privacy enabled.')

        self.privacy_engine = PrivacyEngine(
            accountant='rdp',
            secure_mode=self.privacy_config.secure_mode
        )

        self.model, self.optimizer, self.train_dl = self.privacy_engine.make_private_with_epsilon(
            module=self.model,
            optimizer=self.optimizer,
            data_loader=self.train_dl,
            epochs=self.training_config.epochs,
            target_epsilon=self.privacy_config.target_epsilon,
            target_delta=self.privacy_config.target_delta,
            max_grad_norm=self.privacy_config.max_grad_norm
        )

        if mlflow.active_run():
            mlflow.log_params(self.privacy_config.model_dump())

    def _setup_trainer_components(self):
        """Setup trainer and evaluator components."""
        if self.training_config.early_stopping_enabled:
            logger.info('Early stopping enabled.')

    def _log_configuration(self):
        """Log model summary and configuration details."""
        model_summary = summary(
            model=self.model,
            input_data=self.sample,
            col_names=('input_size', 'output_size', 'num_params', 'params_percent')
        )

        if mlflow.active_run():
            mlflow.log_params({'device': str(self.device)})
            mlflow.log_text(str(model_summary), 'model_summary.txt')

    def train_model(self, start_epoch: int = 0) -> Dict[str, Any]:
        """
        Train the model with current configuration.

        Args:
            start_epoch: Starting epoch for training

        Returns:
            Best checkpoint information
        """
        logger.info('Start Training...')
        self.model.train()

        try:
            trainer = ModelTrainer(
                model=self.model,
                optimizer=self.optimizer,
                criterion=self.training_config.loss_function,
                device=self.device,
                training_config=self.training_config,  # Pass the entire config,
                privacy_engine = self.privacy_engine if hasattr(self, 'privacy_engine') else None
            )
        except Exception as e:
            raise ValueError(f'Error while initializing trainer: {e}')

        try:
            trainer.training(
                train_dl=self.train_dl,
                val_dl=self.val_dl,
                start_epoch=start_epoch
            )
        except KeyboardInterrupt:
            logger.warning('Training interrupted by user...')

        # Set model to best checkpoint
        self.model.load_state_dict(deepcopy(trainer.best_checkpoint['model_state_dict']))

        # Log model to MLFlow if enabled
        if mlflow.active_run():
            self._log_model_to_mlflow()

        logger.info('Training Finished.')

        return trainer.best_checkpoint

    def _log_model_to_mlflow(self):
        """Log trained model to MLFlow with proper signature."""
        self.model.to('cpu')
        sample = next(iter(self.train_dl))[0]['encoder_cont'][:1].to('cpu')

        _input = sample.to('cpu')
        _output = self.model(_input)

        if isinstance(_output, dict):
            _output = {key: val.detach().numpy() for key, val in _output.items()}
        else:
            _output = _output.detach().numpy()

        mlflow.pytorch.log_model(
            pytorch_model=self.model,
            artifact_path='model',
            signature=mlflow.models.infer_signature(
                model_input=_input.detach().numpy(),
                model_output=_output
            ),
            pip_requirements=str(self.paths_config.root_dir.joinpath('requirements.txt'))
        )

    def evaluate_model(self, step: int = 0) -> Dict[str, float]:
        """
        Evaluate the trained model.

        Args:
            step: Current training step/epoch

        Returns:
            Evaluation metrics
        """
        self.model.eval()
        evaluator = ModelEvaluator(
            criterion=self.training_config.loss_function,
            device=self.device
        )

        metrics, figures = evaluator.evaluate(
            self.model,
            self.test_dl,
            prefix='eval',
            step=step
        )

        metrics_logs = '\n'.join([f'{key}: {value}' for key, value in metrics.items()])
        logger.info(f'Test metrics:\n{metrics_logs}')
        return metrics

    def train_eval(self, start_epoch: int = 0) -> Dict[str, float]:
        """
        Complete training and evaluation pipeline.

        Args:
            start_epoch: Starting epoch for training

        Returns:
            Evaluation metrics
        """
        self.train_model(start_epoch=start_epoch)
        return self.evaluate_model(step=start_epoch)

def create_train_pipeline_from_config(
        config_file: Optional[Path] = None,
        **kwargs
) -> TrainPipeline:
    """
    Factory function to create TrainPipeline from configuration file.

    Args:
        config_file: Path to configuration file (optional)
        **kwargs: Additional parameters to override

    Returns:
        Configured TrainPipeline instance
    """
    if config_file:
        # Load configuration from file
        pass

    return TrainPipeline(**kwargs)

def main():
    """Main function with Fire integration for CLI usage."""
    from fire import Fire
    Fire(TrainPipeline)
