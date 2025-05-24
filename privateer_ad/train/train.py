from copy import deepcopy
from datetime import datetime
from typing import Optional, Dict, Any
from pathlib import Path

import mlflow
import torch
from torchinfo import summary
from mlflow.entities import RunStatus

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
            run_name: Optional[str] = None,
            partition_id: int = 0,
            partition: bool = False,
            dp_enabled: Optional[bool] = None,
            nested: bool = False,
            config_overrides: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the training pipeline.

        Args:
            run_name: Optional custom run name for this training session
            partition_id: ID for data partitioning in federated learning
            partition: Whether to enable data partitioning
            dp_enabled: Override for differential privacy setting
            nested: Whether this is a nested run (e.g., in federated learning)
            config_overrides: Dictionary of configuration overrides for testing
        """

        # Initialize instance variables
        self.run_name = run_name
        self.partition_id = partition_id
        self.partition = partition
        self.nested = nested
        self.config_overrides = config_overrides or {}

        # Inject configurations
        self._inject_configurations(dp_enabled)

        # Setup device
        self._setup_device()

        # Setup MLFlow if enabled
        self._setup_mlflow()

        # Setup directories
        self._setup_directories()

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

    def _setup_mlflow(self):
        """Setup MLFlow tracking if enabled."""
        if not self.mlflow_config.enabled:
            return

        logger.info('Initialize MLFlow')
        mlflow.set_tracking_uri(self.mlflow_config.server_address)
        mlflow.set_experiment(self.mlflow_config.experiment_name)

        self.parent_run_id = None
        self.run_id = None

        if not self.run_name:
            self.run_name = datetime.now().strftime('%Y%m%d-%H%M%S')

        if self.privacy_config.dp_enabled:
            self.run_name += '-dp'

        if mlflow.active_run():
            logger.info(f"Found active run {mlflow.active_run().info.run_id}, ending it")
            mlflow.end_run()

        if self.nested and self.mlflow_config.server_run_name:
            # Use a local copy instead of modifying the shared configuration
            self.local_server_run_name = self.mlflow_config.server_run_name
            if self.privacy_config.dp_enabled and not self.local_server_run_name.endswith('-dp'):
                self.local_server_run_name += '-dp'
            self._setup_nested_run()
        self._start_mlflow_run()

    def _setup_nested_run(self):
        """Setup nested MLFlow run for federated learning scenarios."""

        parent_runs = mlflow.search_runs(
            experiment_names=[self.mlflow_config.experiment_name],
            filter_string=f'tags.mlflow.runName = \'{self.local_server_run_name}\'',
            max_results=1
        )

        if len(parent_runs) > 0:
            self.parent_run_id = parent_runs.iloc[0].run_id
            if RunStatus.is_terminated(mlflow.get_run(self.parent_run_id).info.status):
                mlflow.start_run(run_id=self.parent_run_id)

    def _start_mlflow_run(self):
        """Start or resume MLFlow run."""
        runs = mlflow.search_runs(
            experiment_names=[self.mlflow_config.experiment_name],
            filter_string=f'tags.mlflow.runName = \'{self.run_name}\'',
            max_results=1
        )

        if len(runs) > 0:
            self.run_id = runs.iloc[0].run_id
            if not RunStatus.is_terminated(mlflow.get_run(self.run_id).info.status):
                mlflow.MlflowClient().set_terminated(run_id=self.run_id)

        mlflow.start_run(run_id=self.run_id, run_name=self.run_name, parent_run_id=self.parent_run_id)

        self.run_id = mlflow.active_run().info.run_id
        self.run_name = mlflow.active_run().info.run_name
        logger.info(f'Run with name {self.run_name} started.')

    def _setup_directories(self):
        """Setup experiment directories."""
        if self.nested and self.local_server_run_name:
            self.trial_dir = self.paths_config.experiments_dir.joinpath(self.local_server_run_name, self.run_name)
        else:
            self.trial_dir = self.paths_config.experiments_dir.joinpath(self.run_name)

        self.trial_dir.mkdir(parents=True, exist_ok=True)

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

        # Save locally
        with self.trial_dir.joinpath('model_summary.txt').open('w') as f:
            f.write(str(model_summary))

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

        # Save model locally
        torch.save(self.model.state_dict(), self.trial_dir.joinpath('model.pt'))
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

        # Save figures locally
        for name, fig in figures.items():
            fig.savefig(self.trial_dir.joinpath(f'{name}.png'))

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
