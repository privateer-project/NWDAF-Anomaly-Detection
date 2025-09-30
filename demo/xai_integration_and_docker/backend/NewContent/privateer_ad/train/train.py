import logging

from typing import Dict, Any, Tuple

import mlflow
import torch

from torchinfo import summary

from privateer_ad.config import ModelConfig, MLFlowConfig, DataConfig, TrainingConfig, PathConfig, PrivacyConfig
from privateer_ad.etl.transform import DataProcessor
from privateer_ad.architectures import TransformerAD
from privateer_ad.train.trainer import ModelTrainer
from privateer_ad.evaluate.evaluator import ModelEvaluator
from privateer_ad.utils import log_model


class TrainPipeline:
    """
    Comprehensive training pipeline for privacy-preserving anomaly detection models.

    Orchestrates the complete machine learning workflow from data processing through
    model training to evaluation and artifact logging. Supports differential privacy,
    experiment tracking, and configurable training strategies with proper resource
    management and cleanup.

    The pipeline handles complex configurations including federated learning setups,
    privacy-preserving mechanisms, and comprehensive experiment tracking through MLflow
    integration. Designed for both standalone training and integration within larger
    federated learning systems.

    Attributes:
        device (torch.device): Computational device for training operations
        model (TransformerAD): The anomaly detection model being trained
        optimizer: PyTorch optimizer for training
        train_dl, val_dl, test_dl: Data loaders for different training phases
    """

    def __init__(
            self,
            paths_config: PathConfig | None = None,
            mlflow_config: MLFlowConfig | None = None,
            training_config: TrainingConfig | None = None,
            data_config: DataConfig | None = None,
            model_config: ModelConfig | None = None,
            privacy_config: PrivacyConfig | None = None
    ):
        """
        Initialize training pipeline with comprehensive configuration management.

        Sets up all necessary components including data processing, model architecture,
        privacy mechanisms, and experiment tracking. Handles device selection and
        resource allocation with proper error handling and cleanup.

        Args:
            paths_config (PathConfig, optional): File system path configurations
            mlflow_config (MLFlowConfig, optional): Experiment tracking settings
            training_config (TrainingConfig, optional): Training parameters and optimization
            data_config (DataConfig, optional): Data processing and loading configuration
            model_config (ModelConfig, optional): Model architecture specifications
            privacy_config (PrivacyConfig, optional): Differential privacy settings
        """
        logging.info('Initialize training pipeline.')
        # Setup configurations
        self.paths_config = paths_config or PathConfig()
        self.mlflow_config = mlflow_config or MLFlowConfig()
        self.training_config = training_config or TrainingConfig()
        self.data_config = data_config or DataConfig()
        self.model_config = model_config or ModelConfig()
        self.privacy_config = privacy_config or PrivacyConfig()

        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f'Using device: {self.device}')

        self.train_dl = None
        self.val_dl = None
        self.test_dl = None

        try:
            self._setup_mlflow()
            self._setup_data_and_model()
        except Exception as e:
            self._cleanup_mlflow()
            raise e

    def train_model(self, start_epoch: int = 0) -> Dict[str, Any]:
        """
        Execute model training with comprehensive configuration logging and privacy support.

        Runs the complete training process including differential privacy application
        when enabled, progress tracking, and best model checkpoint management. Integrates
        with experiment tracking for detailed training analysis.

        Args:
            start_epoch (int): Starting epoch for training continuation or federated rounds

        Returns:
            Dict[str, Any]: Best checkpoint containing model state and training metrics
        """
        try:
            self.model.train()
            trainer = ModelTrainer(model=self.model,
                                   optimizer=self.optimizer,
                                   device=self.device,
                                   training_config=self.training_config)

            # Log configuration if MLFlow is enabled
            if mlflow.active_run():
                mlflow.log_params({'device': str(self.device)})
                mlflow.log_params(self.training_config.model_dump())
                mlflow.log_params(self.data_config.model_dump())
                mlflow.log_params(self.model_config.model_dump())
                mlflow.log_params(self.privacy_config.model_dump())

            best_checkpoint = trainer.training(train_dl=self.train_dl, val_dl=self.val_dl, start_epoch=start_epoch)
            if self.privacy_config.dp_enabled:
                best_checkpoint['metrics']['epsilon'] = self.privacy_engine.get_epsilon(self.privacy_config.target_delta)

            self.model = trainer.model
            logging.info('Training Finished.')
            return best_checkpoint

        except Exception as e:
            logging.error(f"Training failed: {e}")
            self._cleanup_mlflow()
            raise e

    def evaluate_model(self, step: int = 0) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Comprehensive model evaluation with metrics computation and visualization.

        Performs thorough model assessment on test data, generating performance
        metrics and visualizations for analysis and reporting.

        Args:
            step (int): Current training step for tracking evaluation progression

        Returns:
            Tuple[Dict[str, Any], Dict[str, Any]]: Evaluation metrics and visualization figures
        """
        try:
            self.model.eval()
            evaluator = ModelEvaluator(loss_fn=self.training_config.loss_fn_name, device=self.device)
            _, _ = evaluator.evaluate(self.model, self.val_dl, prefix='eval', step=step)
            metrics, figures = evaluator.evaluate(self.model, self.test_dl, prefix='test', step=step)
            return metrics, figures
        except Exception as e:
            logging.error(f"Evaluation failed: {e}")
            self._cleanup_mlflow()
            raise e

    def train_eval(self, start_epoch: int = 0) -> Tuple[Dict[str, float], Dict[str, Any]]:
        """
        Complete training and evaluation workflow with model logging.

        Executes full machine learning pipeline from training through evaluation
        to model artifact preservation. Handles champion model selection and
        comprehensive result logging for experiment tracking.

        Args:
            start_epoch (int): Starting epoch for training process

        Returns:
            Tuple[Dict[str, float], Dict[str, Any]]: Combined metrics and visualizations
        """
        try:
            best_checkpoint = self.train_model(start_epoch=start_epoch)
            test_metrics, figures = self.evaluate_model(step=start_epoch)

            best_checkpoint['metrics'].update(test_metrics)

            try:
                self.model.to('cpu')

                # Get current experiment ID
                current_experiment = mlflow.get_experiment_by_name(self.mlflow_config.experiment_name)
                experiment_id = current_experiment.experiment_id if current_experiment else None

                log_model(
                    model=self.model,
                    model_name=self.model_config.model_name,
                    sample=self.sample,
                    direction=self.training_config.direction,
                    target_metric=self.training_config.target_metric,
                    current_metrics=best_checkpoint['metrics'],
                    experiment_id=experiment_id,
                    pip_requirements=self.paths_config.requirements_file.as_posix()
                )

                logging.info(f"Model {self.model_config.model_name} logged successfully to MLFlow")

            except Exception as e:
                logging.error(f"Failed to log model with champion tagging: {e}")
            self.model.to('cpu')

            return best_checkpoint['metrics'], figures
        finally:
            self._cleanup_mlflow()
            self._cleanup_resources()

    def _setup_data_and_model(self):
        """Configure data processing pipeline and model architecture with privacy support."""
        # Setup datasets
        logging.info('Setup dataloaders.')

        self.data_proc = DataProcessor(data_config=self.data_config)
        self.model_config.input_size = len(self.data_proc.input_features)
        self.model_config.seq_len = self.data_proc.data_config.seq_len

        # Create dataloaders with configuration
        self.train_dl = self.data_proc.get_dataloader('train', only_benign=True, train=True)
        self.val_dl = self.data_proc.get_dataloader('val', only_benign=False, train=False)
        self.test_dl = self.data_proc.get_dataloader('test', only_benign=False, train=False)

        # Setup model and optimizer
        torch.serialization.add_safe_globals([TransformerAD])

        self.sample = next(iter(self.train_dl))[0]['encoder_cont'][:1].to('cpu')
        self.model = TransformerAD(self.model_config)
        if mlflow.active_run():
            mlflow.log_text(str(summary(model=self.model,
                                        input_data=self.sample,
                                        col_names=('input_size', 'output_size', 'num_params', 'params_percent'))),
                            'model_summary.txt')
        if self.privacy_config.dp_enabled:
            from opacus.validators import ModuleValidator
            self.model = ModuleValidator.fix(self.model)
            ModuleValidator.validate(self.model, strict=True)
            self.model_config.model_name += '_DP'

        # Create optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.training_config.learning_rate)

        # Setup privacy if enabled
        if not self.privacy_config.dp_enabled:
            logging.info('Differential Privacy disabled.')
        else:
            logging.info('Differential Privacy enabled.')
            from opacus import PrivacyEngine
            self.privacy_engine = PrivacyEngine(secure_mode=self.privacy_config.secure_mode)
            self.model, self.optimizer, self.train_dl = self.privacy_engine.make_private_with_epsilon(
                module=self.model,
                optimizer=self.optimizer,
                data_loader=self.train_dl,
                epochs=self.training_config.epochs,
                target_epsilon=self.privacy_config.target_epsilon,
                target_delta=self.privacy_config.target_delta,
                max_grad_norm=self.privacy_config.max_grad_norm
            )
        if self.training_config.es_enabled:
            logging.info('Early stopping enabled.')

    def _setup_mlflow(self):
        """Initialize MLflow experiment tracking with nested run support."""
        mlflow.set_tracking_uri(self.mlflow_config.tracking_uri)
        mlflow.set_experiment(self.mlflow_config.experiment_name)

        # Handle nested runs for autotuning
        run = mlflow.start_run(
            run_id=self.mlflow_config.child_run_id,
            parent_run_id=self.mlflow_config.parent_run_id,
            nested=self.mlflow_config.parent_run_id is None
        )


        if not self.mlflow_config.child_run_id:
            self.mlflow_config.child_run_id = run.info.run_id

        logging.info(f'Started MLFlow run: {run.info.run_name} (ID: {self.mlflow_config.child_run_id})')

    def _cleanup_mlflow(self):
        """Ensure proper MLflow run termination."""
        try:
            if mlflow.active_run() and mlflow.active_run().info.run_id == self.mlflow_config.child_run_id:
                mlflow.end_run()
                logging.info(f"Ended MLflow run: {self.mlflow_config.child_run_id}")
        except Exception as e:
            logging.warning(f"Error during MLflow cleanup: {e}")

    def _cleanup_resources(self):
        """Release computational resources and clear memory."""
        try:
            for dl in [self.train_dl, self.val_dl, self.test_dl]:
                if hasattr(dl, '_iterator') and dl._iterator is not None:
                    try:
                        dl._iterator._shutdown_workers()
                    except:
                        pass
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            import gc
            gc.collect()

        except Exception as e:
            logging.warning(f"Error during resource cleanup: {e}")

    def __del__(self):
        """Cleanup when object is destroyed"""
        self._cleanup_mlflow()

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self._cleanup_mlflow()


def main():
    """CLI entry point for training pipeline execution."""
    from fire import Fire
    Fire(TrainPipeline)
