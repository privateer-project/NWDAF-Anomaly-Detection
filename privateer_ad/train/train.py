import logging

from typing import Dict, Any, Tuple

import mlflow
import torch
from torchinfo import summary

from privateer_ad.config import ModelConfig, MLFlowConfig, DataConfig, TrainingConfig, PathConfig, \
    PrivacyConfig

from privateer_ad.etl.transform import DataProcessor
from privateer_ad.architectures import TransformerAD
from privateer_ad.train.trainer import ModelTrainer
from privateer_ad.evaluate.evaluator import ModelEvaluator
from privateer_ad.utils import get_signature, log_model


class TrainPipeline:
    """
    Training pipeline.

    This class provides an interface for training models.
    """

    def __init__(
            self,
            paths_config: PathConfig | None = None,
            mlflow_config: MLFlowConfig | None = None,
            training_config: TrainingConfig | None = None,
            data_config: DataConfig | None = None,
            model_config: ModelConfig | None = None,
            privacy_config: PrivacyConfig | None = None,
    ):
        """
        Initialize the training pipeline.
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

        # Setup MLFlow
        mlflow.set_tracking_uri(self.mlflow_config.tracking_uri)
        mlflow.set_experiment(self.mlflow_config.experiment_name)
        if mlflow.active_run():
            mlflow.end_run()
        mlflow.start_run(run_id=self.mlflow_config.child_run_id, parent_run_id=self.mlflow_config.parent_run_id)
        if not self.mlflow_config.child_run_id:
            self.mlflow_config.child_run_id = mlflow.active_run().info.run_id
        logging.info(
            f'Started MLFlow run: {mlflow.active_run().info.run_name} (ID: {self.mlflow_config.child_run_id})')

        # Setup datasets
        logging.info('Setup dataloaders.')

        self.dp = DataProcessor(data_config=self.data_config)

        # Create dataloaders with configuration
        self.train_dl = self.dp.get_dataloader('train', only_benign=True, train=True)
        self.val_dl = self.dp.get_dataloader('val', only_benign=False, train=False)
        self.test_dl = self.dp.get_dataloader('test', only_benign=False, train=False)

        # Setup model and optimizer
        torch.serialization.add_safe_globals([TransformerAD])
        self.model_config.seq_len = self.data_config.seq_len
        # Create model instance
        self.model = TransformerAD(self.model_config)
        self.sample = next(iter(self.train_dl))[0]['encoder_cont'][:1].to('cpu')
        model_summary = str(summary(model=self.model, input_data=self.sample,
                                    col_names=('input_size', 'output_size', 'num_params', 'params_percent')))
        if self.privacy_config.dp_enabled:
            from opacus.validators import ModuleValidator
            ModuleValidator.validate(self.model, strict=True)
            self.model = ModuleValidator.fix(self.model)

        # Create optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.training_config.learning_rate)

        # Setup privacy if enabled
        if not self.privacy_config.dp_enabled:
            logging.info('Differential Privacy disabled.')
        else:
            logging.info('Differential Privacy enabled.')
            from opacus import PrivacyEngine

            self.privacy_engine = PrivacyEngine(accountant='rdp', secure_mode=self.privacy_config.secure_mode)

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
        self.trainer = ModelTrainer(model=self.model,
                                    optimizer=self.optimizer,
                                    device=self.device,
                                    training_config=self.training_config)

        self.evaluator = ModelEvaluator(loss_fn=self.training_config.loss_fn_name, device=self.device)

        # Log configuration if MLFlow is enabled
        if mlflow.active_run():
            mlflow.log_params({'device': str(self.device)})
            mlflow.log_params(self.training_config.model_dump())
            mlflow.log_params(self.data_config.model_dump())
            mlflow.log_params(self.model_config.model_dump())
            mlflow.log_params(self.privacy_config.model_dump())
            mlflow.log_text(model_summary, 'model_summary.txt')

    def _log_model(self):
        """Log trained model to MLFlow with proper signature and champion tagging."""
        self.model.to('cpu')
        # Log the model
        mlflow.pytorch.log_model(pytorch_model=self.model,
                                 artifact_path='model',
                                 registered_model_name='TransformerAD',
                                 signature=get_signature(self.model, self.sample),
                                 pip_requirements=self.paths_config.requirements_file.as_posix()
                                 )

    def train_model(self, start_epoch: int = 0) -> Dict[str, Any]:
        """
        Train the model with current configuration.

        Args:
            start_epoch: Starting epoch for training

        Returns:
            Best checkpoint information
        """
        logging.info('Start Training...')
        best_checkpoint = self.trainer.training(train_dl=self.train_dl, val_dl=self.val_dl, start_epoch=start_epoch)

        # Set model to best checkpoint
        self.model.load_state_dict(best_checkpoint['model_state_dict'])
        logging.info('Training Finished.')

        return best_checkpoint

    def evaluate_model(self, step: int = 0) -> Tuple[Dict[str, float], Dict[str, Any]]:
        """
        Evaluate the trained model.

        Args:
            step: Current training step/epoch

        Returns:
            Evaluation metrics
        """
        self.model.eval()

        metrics, figures = self.evaluator.evaluate(self.model, self.test_dl, prefix='test', step=step)

        metrics_logs = '\n'.join([f'{key}: {value}' for key, value in metrics.items()])
        logging.info(f'Test metrics:\n{metrics_logs}')
        return metrics, figures

    def train_eval(self, start_epoch: int = 0) -> Tuple[Dict[str, float], Dict[str, Any]]:
        """
        Complete training and evaluation pipeline.

        Args:
            start_epoch: Starting epoch for training

        Returns:
            Evaluation metrics
        """
        best_checkpoint = self.train_model(start_epoch=start_epoch)
        metrics, figures = self.evaluate_model(step=start_epoch)
        model_name = 'TransformerAD'
        if self.privacy_config.dp_enabled:
            metrics['epsilon'] = self.privacy_engine.get_epsilon(self.privacy_config.target_delta)
            model_name += '_DP'

        log_model(model=self.model,
                  model_name=model_name,
                  sample=self.sample,
                  direction=self.training_config.direction,
                  target_metric=self.training_config.target_metric,
                  current_metrics=best_checkpoint['metrics'],
                  experiment_id=mlflow.get_experiment_by_name(self.mlflow_config.experiment_name).experiment_id,
                  pip_requirements=self.paths_config.requirements_file.as_posix())
        return metrics, figures

    def __exit__(self, exc_type, exc_val, exc_tb):
        # End parent run
        if mlflow.active_run():
            mlflow.end_run()


def main():
    """Main function with Fire integration for CLI usage."""
    from fire import Fire
    Fire(TrainPipeline)
