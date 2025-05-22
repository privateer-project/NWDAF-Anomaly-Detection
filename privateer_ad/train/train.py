from copy import deepcopy
from datetime import datetime

import mlflow
import torch

from torchinfo import summary
from mlflow.entities import RunStatus

from privateer_ad import logger
from privateer_ad.config import MLFlowConfig, HParams, DPConfig, PathsConf
from privateer_ad.etl.transform import DataProcessor
from privateer_ad.models import TransformerAD, TransformerADConfig
from privateer_ad.train.trainer import ModelTrainer
from privateer_ad.evaluate.evaluator import ModelEvaluator


class TrainPipeline:
    def __init__(self, run_name=None, partition_id=0, partition=False, dp=False, nested=False):
        # Setup project configs
        self.run_name = run_name
        self.partition_id = partition_id
        self.partition = partition
        self.mlflow_config = MLFlowConfig()
        self.paths = PathsConf()
        self.hparams = HParams()
        self.dp_config = DPConfig()
        self.dp_config.enable = dp
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Setup MLFlow
        if self.mlflow_config.track:
            logger.info('Initialize MLFlow')
            mlflow.set_tracking_uri(self.mlflow_config.server_address)
            mlflow.set_experiment(self.mlflow_config.experiment_name)

            self.parent_run_id = None
            self.run_id = None
            if not self.run_name:
                self.run_name = datetime.now().strftime('%Y%m%d-%H%M%S')

            if self.dp_config.enable:
                self.run_name += '-dp'
                self.mlflow_config.server_run_name += '-dp'  #todo check if exists

            if mlflow.active_run():
                logger.info(f"Found active run {mlflow.active_run().info.run_id}, ending it")
                mlflow.end_run()

            if nested and self.mlflow_config.server_run_name:
                parent_runs = mlflow.search_runs(experiment_names=[self.mlflow_config.experiment_name],
                                                 filter_string=f'tags.mlflow.runName = \'{self.mlflow_config.server_run_name}\'',
                                                 max_results=1)
                if len(parent_runs) > 0:
                    self.parent_run_id = parent_runs.iloc[0].run_id
                    if RunStatus.is_terminated(mlflow.get_run(self.parent_run_id).info.status):
                        mlflow.start_run(run_id=self.parent_run_id)

            runs = mlflow.search_runs(experiment_names=[self.mlflow_config.experiment_name],
                                      filter_string=f'tags.mlflow.runName = \'{self.run_name}\'',
                                      max_results=1)
            if len(runs) > 0:
                self.run_id = runs.iloc[0].run_id
                if not RunStatus.is_terminated(mlflow.get_run(self.run_id).info.status):
                    mlflow.MlflowClient().set_terminated(run_id=self.run_id)
            mlflow.start_run(run_id=self.run_id, run_name=self.run_name, parent_run_id=self.parent_run_id)

            self.run_id = mlflow.active_run().info.run_id
            self.run_name= mlflow.active_run().info.run_name
            logger.info(f'Run with name {self.run_name} started.')

        logger.info(f'Using device: {self.device}')

        logger.info('Setup dataloaders.')
        self.data_processor = DataProcessor(partition=partition)

        # Setup trial dir
        if nested and self.mlflow_config.server_run_name:
            self.trial_dir = self.paths.experiments_dir.joinpath(self.mlflow_config.server_run_name, self.run_name)
        else:
            self.trial_dir = self.paths.experiments_dir.joinpath(self.run_name)

        self.trial_dir.mkdir(parents=True, exist_ok=True)

        self.train_dl = self.data_processor.get_dataloader('train',
                                                           batch_size=self.hparams.batch_size,
                                                           seq_len=self.hparams.seq_len,
                                                           partition_id=self.partition_id,
                                                           only_benign=True)

        self.val_dl = self.data_processor.get_dataloader(
            'val',
            batch_size=self.hparams.batch_size,
            partition_id=self.partition_id,
            seq_len=self.hparams.seq_len,
            only_benign=True)

        self.test_dl = self.data_processor.get_dataloader(
            'test',
            batch_size=self.hparams.batch_size,
            partition_id=self.partition_id,
            seq_len=self.hparams.seq_len,
            only_benign=False)

        self.sample = next(iter(self.train_dl))[0]['encoder_cont'][:1].to('cpu')

        # Setup model
        self.model_config = TransformerADConfig(seq_len=self.hparams.seq_len, input_size=self.sample.shape[-1])
        torch.serialization.add_safe_globals([TransformerAD])
        if mlflow.active_run():
            mlflow.log_params(self.model_config.__dict__)
            mlflow.log_params(self.hparams.__dict__)

        self.model = TransformerAD(self.model_config)

        if self.dp_config.enable:
            from opacus import PrivacyEngine
            from opacus.validators import ModuleValidator
            logger.info('Differential Privacy enabled.')
            ModuleValidator.validate(self.model, strict=True)
            self.privacy_engine = PrivacyEngine(accountant='rdp', secure_mode=self.dp_config.secure_mode)

            self.model = ModuleValidator.fix(self.model)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.learning_rate)

            self.model, self.optimizer, self.train_dl = self.privacy_engine.make_private_with_epsilon(
                module=self.model,
                optimizer=self.optimizer,
                data_loader=self.train_dl,
                epochs=100, #self.hparams.epochs,
                target_epsilon=self.dp_config.target_epsilon,
                target_delta=self.dp_config.target_delta,
                max_grad_norm=self.dp_config.max_grad_norm
            )
            mlflow.log_params(self.dp_config.__dict__)
        else:
            logger.info('Differential Privacy disabled.')
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.learning_rate)

        if self.hparams.early_stopping:
            logger.info('Early stopping enabled.')

        model_summary = summary(model=self.model,
                                input_data=self.sample,
                                col_names=('input_size', 'output_size', 'num_params', 'params_percent'))

        if mlflow.active_run():
            mlflow.log_params({'device': self.device})
            mlflow.log_text(str(model_summary), 'model_summary.txt')
        # Save locally
        with self.trial_dir.joinpath('model_summary.txt').open('w') as f:
            f.write(str(model_summary))


    def train_model(self, start_epoch=0):
        logger.info('Start Training...')
        self.model.train()
        try:
            trainer = ModelTrainer(model=self.model,
                                   optimizer=self.optimizer,
                                   criterion=self.hparams.loss,
                                   device=self.device,
                                   hparams=self.hparams)
        except Exception as e:
            raise ValueError(f'Error while initializing trainer: {e}')
        try:
            trainer.training(train_dl=self.train_dl, val_dl=self.val_dl, start_epoch=start_epoch)
        except KeyboardInterrupt: # Break training loop when Ctrl+C pressed (Manual early stopping)
            logger.warning('Training interrupted by user...')
        # Set model to best so far
        self.model.load_state_dict(deepcopy(trainer.best_checkpoint['model_state_dict']))

        if mlflow.active_run():
            # log model with signature to mlflow
            self.model.to('cpu')
            sample = next(iter(self.train_dl))[0]['encoder_cont'][:1].to('cpu')

            _input = sample.to('cpu')
            _output = self.model(_input)
            if isinstance(_output, dict):
                _output = {key: val.detach().numpy() for key, val in _output.items()}
            else:
                _output = _output.detach().numpy()

            mlflow.pytorch.log_model(pytorch_model=self.model,
                                     artifact_path='model',
                                     signature=mlflow.models.infer_signature(model_input=_input.detach().numpy(),
                                                               model_output=_output),
                                     pip_requirements=self.paths.root.joinpath('requirements.txt').as_posix())

        logger.info('Training Finished.')
        if mlflow.active_run() and self.dp_config.enable:
            mlflow.log_metrics({'epsilon': self.privacy_engine.get_epsilon(self.dp_config.target_delta)},
                               step=start_epoch)

        torch.save(self.model.state_dict(), self.trial_dir.joinpath('model.pt'))
        return trainer.best_checkpoint

    def evaluate_model(self, step=0):
        self.model.eval()
        evaluator = ModelEvaluator(criterion=self.hparams.loss, device=self.device)
        metrics, figures = evaluator.evaluate(self.model, self.test_dl, prefix='eval', step=step)

        metrics_logs = '\n'.join([f'{key}: {value}' for key, value in metrics.items()])
        logger.info(f'Test metrics:\n{metrics_logs}')

        # Save locally
        for name, fig in figures.items():
            fig.savefig(self.trial_dir.joinpath(f'{name}.png'))
        return metrics

    def train_eval(self, start_epoch=0):
        self.train_model(start_epoch=start_epoch)
        return self.evaluate_model(step=start_epoch)


def main():
    from fire import Fire
    Fire(TrainPipeline)
