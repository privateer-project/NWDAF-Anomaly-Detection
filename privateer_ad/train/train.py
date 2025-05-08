import os
from copy import deepcopy
from datetime import datetime

import mlflow
import torch
from torchinfo import summary
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator

from privateer_ad.config import PathsConf, MLFlowConfig, HParams, DifferentialPrivacyConfig, setup_logger
from privateer_ad.data_utils.transform import DataProcessor
from privateer_ad.models import AttentionAutoencoder, AttentionAutoencoderConfig
from privateer_ad.train.trainer import ModelTrainer
from privateer_ad.evaluate.evaluator import ModelEvaluator


class TrainPipeline:
    def __init__(self, run_name=None, nested=False, partition_id=0, num_partitions=1):
        # Setup project configs
        self.nested = nested
        self.partition_id = partition_id
        self.num_partitions = num_partitions
        self.mlflow_config = MLFlowConfig()
        self.paths = PathsConf()
        self.hparams = HParams()
        self.dp_config = DifferentialPrivacyConfig()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if run_name is None:
            run_name = datetime.now().strftime('%Y%m%d-%H%M%S')
        self.logger = setup_logger('local-training')
        self.logger.info(f'Start run with name: {run_name}')

        if self.hparams.apply_dp:
            self.privacy_engine = PrivacyEngine(accountant='rdp', secure_mode=self.dp_config.secure_mode)
            self.logger.info('Differential Privacy enabled.')

        # Setup MLFlow
        self.run_id = None
        if self.mlflow_config.track:
            self.logger.info('Initialize MLFlow')
            mlflow.set_tracking_uri(self.mlflow_config.server_address)
            mlflow.set_experiment(self.mlflow_config.experiment_name)
            runs = mlflow.search_runs(experiment_names=[self.mlflow_config.experiment_name],
                                      filter_string=f'tags.mlflow.runName = \'{run_name}\'',
                                      max_results=1)
            if len(runs) > 0:
                # Use existing run
                self.run_id = runs.iloc[0].run_id
            else:
                with mlflow.start_run(run_name=run_name, nested=self.nested):
                    self.run_id = mlflow.active_run().info.run_id

        self.logger.info(f'Using device: {self.device}')
        self.logger.info(f'Starting run {run_name}')

        self.logger.info('Setup dataloaders.')
        self.data_processor = DataProcessor()

        # Setup trial dir
        self.trial_dir = os.path.join(self.paths.experiments_dir, run_name)
        os.makedirs(os.path.join(self.trial_dir), exist_ok=True)

        self.train_dl = self.data_processor.get_dataloader(
            'train',
            use_pca=self.hparams.use_pca,
            batch_size=self.hparams.batch_size,
            partition_id=self.partition_id,
            num_partitions=self.num_partitions,
            seq_len=self.hparams.seq_len,
            only_benign=True)

        self.val_dl = self.data_processor.get_dataloader('val',
                                                         use_pca=self.hparams.use_pca,
                                                         batch_size=self.hparams.batch_size,
                                                         partition_id=self.partition_id,
                                                         num_partitions=self.num_partitions,
                                                         seq_len=self.hparams.seq_len,
                                                         only_benign=True)

        self.test_dl = self.data_processor.get_dataloader('test',
                                                          use_pca=self.hparams.use_pca,
                                                          batch_size=self.hparams.batch_size,
                                                          partition_id=self.partition_id,
                                                          num_partitions=self.num_partitions,
                                                          seq_len=self.hparams.seq_len,
                                                          only_benign=False)

        self.sample = next(iter(self.train_dl))[0]['encoder_cont'][:1].to('cpu')

        # Setup model
        self.model_config = AttentionAutoencoderConfig(seq_len=self.hparams.seq_len, input_size=self.sample.shape[-1])
        self.model_dir = os.path.join(self.trial_dir, 'model.pt')
        torch.serialization.add_safe_globals([AttentionAutoencoder])
        self.model = AttentionAutoencoder(self.model_config)

        if self.hparams.apply_dp:
            self.model = ModuleValidator.fix(self.model)
            ModuleValidator.validate(self.model, strict=True)
            
        print(f'MODEL: {self.model} trainer initialized')

    def train_model(self, start_epoch=0):
        print(f'Starting training with epochs')
        print(f'MODEL: {self.model}')
        self.model.train()

        model_summary = summary(self.model,
                                input_data=self.sample,
                                col_names=('input_size', 'output_size', 'num_params', 'params_percent'))

        with open(os.path.join(self.trial_dir, 'model_summary.txt'), 'w') as f:
            f.write(str(model_summary))

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.learning_rate)

        if self.hparams.apply_dp:
            self.model, optimizer, train_dl = self.privacy_engine.make_private_with_epsilon(
                module=self.model,
                optimizer=optimizer,
                data_loader=self.train_dl,
                epochs=self.hparams.epochs,
                target_epsilon=self.dp_config.target_epsilon,
                target_delta=self.dp_config.target_delta,
                max_grad_norm=self.dp_config.max_grad_norm
            )

        self.logger.info('Instantiate ModelTrainer...')
        self.logger.info('Start Training...')

        trainer = ModelTrainer(model=self.model,
                               optimizer=optimizer,
                               criterion=self.hparams.loss,
                               device=self.device,
                               hparams=self.hparams,
                               run_id=self.run_id)
        best_checkpoint = trainer.training(train_dl=self.train_dl, val_dl=self.val_dl, start_epoch=start_epoch)

        self.logger.info('Training Finished.')

        if self.run_id:
            mlflow.log_params({'device': self.device}, run_id=self.run_id)
            mlflow.log_text(str(model_summary), 'model_summary.txt', run_id=self.run_id)
            if self.hparams.apply_dp:
                mlflow.log_params(*optimizer.state_dict()['param_groups'], run_id=self.run_id)
                mlflow.log_params(self.dp_config.__dict__, run_id=self.run_id)
                mlflow.log_params({'epsilon': self.privacy_engine.get_epsilon(self.dp_config.target_delta)},
                                  run_id=self.run_id)

        # Set model to best so far
        self.model.load_state_dict(deepcopy(best_checkpoint['model_state_dict']))
        torch.save(self.model.state_dict(), self.model_dir)
        return best_checkpoint

    def evaluate_model(self, step=0):
        evaluator = ModelEvaluator(criterion=self.hparams.loss, device=self.device, run_id=self.run_id)
        self.model.eval()

        metrics, figures = evaluator.evaluate(self.model, self.test_dl, prefix='eval', step=step)
        for name, fig in figures.items():
            fig.savefig(os.path.join(self.trial_dir, f'{name}.png'))

        metrics_logs = '\n'.join([f'{key}: {value}' for key, value in metrics.items()])
        self.logger.info(f'Test metrics:\n{metrics_logs}')
        return metrics

    def train_eval(self, start_epoch=0):
        self.train_model(start_epoch=start_epoch)
        return self.evaluate_model(step=start_epoch)


def main():
    from fire import Fire
    Fire(TrainPipeline)

if __name__ == "__main__":
    main()
