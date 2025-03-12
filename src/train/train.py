import os
from datetime import datetime

import mlflow
import torch
from mlflow.models import infer_signature
from torchinfo import summary

from src import models, config
from src.config import PathsConf, MLFlowConfig, HParams, OptimizerConfig, logger
from src.data_utils.transform import DataProcessor
from src.utils import set_config
from src.train.trainer import ModelTrainer
from src.evaluate.evaluator import ModelEvaluator


def train(**kwargs):
    # Initialize project paths
    paths = PathsConf()

    # Setup mlflow
    mlflow_config = set_config(MLFlowConfig, kwargs)
    if mlflow_config.track:
        mlflow.set_tracking_uri(mlflow_config.server_address)
        mlflow.set_experiment(mlflow_config.experiment_name)
        mlflow.start_run(run_name=kwargs.get('run_name', datetime.now().strftime("%Y%m%d-%H%M%S")),
                         nested=mlflow.active_run() is not None)

    # Setup trial dir
    trial_dir = os.path.join(paths.experiments_dir, mlflow.active_run().info.run_name)
    os.makedirs(os.path.join(trial_dir), exist_ok=True)

    # Setup device to run training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if mlflow.active_run():
        mlflow.log_params({'device': device})

    # Setup configurations
    hparams = set_config(HParams, kwargs)
    if mlflow.active_run():
        mlflow.log_params(hparams.__dict__)

    # Setup dataloaders
    data_processor = DataProcessor()
    train_dl = data_processor.get_dataloader('train',
                                             use_pca=hparams.use_pca,
                                             batch_size=hparams.batch_size,
                                             seq_len=hparams.seq_len,
                                             only_benign=True)
    val_dl = data_processor.get_dataloader('val',
                                             use_pca=hparams.use_pca,
                                             batch_size=hparams.batch_size,
                                             seq_len=hparams.seq_len,
                                             only_benign=True)
    test_dl = data_processor.get_dataloader('test',
                                             use_pca=hparams.use_pca,
                                             batch_size=hparams.batch_size,
                                             seq_len=hparams.seq_len,
                                             only_benign=False)
    sample = next(iter(train_dl))[0]['encoder_cont'][:1].to('cpu')

    # Setup model
    model_config = set_config(getattr(config, f"{hparams.model}Config", None), kwargs)
    model_config.seq_len = hparams.seq_len
    model_config.input_size = sample.shape[-1]
    model = getattr(models, hparams.model)(model_config)
    model_summary = summary(model,
                            input_data=sample,
                            col_names=('input_size', 'output_size', 'num_params', 'params_percent'))
    if mlflow.active_run():
        mlflow.log_text(str(model_summary), 'model_summary.txt')
        mlflow.log_params(model_config.__dict__)

    with open(os.path.join(trial_dir, 'model_summary.txt'), 'w') as f:
        f.write(str(model_summary))

    # Setup optimizer
    optimizer_config = set_config(OptimizerConfig, kwargs)
    optimizer = getattr(torch.optim, optimizer_config.type)(model.parameters(),
                                                            lr=hparams.learning_rate,
                                                            **optimizer_config.params)
    if mlflow.active_run():
        mlflow.log_params(optimizer_config.__dict__)

    trainer = ModelTrainer(train_dl=train_dl,
                           val_dl=val_dl,
                           device=device,
                           hparams=hparams,
                           model=model.to(device),
                           optimizer=optimizer,
                           criterion=hparams.loss)
    best_checkpoint = trainer.training()
    model.load_state_dict(best_checkpoint['model_state_dict'])
    torch.save(model.state_dict(), os.path.join(trial_dir, 'model.pt'))
    torch.jit.script(model)
    if mlflow.active_run():
        mlflow.log_metrics({key: value for key, value in best_checkpoint['metrics'].items()})


    evaluator = ModelEvaluator(criterion=hparams.loss,
                               device=device)
    test_report, figures = evaluator.evaluate(trainer.model, test_dl)
    for name, fig in figures.items():
        fig.savefig(os.path.join(trial_dir, f'{name}.png'))

    if mlflow.active_run():
        # log model with signature to mlflow
        model = trainer.model.to('cpu')
        model_input = sample.to('cpu')
        model_output = model(model_input)
        if isinstance(model_output, dict):
            model_output = {key: val.detach().numpy() for key, val in model_output.items()}
        mlflow.pytorch.log_model(pytorch_model=model,
                                 artifact_path='model',
                                 signature=infer_signature(model_input=model_input.detach().numpy(),
                                                           model_output=model_output),
                                 pip_requirements=os.path.join(paths.root, 'requirements.txt'))
        # log metrics
        mlflow.log_metrics(test_report)
        for name, fig in figures.items():
            mlflow.log_figure(fig, f'{name}.png')

    logger.info(f'Test metrics:\n{''.join([f'{key}: {value}\n' for key, value in test_report.items()])}')
    if mlflow.active_run():
        mlflow.end_run()
    return test_report

if __name__ == "__main__":
    from fire import Fire
    Fire(train)
