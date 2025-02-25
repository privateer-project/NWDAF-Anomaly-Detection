import logging
import os
from datetime import datetime

import mlflow
import torch
from mlflow.models import infer_signature
from torch import nn
from torchinfo import summary

from src import config, models
from src.config import *
from src.utils import set_config
from src.data_utils.load import NWDAFDataloader
from src.handlers import ModelTrainer, ModelEvaluator

def train(**kwargs):
    """Train a model with dynamically parsed configuration using Fire.

    Example usage:
      # LSTM Autoencoder
      python script.py --model=LSTMAutoencoder --epochs=10 --hidden_size1=128 --hidden_size2=64

      # TransformerAD
      python script.py --model=TransformerAD --epochs=10 --n_head=8 --d_model=64
    """
    # Initialize project paths
    paths = PathsConf()

    # setup mlflow
    mlflow_config = set_config(MLFlowConfig, kwargs)
    if mlflow_config.track:
        mlflow.set_tracking_uri(mlflow_config.server_address)
        mlflow.set_experiment(mlflow_config.experiment_name)
        mlflow.start_run(run_name=kwargs.get('run_name', datetime.now().strftime("%Y%m%d-%H%M%S")),
                         nested=mlflow.active_run() is not None)

    # Setup trial dir
    trial_dir = os.path.join(paths.experiments_dir, mlflow.active_run().info.run_name)
    os.makedirs(os.path.join(trial_dir), exist_ok=True)

    # setup device to run training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if mlflow.active_run():
        mlflow.log_params({'device': device})

    # Setup configurations
    hparams = set_config(HParams, kwargs)
    if mlflow.active_run():
        mlflow.log_params(hparams.__dict__)

    # Setup dataloaders
    nwdaf_dl = NWDAFDataloader(hparams=hparams)
    dataloaders = nwdaf_dl.get_dataloaders()
    sample = next(iter(dataloaders['train']))[0]['encoder_cont'][:1].to('cpu')

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

    # Setup loss
    loss_fn = getattr(nn, hparams.loss)(reduction='mean')


    trainer = ModelTrainer(train_dl=dataloaders['train'],
                           val_dl=dataloaders['val'],
                           device=device,
                           hparams=hparams,
                           model=model.to(device),
                           optimizer=optimizer,
                           criterion=loss_fn,
                           apply_dp=kwargs.get('apply_dp', False),
                           early_stopping=kwargs.get('early_stopping', False))

    best_checkpoint = trainer.training()
    model.load_state_dict(best_checkpoint['model_state_dict'])
    torch.save(model.state_dict(), os.path.join(trial_dir, 'model.pt'))

    if mlflow.active_run():
        mlflow.log_metrics({'best_' + key: value for key, value in best_checkpoint['metrics'].items()})

    evaluator = ModelEvaluator(criterion=hparams.loss, device=device)
    test_metrics, figures = evaluator.evaluate(trainer.model, dataloaders['test'])
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
        # log test metrics
        mlflow.log_metrics(test_metrics)
        for name, fig in figures.items():
            mlflow.log_figure(fig, f'{name}.png')

    logging.info(f'Test metrics:\n{''.join([f'{key}: {value}\n' for key, value in test_metrics.items()])}')
    if mlflow.active_run():
        mlflow.end_run()
    return test_metrics

if __name__ == "__main__":
    from fire import Fire
    Fire(train)
