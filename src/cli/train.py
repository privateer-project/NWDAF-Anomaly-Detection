import os
from datetime import datetime

import mlflow
import torch

from src import config
from src.config import *
from src.data_handling.load import DataLoaderFactory
from src.cli.utils import filter_config_kwargs
from src.training import ModelTrainer, ModelEvaluator




def main(**kwargs):
    """Train a model with dynamically parsed configuration using Fire.

    Example usage:
      # LSTM Autoencoder
      python script.py --model=LSTMAutoencoder --epochs=10 --hidden_size1=128 --hidden_size2=64

      # TransformerAD
      python script.py --model=TransformerAD --epochs=10 --n_head=8 --d_model=64
    """
    paths = ProjectPaths()
    metadata = MetaData()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    configs = {
        'hparams': HParams(**filter_config_kwargs(HParams, kwargs)),
        'optimizer_config': OptimizerConfig(**filter_config_kwargs(OptimizerConfig, kwargs)),
        'mlflow_config': MLFlowConfig(**filter_config_kwargs(MLFlowConfig, kwargs)),
        'diff_privacy_config': DifferentialPrivacyConfig(**filter_config_kwargs(DifferentialPrivacyConfig, kwargs)),
        'partition_config': PartitionConfig(**filter_config_kwargs(PartitionConfig, kwargs))
    }

    config_class_name = f"{configs['hparams'].model}Config"
    model_config_class = getattr(config, config_class_name, None)
    if not model_config_class:
        raise ValueError(f"Config class not found for model: {configs['hparams'].model}")
    configs['model_config'] = model_config_class(**filter_config_kwargs(model_config_class, kwargs))

    if configs['hparams'].model == "TransformerAD":
        configs['model_config'].seq_len = configs['hparams'].window_size

    if configs['mlflow_config'].track:
        mlflow.set_tracking_uri(configs['mlflow_config'].server_address)
        mlflow.set_experiment(configs['mlflow_config'].experiment_name)
        mlflow.start_run(run_name=datetime.now().strftime("%Y%m%d-%H%M%S"),
                         nested=mlflow.active_run() is not None)

    # Log parameters to MLFlow.
    if configs['mlflow_config'].track:
        mlflow.log_param('device', device)
        mlflow.log_params(configs['hparams'].__dict__)
        mlflow.log_params(configs['model_config'].__dict__)
        mlflow.log_params(configs['optimizer_config'].__dict__)
        if configs['diff_privacy_config'].enable:
            mlflow.log_params(configs['diff_privacy_config'].__dict__)

    loader_factory = DataLoaderFactory(metadata, paths, **{k: v for k, v in configs.items() if k in ['hparams']})
    dataloader_params = {'num_workers': os.cpu_count(),
                         'pin_memory': True,
                         'prefetch_factor': configs['hparams'].batch_size * 100,
                         'persistent_workers': True
                         }
    dataloaders = loader_factory.get_dataloaders(**dataloader_params)
    trainer = ModelTrainer(train_dl=dataloaders['train'], val_dl=dataloaders['val'], device=device, **configs)
    trainer.training()

    evaluator = ModelEvaluator(criterion=configs['hparams'].loss, device=device)
    metrics, figures = evaluator.evaluate(trainer.model, dataloaders['test'])
    if configs['mlflow_config'].track:
        mlflow.pytorch.log_model(pytorch_model=trainer.model.to('cpu'),
                                 artifact_path='model',
                                 input_example=trainer.sample.to('cpu').numpy())

        mlflow.log_metrics(metrics)
        for path, fig in figures.items():
            mlflow.log_figure(fig, os.path.join('figures', path))

    print("Test metrics:", metrics)

    if configs['mlflow_config'].track:
        mlflow.end_run()
    return metrics


if __name__ == "__main__":
    from fire import Fire
    Fire(main)
