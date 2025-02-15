import os
from datetime import datetime

import mlflow
import torch

from src import config
from src.config import *
from src.cli.utils import filter_config_kwargs
from src.data_handling.load import NWDAFDataloader
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

    mlflow_config = MLFlowConfig(**filter_config_kwargs(MLFlowConfig, kwargs))
    diff_privacy_config = DifferentialPrivacyConfig(**filter_config_kwargs(DifferentialPrivacyConfig, kwargs))

    hparams = HParams(**filter_config_kwargs(HParams, kwargs))

    optimizer_config = OptimizerConfig(**filter_config_kwargs(OptimizerConfig, kwargs))
    partition_config = PartitionConfig(**filter_config_kwargs(PartitionConfig, kwargs))

    model_config_name = f"{hparams.model}Config"
    model_config_class = getattr(config, model_config_name, None)

    if not model_config_class:
        raise ValueError(f"Config class not found for model: {hparams.model}")
    model_config = model_config_class(**filter_config_kwargs(model_config_class, kwargs))

    # [pprint(conf) for conf in configs.values()]
    if mlflow_config.track:
        mlflow.set_tracking_uri(mlflow_config.server_address)
        mlflow.set_experiment(mlflow_config.experiment_name)
        mlflow.start_run(run_name=kwargs.get('run_name', datetime.now().strftime("%Y%m%d-%H%M%S")),
                         nested=mlflow.active_run() is not None)
    nwdaf_dl = NWDAFDataloader(features=metadata.features,
                                     hparams=hparams,
                                     paths=paths,
                                     partition_config=partition_config,
                                     )
    dataloader_params = {'num_workers': os.cpu_count(),
                         'pin_memory': True,
                         'prefetch_factor': hparams.batch_size * 100,
                         'persistent_workers': True
                         }

    dataloaders = nwdaf_dl.get_dataloaders(**dataloader_params)
    trainer = ModelTrainer(train_dl=dataloaders['train'],
                           val_dl=dataloaders['val'],
                           device=device,
                           hparams=hparams,
                           model_config=model_config,
                           optimizer_config=optimizer_config,
                           diff_privacy_config=diff_privacy_config,
                           )
    best_metrics = trainer.training()
    if mlflow_config.track:
        mlflow.log_metrics(best_metrics)
    evaluator = ModelEvaluator(criterion=hparams.loss, device=device)
    test_metrics, figures = evaluator.evaluate(trainer.model, dataloaders['test'])

    if mlflow_config.track:
        mlflow.pytorch.log_model(pytorch_model=trainer.model.to('cpu'),
                                 artifact_path='model',
                                 input_example=trainer.sample.to('cpu').numpy())
        mlflow.log_metrics(test_metrics)
        for path, fig in figures.items():
            mlflow.log_figure(fig, os.path.join('figures', path))

    print("Test metrics:", test_metrics)

    if mlflow_config.track:
        mlflow.end_run()
    return test_metrics


if __name__ == "__main__":
    from fire import Fire

    Fire(main)
