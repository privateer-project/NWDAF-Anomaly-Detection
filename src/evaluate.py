import os

import mlflow
import torch

from src import config, models
from src.config import *
from src.data_utils.load import NWDAFDataloader
from src.utils import set_config
from src.handlers import ModelEvaluator


def main(model_path, **kwargs):
    hparams = set_config(HParams, kwargs)
    partition_config = set_config(PartitionConfig, kwargs)
    mlflow_config = set_config(MLFlowConfig, kwargs)
    metadata = MetaData()
    dataloader_params = {'num_workers': os.cpu_count(),
                         'pin_memory': True,
                         'prefetch_factor': hparams.batch_size * 100,
                         'persistent_workers': True
                         }
    dl = NWDAFDataloader(hparams=hparams)
    test_dl = dl.get_dataloaders()['test']
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config_class_name = f"{hparams.model}Config"
    model_config_class = getattr(config, config_class_name, None)
    if not model_config_class:
        raise ValueError(f"Config class not found for model: {hparams.model}")
    # model_config = set_config(model_config_class, kwargs)
    #
    # model_class = getattr(models, hparams.model)
    # model = model_class(**model_config.__dict__)
    # Load model checkpoint
    model_uri = 'runs:/71045813a07646f48fe505ba19b48809/model'
    # This is the input example logged with the model
    model = mlflow.pyfunc.load_model(model_uri)
    model = model.to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    evaluator = ModelEvaluator(criterion=hparams.loss, device=device)
    metrics, figures = evaluator.evaluate(model, test_dl)

    if mlflow_config.track:
        mlflow.log_metrics(metrics)
        for path, fig in figures.items():
            mlflow.log_figure(fig, os.path.join('figures', path))
    _fmt = ''.join([f'{key}: {value}\n' for key, value in metrics.items()])
    print(f'Test metrics:\n{_fmt}')

    if mlflow_config.track:
        mlflow.end_run()
    return metrics

if __name__ == '__main__':
    from fire import Fire
    Fire(main)
