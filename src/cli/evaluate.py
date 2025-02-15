import os

import mlflow
import torch

from src import architectures, config
from src.config import *
from src.data_handling.load import DataLoaderFactory
from src.cli.utils import filter_config_kwargs
from src.training import ModelEvaluator

def main(model_path, **kwargs):
    configs = {
        'hparams': HParams(**filter_config_kwargs(HParams, kwargs)),
        'partition_config': PartitionConfig(**filter_config_kwargs(PartitionConfig, kwargs))
    }
    dataloader_params = {'num_workers': os.cpu_count(),
                         'pin_memory': True,
                         'prefetch_factor': configs['hparams'].batch_size * 100,
                         'persistent_workers': True
                         }

    test_dl = DataLoaderFactory.get_single_dataloader(train=False,
                                                      path='test',
                                                      window_size=configs['hparams'].seq_len,
                                                      partition_config=configs['partition_config'],
                                                      **dataloader_params)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config_class_name = f"{configs['hparams'].model}Config"
    model_config_class = getattr(config, config_class_name, None)
    if not model_config_class:
        raise ValueError(f"Config class not found for model: {configs['hparams'].model}")
    configs['model_config'] = model_config_class(**filter_config_kwargs(model_config_class, kwargs))

    model_class = getattr(architectures, configs['hparams'].model)
    model = model_class(**configs['model_config'].__dict__)

    # Load model checkpoint
    checkpoint = torch.load(model_path, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    evaluator = ModelEvaluator(criterion=configs['hparams'].loss, device=device)
    metrics, figures = evaluator.evaluate(model, test_dl)
    if configs['mlflow_config'].track:
        mlflow.log_metrics(metrics)
        for path, fig in figures.items():
            mlflow.log_figure(fig, os.path.join('figures', path))

    print("Test metrics:", metrics)

    if configs['mlflow_config'].track:
        mlflow.end_run()
    return metrics

if __name__ == '__main__':
    from fire import Fire
    Fire(main)
