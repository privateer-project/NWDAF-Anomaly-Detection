import os

import mlflow
import torch

from src import architectures, config
from src.config import *
from src.data_handling.load import NWDAFDataloader
from src.cli.utils import filter_config_kwargs
from src.training import ModelEvaluator

def main(model_path, **kwargs):
    hparams = HParams(**filter_config_kwargs(HParams, kwargs))
    partition_config = PartitionConfig(**filter_config_kwargs(PartitionConfig, kwargs))
    mlflow_config = MLFlowConfig(**filter_config_kwargs(MLFlowConfig, kwargs))
    metadata = MetaData()
    dataloader_params = {'num_workers': os.cpu_count(),
                         'pin_memory': True,
                         'prefetch_factor': hparams.batch_size * 100,
                         'persistent_workers': True
                         }
    dl = NWDAFDataloader(metadata.features,
                         hparams=hparams,
                         paths=ProjectPaths(),
                         partition_config=partition_config)
    test_dl = dl.get_dataloader(path='test',
                                train=False,
                                **dataloader_params)
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config_class_name = f"{hparams.model}Config"
    model_config_class = getattr(config, config_class_name, None)
    if not model_config_class:
        raise ValueError(f"Config class not found for model: {hparams.model}")
    model_config = model_config_class(**filter_config_kwargs(model_config_class, kwargs))

    model_class = getattr(architectures, hparams.model)
    model = model_class(**model_config.__dict__)
    # Load model checkpoint
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
