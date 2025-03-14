import os

import mlflow
import torch

import config, models
from config import *
from data_utils.transform import DataProcessor
from evaluate.evaluator import ModelEvaluator
from utils import set_config


def evaluate(model_path, **kwargs):
    hparams = set_config(HParams, kwargs)
    mlflow_config = set_config(MLFlowConfig, kwargs)
    dl = DataProcessor()
    test_dl = dl.get_dataloader('test',
                                use_pca=hparams.use_pca,
                                batch_size=hparams.batch_size,
                                seq_len=hparams.seq_len)
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_config_class = getattr(config, f"{hparams.model}Config", None)
    if not model_config_class:
        raise ValueError(f"Config class not found for model: {hparams.model}")
    # Setup model
    model_config = set_config(getattr(config, f"{hparams.model}Config", None), kwargs)
    model_config.seq_len = hparams.seq_len
    model = getattr(models, hparams.model)(model_config)
    checkpoint = torch.load(model_path)
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
    Fire(evaluate)
