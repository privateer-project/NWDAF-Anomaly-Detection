import os

import mlflow
import torch

from privateer_ad import config, models
from privateer_ad.data_utils.transform import DataProcessor
from privateer_ad.evaluate.evaluator import ModelEvaluator
from privateer_ad.utils import set_config


def evaluate(model_path, data_path='test', threshold=None, **kwargs):
    hparams = set_config(config.HParams, kwargs)
    mlflow_config = set_config(config.MLFlowConfig, kwargs)
    paths = set_config(config.PathsConf, kwargs)
    dp = DataProcessor()
    dl = dp.get_dataloader(data_path,
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
    if paths.experiments_dir.joinpath(model_path).exists():
        model_state_dict = torch.load(paths.experiments_dir.joinpath(model_path).joinpath('model.pt'))
    else:
        model_state_dict = torch.load(model_path)
    model_state_dict = {key.removeprefix('_module.'): value for key, value in model_state_dict.items()}
    model.load_state_dict(model_state_dict)
    model = model.to(device)
    model.eval()

    evaluator = ModelEvaluator(criterion=hparams.loss, device=device)
    metrics, figures = evaluator.evaluate(model, dl, threshold)

    if mlflow_config.track and mlflow.active_run():
        mlflow.log_metrics(metrics)
        for path, fig in figures.items():
            mlflow.log_figure(fig, os.path.join('figures', path))
    _fmt = ''.join([f'{key}: {value}\n' for key, value in metrics.items()])
    print(f'Test metrics:\n{_fmt}')

    if mlflow_config.track:
        mlflow.end_run()
    return metrics, figures


def main():
    from fire import Fire
    Fire(evaluate)
