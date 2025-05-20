import os

import mlflow
import torch

from privateer_ad import config, models, logger
from privateer_ad.etl.transform import DataProcessor
from privateer_ad.evaluate.evaluator import ModelEvaluator
from privateer_ad.config import HParams, MLFlowConfig, PathsConf


def evaluate(model_path, data_path='test', threshold=None, **kwargs):
    # Setup configs
    hparams = HParams()
    mlflow_config = MLFlowConfig()
    paths = PathsConf()
    model_config = models.AttentionAutoencoderConfig(seq_len=hparams.seq_len)

    dp = DataProcessor()
    dl = dp.get_dataloader(data_path,
                           batch_size=hparams.batch_size,
                           seq_len=hparams.seq_len)
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')
    model_config_class = getattr(config, f'{hparams.model}Config', None)
    if not model_config_class:
        raise ValueError(f'Config class not found for model: {hparams.model}')

    model = models.AttentionAutoencoder(model_config)
    if paths.experiments_dir.joinpath(model_path).exists():
        model_state_dict = torch.load(paths.experiments_dir.joinpath(model_path).joinpath('model.pt'))
    else:
        model_state_dict = torch.load(model_path)
    model_state_dict = {key.removeprefix('_module.'): value for key, value in model_state_dict.items()}
    model.load_state_dict(model_state_dict)
    model = model.to(device)
    model.eval()

    logger.info(f'Initialize evaluator.')
    evaluator = ModelEvaluator(criterion=hparams.loss, device=device)
    logger.info(f'Evaluate model on {data_path} dataset.')
    metrics, figures = evaluator.evaluate(model, dl, threshold)

    if mlflow_config.track and mlflow.active_run():
        logger.info(f'Log metrics to MLFlow server.')
        mlflow.log_metrics(metrics)
        for path, fig in figures.items():
            mlflow.log_figure(fig, os.path.join('figures', path))
    _fmt = ''.join([f'{key}: {value}\n' for key, value in metrics.items()])
    logger.info(f'Test metrics:\n{_fmt}')

    if mlflow_config.track:
        logger.info(f'Stop MLFlow run.')
        mlflow.end_run()
    return metrics, figures


def main():
    from fire import Fire
    Fire(evaluate)
