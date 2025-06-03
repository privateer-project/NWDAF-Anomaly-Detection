import os
import logging

import mlflow

from privateer_ad.config import MLFlowConfig
from privateer_ad.train import ModelAutoTuner


def autotune():
    logging.info('Initialize auto-tuning.')
    mlflow_config = MLFlowConfig()

    mlflow.set_tracking_uri(mlflow_config.tracking_uri)
    experiment = mlflow.get_experiment_by_name(mlflow_config.experiment_name)
    if experiment is not None:
        mlflow.set_experiment(experiment_id=experiment.experiment_id)
    mlflow.start_run()
    tuner = ModelAutoTuner(parent_run_id=mlflow.active_run().info.run_id)
    logging.info('Start autotuning.')
    param_importance_fig, optimization_hist_fig = tuner.run()

    param_importance_fig.write_html('param_importances.html')
    optimization_hist_fig.write_html('optimization_history.html')

    logging.info('Autotuning finished.')
    mlflow.log_figure(param_importance_fig, 'param_importances.png')
    mlflow.log_figure(optimization_hist_fig, 'optimization_history.png')
    mlflow.log_artifact('param_importances.html', 'param_importances.html')
    mlflow.log_artifact('optimization_history.html', 'optimization_history.html')
    os.remove('param_importances.html')
    os.remove('optimization_history.html')
    mlflow.end_run()

def main():
    from fire import Fire
    Fire(autotune)
