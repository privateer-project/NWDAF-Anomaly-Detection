import os

import mlflow

from privateer_ad import logger
from privateer_ad.train import AutotuneConfig, ModelAutoTuner
from privateer_ad.config import MLFlowConfig, update_config

def autotune(**kwargs):
    logger.info('Initialize auto-tuning.')
    mlflow_config = update_config(MLFlowConfig, kwargs)
    autotune_config = update_config(AutotuneConfig, kwargs)

    if mlflow_config.track:
        mlflow.set_tracking_uri(mlflow_config.server_address)
        experiment = mlflow.get_experiment_by_name(mlflow_config.experiment_name)
        if experiment is not None:
            mlflow.set_experiment(experiment_id=experiment.experiment_id)
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=f'tags.mlflow.runName = "{autotune_config.study_name}"',
            max_results=1)
        if len(runs) > 0:
            mlflow.start_run(run_id=runs.iloc[0].run_id)
        else:
            mlflow.start_run(run_name=autotune_config.study_name)

    tuner = ModelAutoTuner(**kwargs)
    logger.info('Start autotuning.')
    param_importance_fig, optimization_hist_fig = tuner.autotune()

    param_importance_fig.write_html('param_importances.html')
    optimization_hist_fig.write_html('optimization_history.html')

    logger.info('Autotuning finished.')
    if mlflow_config.track:
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
