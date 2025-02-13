import os
from datetime import datetime

import mlflow

from src.training import ModelTuner
from src.config import MLFlowConfig, AutotuneConfig
from src.cli.utils import filter_config_kwargs

def main(**kwargs):
    run_name = 'autotune-' + f'{datetime.now().strftime("%Y%m%d-%H%M%S")}'
    mlflow_config = MLFlowConfig(**filter_config_kwargs(MLFlowConfig, kwargs))
    if mlflow_config.track:
        mlflow.set_tracking_uri(mlflow_config.server_address)
        mlflow.set_experiment(mlflow_config.experiment_name)
        mlflow.start_run(run_name=run_name)

    autotune_config = AutotuneConfig(**filter_config_kwargs(AutotuneConfig, kwargs)).__dict__
    tuner = ModelTuner(kwargs, **autotune_config)
    param_importance_fig, optimization_hist_fig = tuner.autotune()

    mlflow.log_figure(param_importance_fig, 'param_importances.png')
    mlflow.log_figure(optimization_hist_fig, 'optimization_history.png')

    param_importance_fig.write_html('param_importances.html')
    optimization_hist_fig.write_html('optimization_history.html')
    mlflow.log_artifact('param_importances.html', 'param_importances.html')
    mlflow.log_artifact('optimization_history.html', 'optimization_history.html')

    os.remove('param_importances.html')
    os.remove('optimization_history.html')
    mlflow.end_run()


if __name__ == '__main__':
    from fire import Fire
    Fire(main)
