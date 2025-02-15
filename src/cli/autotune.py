import os
import mlflow

from src.training import ModelAutoTuner
from src.config import MLFlowConfig, AutotuneConfig
from src.cli.utils import filter_config_kwargs

def main(**kwargs):
    mlflow_config = MLFlowConfig(**filter_config_kwargs(MLFlowConfig, kwargs))
    autotune_config = AutotuneConfig(**filter_config_kwargs(AutotuneConfig, kwargs))

    if mlflow_config.track:
        mlflow.set_tracking_uri(mlflow_config.server_address)
        experiment = mlflow.get_experiment_by_name(mlflow_config.experiment_name)
        if experiment is not None:
            mlflow.set_experiment(experiment_id=experiment.experiment_id)
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"tags.mlflow.runName = '{autotune_config.study_name}'",
            max_results=1)
        if len(runs) > 0:
            mlflow.start_run(run_id=runs.iloc[0].run_id,)
        else:
            mlflow.start_run(run_name=autotune_config.study_name)

    tuner = ModelAutoTuner(**kwargs)
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
