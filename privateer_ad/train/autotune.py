import os
import logging

import mlflow

from privateer_ad.config import MLFlowConfig
from privateer_ad.train import ModelAutoTuner


def autotune():
    """
    Execute hyperparameter optimization with comprehensive experiment tracking.

    Orchestrates the complete autotuning workflow from MLflow initialization
    through parameter optimization to result visualization and artifact logging.
    The function manages experiment tracking infrastructure, runs Optuna-based
    optimization, and preserves all results for subsequent analysis.

    The process generates parameter importance plots and optimization history
    visualizations that provide insights into the hyperparameter search process.
    All artifacts are logged to MLflow for persistent storage and later retrieval,
    while temporary files are cleaned up to maintain system hygiene.

    Note:
        Creates a parent MLflow run that coordinates child runs created during
        individual optimization trials, establishing proper experiment hierarchy
        for detailed analysis of the optimization process.
    """
    logging.info('Initialize auto-tuning.')
    mlflow_config = MLFlowConfig()

    mlflow.set_tracking_uri(mlflow_config.tracking_uri)
    mlflow.set_experiment(mlflow_config.experiment_name)
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
    """
    Command-line interface entry point for hyperparameter optimization.

    Provides Fire-based CLI access to the autotuning functionality, enabling
    execution from command line with automatic argument parsing and help
    generation for operational deployment scenarios.
    """
    from fire import Fire
    Fire(autotune)
