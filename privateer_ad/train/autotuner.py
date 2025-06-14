import logging
from pprint import pformat

from typing import Optional, List
from dataclasses import dataclass

import numpy as np
import optuna
import mlflow

from privateer_ad.config import AutotuningConfig, ModelConfig, TrainingConfig, MLFlowConfig, DataConfig
from privateer_ad.train.train import TrainPipeline


@dataclass
class AutotuneParam:
    """
    Configuration specification for individual hyperparameters in optimization studies.

    Defines the search space and constraints for a single hyperparameter, supporting
    various parameter types including categorical choices, continuous ranges, and
    logarithmic distributions commonly used in machine learning optimization.

    Attributes:
        name (str): Parameter identifier matching configuration attribute names
        type (str): Parameter type ('categorical', 'float', 'int', 'loguniform', etc.)
        choices (Optional[List]): Available options for categorical parameters
        low (Optional[float]): Lower bound for numeric parameters
        high (Optional[float]): Upper bound for numeric parameters
        step (Optional[float]): Step size for discrete parameters
        q (Optional[float]): Quantization factor for discrete sampling
        log (Optional[bool]): Whether to use logarithmic scaling for numeric ranges
    """
    name: str
    type: str
    choices: Optional[List] = None
    low: Optional[float] = None
    high: Optional[float] = None
    step: Optional[float] = None
    q: Optional[float] = None
    log: Optional[bool] = False


# Parameter definitions
DATA_PARAMS = [
    AutotuneParam(name='seq_len', type='categorical', choices=[12]),  # [1, 2, 6, 12, 24, 120]
]
MODEL_PARAMS = [
    AutotuneParam(name='num_layers', type='categorical', choices=[1, 2, 3, 4]),
    AutotuneParam(name='hidden_dim', type='categorical', choices=[16, 32, 64, 128]),
    AutotuneParam(name='latent_dim', type='categorical', choices=[8, 16, 32, 64]),
    AutotuneParam(name='num_heads', type='categorical', choices=[1, 2, 4, 8]),
    AutotuneParam(name='dropout', type='float', low=0.0, high=0.5, step=0.05),
]

TRAINING_PARAMS = [
    AutotuneParam(name='learning_rate', type='loguniform', low=1e-5, high=1e-2),
    AutotuneParam(name='batch_size', type='categorical', choices=[256, 512, 1024, 2048, 4096, 8192]),
]

ALL_PARAMS = MODEL_PARAMS + TRAINING_PARAMS


class ModelAutoTuner:
    """
    Optuna-based hyperparameter optimization for anomaly detection models.

    Manages systematic exploration of hyperparameter spaces using Bayesian
    optimization to identify optimal model configurations. Integrates with
    MLflow for comprehensive experiment tracking and supports resumable
    optimization studies through persistent storage.

    The tuner coordinates trial execution, manages resource cleanup, and
    generates visualization artifacts for optimization analysis. Each trial
    runs complete training pipelines with different parameter combinations
    to evaluate performance on the target metric.

    Attributes:
        autotune_config (AutotuningConfig): Optimization settings and constraints
        study (optuna.Study): Optuna study managing the optimization process
        parent_run_id (str): MLflow parent run for experiment hierarchy
    """

    def __init__(self, autotune_config: Optional[AutotuningConfig] = None, parent_run_id: Optional[str] = None):
        """
        Initialize hyperparameter optimization with study configuration.

        Sets up the Optuna study with persistent storage and experiment tracking
        integration. Configures parameter search spaces and optimization direction
        based on the target metric requirements.

        Args:
            autotune_config (AutotuningConfig, optional): Optimization configuration
                                                        including trial limits and
                                                        target metrics
            parent_run_id (str, optional): MLflow parent run for organizing
                                         trial experiments
        """
        self.autotune_config = autotune_config or AutotuningConfig()
        self.model_params = MODEL_PARAMS
        self.data_params = DATA_PARAMS
        self.training_params = TRAINING_PARAMS
        self.parent_run_id = parent_run_id

        storage = self.autotune_config.study_name
        if not storage.startswith(('sqlite://', 'mysql://', 'postgresql://')):
            storage = f"sqlite:///{self.autotune_config.study_name}.db"

        self.study = optuna.create_study(
            study_name=self.autotune_config.study_name,
            direction=self.autotune_config.direction,
            storage=storage,
            load_if_exists=True
        )

    def run(self) -> tuple:
        """
        Execute hyperparameter optimization and generate analysis visualizations.

        Runs the configured number of optimization trials, tracking the best
        performance achieved and generating parameter importance and optimization
        history plots for analysis.

        Returns:
            tuple: Parameter importance plot and optimization history visualization
        """
        logging.info(f'Starting optimization: {self.autotune_config.n_trials} trials')

        self.study.optimize(self.objective, n_trials=self.autotune_config.n_trials,
                            timeout=self.autotune_config.timeout, show_progress_bar=True)

        # Log best result
        best = self.study.best_trial
        logging.info(f'Best {self.autotune_config.target_metric}: {best.value:.4f}')
        logging.info(f'Best params: {best.params}')

        # Return plots
        return (
            optuna.visualization.plot_param_importances(self.study),
            optuna.visualization.plot_optimization_history(self.study, target_name=self.autotune_config.target_metric)
        )

    def objective(self, trial: optuna.Trial) -> float | None:
        """
        Execute single optimization trial with comprehensive resource management.

        Runs complete training pipeline with trial-specific hyperparameters,
        evaluates performance on target metric, and ensures proper resource
        cleanup regardless of trial outcome.

        Args:
            trial (optuna.Trial): Optuna trial object for parameter sampling

        Returns:
            float: Target metric value for optimization, or infinity for failed trials
        """
        training_updates = {
            param.name: self._suggest_value(trial, param)
            for param in self.training_params
        }
        model_updates = {
            param.name: self._suggest_value(trial, param)
            for param in self.model_params
        }
        data_updates = {
            param.name: self._suggest_value(trial, param)
            for param in self.data_params
        }

        logging.info(f'Trial {trial.number}')
        logging.info(pformat(training_updates))
        logging.info(pformat(model_updates))
        logging.info(pformat(data_updates))

        pipeline = None
        try:
            # Create MLflow config for this trial
            trial_mlflow_config = MLFlowConfig(parent_run_id=self.parent_run_id)

            pipeline = TrainPipeline(
                mlflow_config=trial_mlflow_config,
                training_config=TrainingConfig(**training_updates),
                data_config=DataConfig(**data_updates),
                model_config=ModelConfig(**model_updates)
            )

            metrics, figs = pipeline.train_eval()
            target_value = metrics[self.autotune_config.target_metric]
            return target_value

        except Exception as e:
            logging.error(f'Trial {trial.number} failed: {e}')
            return -np.inf if self.autotune_config.direction == 'maximize' else np.inf

        finally:
            if pipeline:
                try:
                    pipeline._cleanup_resources()
                    pipeline._cleanup_mlflow()
                    del pipeline
                except Exception as cleanup_error:
                    logging.warning(f"Error during pipeline cleanup: {cleanup_error}")

            import gc
            gc.collect()

            try:
                if mlflow.active_run():
                    active_run_id = mlflow.active_run().info.run_id
                    if active_run_id != self.parent_run_id:
                        mlflow.end_run()
            except:
                pass

    def _suggest_value(self, trial: optuna.Trial, param: AutotuneParam):
        """
        Generate parameter values based on configured search space specifications.

        Translates AutotuneParam configurations into appropriate Optuna sampling
        calls, handling various parameter types and their specific constraints.

        Args:
            trial (optuna.Trial): Optuna trial for parameter sampling
            param (AutotuneParam): Parameter specification with type and constraints

        Returns:
            Sampled parameter value appropriate for the specified type and constraints

        Raises:
            ValueError: For incomplete parameter specifications
            NotImplementedError: For unsupported parameter types
        """
        if param.type == 'categorical':
            if not param.choices:
                raise ValueError(f"Parameter '{param.name}' of type 'categorical' requires 'choices'")
            return trial.suggest_categorical(name=param.name, choices=param.choices)

        elif param.type == 'float':
            if param.low is None or param.high is None:
                raise ValueError(f"Parameter '{param.name}' of type 'float' requires 'low' and 'high'")
            return trial.suggest_float(
                name=param.name,
                low=param.low,
                high=param.high,
                step=param.step,
                log=param.log
            )

        elif param.type == 'int':
            if param.low is None or param.high is None:
                raise ValueError(f"Parameter '{param.name}' of type 'int' requires 'low' and 'high'")
            return trial.suggest_int(
                name=param.name,
                low=int(param.low),
                high=int(param.high),
                step=int(param.step) if param.step else None,
                log=param.log
            )

        elif param.type == 'loguniform':
            if param.low is None or param.high is None:
                raise ValueError(f"Parameter '{param.name}' of type 'loguniform' requires 'low' and 'high'")
            return trial.suggest_float(
                name=param.name,
                low=param.low,
                high=param.high,
                log=True
            )

        elif param.type == 'uniform':
            if param.low is None or param.high is None:
                raise ValueError(f"Parameter '{param.name}' of type 'uniform' requires 'low' and 'high'")
            return trial.suggest_float(
                name=param.name,
                low=param.low,
                high=param.high
            )

        elif param.type == 'discrete_uniform':
            if param.low is None or param.high is None or param.step is None:
                raise ValueError(
                    f"Parameter '{param.name}' of type 'discrete_uniform' requires 'low', 'high', and 'step'")
            return trial.suggest_float(
                name=param.name,
                low=param.low,
                high=param.high,
                step=param.step
            )

        else:
            raise NotImplementedError(f"Parameter type '{param.type}' not implemented.")