import logging

from typing import Optional, List
from dataclasses import dataclass

import numpy as np
import optuna

from privateer_ad.config import AutotuningConfig, ModelConfig, TrainingConfig, MLFlowConfig
from privateer_ad.train.train import TrainPipeline


@dataclass
class AutotuneParam:
    """Parameter to tune"""
    name: str
    type: str
    choices: Optional[List] = None
    low: Optional[float] = None
    high: Optional[float] = None
    step: Optional[float] = None
    q: Optional[float] = None
    log: Optional[bool] = False


# Simple parameter definitions - no class needed
MODEL_PARAMS = [
    # AutotuneParam(name='seq_len', type='categorical' , choices=[1, 2, 6, 12, 24, 120]),
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
    """Hyperparameter optimization using Optuna"""

    def __init__(self, autotune_config: Optional[AutotuningConfig] = None, parent_run_id: Optional[str] = None):
        self.autotune_config = autotune_config or AutotuningConfig()
        self.model_params = MODEL_PARAMS
        self.training_params = TRAINING_PARAMS
        self.mlflow_config = MLFlowConfig(parent_run_id=parent_run_id)
        self.training_config = TrainingConfig()
        self.model_config = ModelConfig()

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
        """Run optimization and return plots"""
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

    def objective(self, trial: optuna.Trial) -> float:
        """Run one trial"""
        training_updates = {
            param.name: self._suggest_value(trial, param)
            for param in self.training_params
        }
        model_updates = {
            param.name: self._suggest_value(trial, param)
            for param in self.model_params
        }

        # Create new config instances with updates
        training_config = TrainingConfig(**training_updates)
        model_config = ModelConfig(**model_updates)
        try:
            pipeline = TrainPipeline(mlflow_config=self.mlflow_config,
                                     training_config=training_config,
                                     model_config=model_config)

            best_checkpoint = pipeline.train_model()
            pipeline.evaluate_model()
            return best_checkpoint['metrics'][self.autotune_config.target_metric]
        except Exception as e:
            logging.error(f'Trial {trial.number} failed: {e}')
            return -np.inf if self.autotune_config.direction == 'maximize' else np.inf

    def _suggest_value(self, trial: optuna.Trial, param: AutotuneParam):
        """Get parameter value from trial"""

        # Validate that required attributes exist for each type
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
            # Use suggest_float with log=True instead of deprecated suggest_loguniform
            return trial.suggest_float(
                name=param.name,
                low=param.low,
                high=param.high,
                log=True
            )

        elif param.type == 'uniform':
            if param.low is None or param.high is None:
                raise ValueError(f"Parameter '{param.name}' of type 'uniform' requires 'low' and 'high'")
            # Use suggest_float instead of deprecated suggest_uniform
            return trial.suggest_float(
                name=param.name,
                low=param.low,
                high=param.high
            )

        elif param.type == 'discrete_uniform':
            if param.low is None or param.high is None or param.step is None:
                raise ValueError(
                    f"Parameter '{param.name}' of type 'discrete_uniform' requires 'low', 'high', and 'step'")
            # Use suggest_float with step instead of deprecated suggest_discrete_uniform
            return trial.suggest_float(
                name=param.name,
                low=param.low,
                high=param.high,
                step=param.step
            )

        else:
            raise NotImplementedError(f"Parameter type '{param.type}' not implemented.")
