from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import optuna
from privateer_ad import logger
from privateer_ad.config import AutotuningConfig
from privateer_ad.train.train import TrainPipeline


@dataclass
class AutotuneParam:
    """Parameter to tune"""
    name: str
    type: str
    target_config: str = 'model'
    choices: Optional[List] = None
    low: Optional[float] = None
    high: Optional[float] = None
    step: Optional[float] = None
    q: Optional[float] = None
    log: bool = False


# Simple parameter definitions - no class needed
MODEL_PARAMS = [
    AutotuneParam('seq_len', 'categorical', 'model', choices=[1, 2, 6, 12, 24, 120]),
    AutotuneParam('num_layers', 'categorical', 'model', choices=[1, 2, 3, 4]),
    AutotuneParam('hidden_dim', 'categorical', 'model', choices=[16, 32, 64, 128]),
    AutotuneParam('latent_dim', 'categorical', 'model', choices=[8, 16, 32, 64]),
    AutotuneParam('num_heads', 'categorical', 'model', choices=[1, 2, 4, 8]),
    AutotuneParam('dropout', 'float', 'model', low=0.0, high=0.5, step=0.05),
]

TRAINING_PARAMS = [
    AutotuneParam('learning_rate', 'loguniform', 'training', low=1e-5, high=1e-2),
    AutotuneParam('batch_size', 'categorical', 'training', choices=[512, 1024, 2048, 4096]),
]

ALL_PARAMS = MODEL_PARAMS + TRAINING_PARAMS


class ModelAutoTuner:
    """Hyperparameter optimization using Optuna"""

    def __init__(self, config: Optional[AutotuningConfig] = None,
                 params: Optional[List[AutotuneParam]] = None,
                 parent_run_id: Optional[str] = None):

        self.config = config or AutotuningConfig()
        self.params = params or ALL_PARAMS
        self.parent_run_id = parent_run_id

        storage = self.config.study_name
        if not storage.startswith(('sqlite://', 'mysql://', 'postgresql://')):
            storage = f"sqlite:///{self.config.study_name}.db"

        self.study = optuna.create_study(
            study_name=self.config.study_name,
            direction=self.config.direction,
            storage=storage,
            load_if_exists=True
        )

    def objective(self, trial: optuna.Trial) -> float:
        """Run one trial"""
        # Generate config overrides
        overrides = {}
        for param in self.params:
            key = f'{param.target_config}.{param.name}'
            overrides[key] = self._suggest_value(trial, param)

        # Train and evaluate
        try:
            pipeline = TrainPipeline(parent_run_id=self.parent_run_id, config_overrides=overrides)
            results = pipeline.train_eval()
            return results[self.config.target_metric]
        except Exception as e:
            logger.error(f'Trial {trial.number} failed: {e}')
            return float('-inf') if self.config.direction == 'maximize' else float('inf')

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

    def run(self) -> tuple:
        """Run optimization and return plots"""
        logger.info(f'Starting optimization: {self.config.n_trials} trials')

        self.study.optimize(self.objective, n_trials=self.config.n_trials,
                            timeout=self.config.timeout, show_progress_bar=True)

        # Log best result
        best = self.study.best_trial
        logger.info(f'Best {self.config.target_metric}: {best.value:.4f}')
        logger.info(f'Best params: {best.params}')

        # Return plots
        return (
            optuna.visualization.plot_param_importances(self.study),
            optuna.visualization.plot_optimization_history(self.study, target_name=self.config.target_metric)
        )

    @property
    def best_params(self) -> Dict[str, Any]:
        """Get best parameters as config overrides"""
        if not self.study.best_trial:
            return {}

        return {f'{p.target_config}.{p.name}': self.study.best_trial.params[p.name]
                for p in self.params if p.name in self.study.best_trial.params}