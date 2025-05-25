from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass

import optuna
from optuna.distributions import CategoricalChoiceType

from privateer_ad import logger
from privateer_ad.config import AutotuningConfig
from privateer_ad.train.train import TrainPipeline


# =============================================================================
# PARAMETER DEFINITIONS
# =============================================================================

@dataclass
class AutotuneParam:
    """Definition of a parameter to tune"""
    name: str
    type: str  # 'categorical', 'float', 'int', 'uniform', 'loguniform', 'discrete_uniform'
    choices: Optional[List[CategoricalChoiceType]] = None
    low: Optional[Union[int, float]] = None
    high: Optional[Union[int, float]] = None
    step: Optional[Union[int, float]] = None
    q: Optional[float] = None
    log: bool = False
    target_config: str = 'model'  # Which config section this parameter belongs to


class DefaultAutotuneParams:
    """Default parameter search space definitions"""

    @staticmethod
    def get_model_params() -> List[AutotuneParam]:
        """Get model architecture parameters to tune"""
        return [
            AutotuneParam(
                name='seq_len',
                type='categorical',
                choices=[1, 2, 6, 12, 24, 120],
                target_config='model'
            ),
            AutotuneParam(
                name='num_layers',
                type='categorical',
                choices=[1, 2, 3, 4],
                target_config='model'
            ),
            AutotuneParam(
                name='hidden_dim',
                type='categorical',
                choices=[16, 32, 64, 128],
                target_config='model'
            ),
            AutotuneParam(
                name='latent_dim',
                type='categorical',
                choices=[8, 16, 32, 64],
                target_config='model'
            ),
            AutotuneParam(
                name='num_heads',
                type='categorical',
                choices=[1, 2, 4, 8],
                target_config='model'
            ),
            AutotuneParam(
                name='dropout',
                type='float',
                low=0.0,
                high=0.5,
                step=0.05,
                target_config='model'
            ),
        ]

    @staticmethod
    def get_training_params() -> List[AutotuneParam]:
        """Get training parameters to tune"""
        return [
            AutotuneParam(
                name='learning_rate',
                type='loguniform',
                low=1e-5,
                high=1e-2,
                target_config='training'
            ),
            AutotuneParam(
                name='batch_size',
                type='categorical',
                choices=[512, 1024, 2048, 4096],
                target_config='training'
            ),
        ]

    @staticmethod
    def get_all_params() -> List[AutotuneParam]:
        """Get all default parameters"""
        return (
                DefaultAutotuneParams.get_model_params() +
                DefaultAutotuneParams.get_training_params()
        )


# =============================================================================
# AUTOTUNER IMPLEMENTATION
# =============================================================================

class ModelAutoTuner:
    """
    Automatic hyperparameter tuning.

    This class handles hyperparameter optimization using Optuna.
    """

    def __init__(
            self,
            autotuning_config: Optional[AutotuningConfig] = None,
            tune_params: Optional[List[AutotuneParam]] = None,
            config_overrides: Optional[Dict[str, Any]] = None,
            parent_run_id: Optional[str] = None
    ):
        """
        Initialize the autotuner.

        Args:
            autotuning_config: Optional autotuning configuration override
            tune_params: Optional custom parameters to tune
            config_overrides: Optional configuration overrides
        """
        # Get configuration
        self.autotuning_config = autotuning_config or AutotuningConfig()
        self.config_overrides = config_overrides or {}

        # Set up parameters to tune
        self.tune_params = tune_params or DefaultAutotuneParams.get_all_params()

        # Setup storage
        storage_url = self._setup_storage()
        self.autotune_run_id = parent_run_id
        logger.info(
            f'Study: {self.autotuning_config.study_name} | '
            f'target: {self.autotuning_config.target_metric} '
            f'direction: {self.autotuning_config.direction}'
        )

        # Create or load study
        self.study = optuna.create_study(
            study_name=self.autotuning_config.study_name,
            direction=self.autotuning_config.direction,
            storage=storage_url,
            load_if_exists=True
        )

    def _setup_storage(self) -> str:
        """Setup Optuna storage URL"""
        storage = self.autotuning_config.storage_url

        # Handle simple names by converting to SQLite URL
        if not storage.startswith(('sqlite://', 'mysql://', 'postgresql://')):
            if not storage.endswith('.db'):
                storage = f'{storage}.db'
            # Prefix with study name
            storage = f'{self.autotuning_config.study_name}-{storage}'
            storage = f'sqlite:///{storage}'

        return storage

    def objective(self, trial: optuna.Trial) -> float:
        """
        Objective function for Optuna optimization.

        Args:
            trial: Optuna trial object

        Returns:
            Objective value to optimize
        """
        logger.info(f'Starting trial {trial.number}')

        # Generate trial-specific config overrides
        trial_overrides = self._generate_trial_config(trial)

        # Merge with any existing config overrides
        combined_overrides = {**self.config_overrides, **trial_overrides}

        # Create training pipeline with trial-specific configuration
        train_pipeline = TrainPipeline(
            parent_run_id=self.autotune_run_id,
            config_overrides=combined_overrides
        )

        # Run training and evaluation
        try:
            results = train_pipeline.train_eval()
            objective_value = results[self.autotuning_config.target_metric]

            logger.info(
                f'Trial {trial.number} completed with '
                f'{self.autotuning_config.target_metric}: {objective_value:.5f}'
            )

            return objective_value

        except Exception as e:
            logger.error(f'Trial {trial.number} failed: {str(e)}')
            # Return worst possible value for failed trials
            if self.autotuning_config.direction == 'maximize':
                return float('-inf')
            else:
                return float('inf')

    def _generate_trial_config(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Generate configuration overrides for a specific trial.

        Args:
            trial: Optuna trial object

        Returns:
            Dictionary of configuration overrides
        """
        trial_config = {}

        for param in self.tune_params:
            suggest_func = self._get_suggest_function(trial, param)
            value = suggest_func()

            # Format as config path (e.g., 'model.seq_len')
            config_path = f'{param.target_config}.{param.name}'
            trial_config[config_path] = value

        return trial_config

    def _get_suggest_function(self, trial: optuna.Trial, param: AutotuneParam):
        """
        Get the appropriate Optuna suggest function for a parameter.

        Args:
            trial: Optuna trial object
            param: Parameter definition

        Returns:
            Callable that suggests parameter value
        """
        if param.type == 'categorical':
            return lambda: trial.suggest_categorical(param.name, param.choices)

        elif param.type == 'float':
            return lambda: trial.suggest_float(
                param.name,
                param.low,
                param.high,
                step=param.step,
                log=param.log
            )

        elif param.type == 'int':
            return lambda: trial.suggest_int(
                param.name,
                param.low,
                param.high,
                step=param.step,
                log=param.log
            )

        elif param.type == 'uniform':
            return lambda: trial.suggest_uniform(param.name, param.low, param.high)

        elif param.type == 'loguniform':
            return lambda: trial.suggest_loguniform(param.name, param.low, param.high)

        elif param.type == 'discrete_uniform':
            return lambda: trial.suggest_discrete_uniform(
                param.name,
                param.low,
                param.high,
                param.q
            )
        else:
            raise NotImplementedError(f'Parameter type "{param.type}" not implemented')

    def autotune(self) -> tuple:
        """
        Run hyperparameter tuning.

        Returns:
            Tuple of (param_importance_fig, optimization_history_fig)
        """
        logger.info('Starting hyperparameter optimization...')

        # Run optimization
        self.study.optimize(
            self.objective,
            n_trials=self.autotuning_config.n_trials,
            timeout=self.autotuning_config.timeout,
            show_progress_bar=True
        )

        # Log results
        self._log_results()

        # Generate visualization plots
        param_importance_fig = optuna.visualization.plot_param_importances(self.study)
        optimization_hist_fig = optuna.visualization.plot_optimization_history(
            self.study,
            target_name=self.autotuning_config.target_metric
        )

        return param_importance_fig, optimization_hist_fig

    def _log_results(self):
        """Log optimization results"""
        logger.info('Optimization completed!')
        logger.info('Best trial:')

        trial = self.study.best_trial
        logger.info(f'  Value: {trial.value:.5f}')

        logger.info('  Params:')
        for key, value in trial.params.items():
            logger.info(f'    {key}: {value}')

    def get_best_config_overrides(self) -> Dict[str, Any]:
        """
        Get the best configuration overrides found during optimization.

        Returns:
            Dictionary of configuration overrides for best trial
        """
        if self.study.best_trial is None:
            raise ValueError("No trials completed yet")

        # Convert best trial params back to config overrides format
        best_overrides = {}
        for param in self.tune_params:
            if param.name in self.study.best_trial.params:
                config_path = f'{param.target_config}.{param.name}'
                best_overrides[config_path] = self.study.best_trial.params[param.name]

        return best_overrides

    def get_optimization_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the optimization process.

        Returns:
            Dictionary containing optimization summary
        """
        return {
            'study_name': self.autotuning_config.study_name,
            'n_trials': len(self.study.trials),
            'best_value': self.study.best_value,
            'best_params': self.study.best_params,
            'target_metric': self.autotuning_config.target_metric,
            'direction': self.autotuning_config.direction,
            'best_config_overrides': self.get_best_config_overrides()
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_autotuner_from_config(
        config_overrides: Optional[Dict[str, Any]] = None,
        custom_params: Optional[List[AutotuneParam]] = None
) -> ModelAutoTuner:
    """
    Create autotuner with configuration from global config.

    Args:
        config_overrides: Optional configuration overrides
        custom_params: Optional custom parameters to tune

    Returns:
        Configured ModelAutoTuner instance
    """
    # Get autotuning config from global config
    # Note: This would need to be added to the main config if not present
    autotuning_config = AutotuningConfig()

    return ModelAutoTuner(
        autotuning_config=autotuning_config,
        tune_params=custom_params,
        config_overrides=config_overrides
    )


def run_autotuning_session(
        study_name: Optional[str] = None,
        n_trials: int = 10,
        target_metric: str = 'f1-score',
        direction: str = 'maximize',
        config_overrides: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Run a complete autotuning session with specified parameters.

    Args:
        study_name: Name for the study
        n_trials: Number of trials to run
        target_metric: Metric to optimize
        direction: Optimization direction
        config_overrides: Configuration overrides

    Returns:
        Optimization summary
    """
    # Create custom autotuning config
    autotuning_config = AutotuningConfig(
        study_name=study_name or 'auto_session',
        n_trials=n_trials,
        target_metric=target_metric,
        direction=direction
    )

    # Create autotuner
    autotuner = ModelAutoTuner(
        autotuning_config=autotuning_config,
        config_overrides=config_overrides
    )

    # Run optimization
    param_fig, hist_fig = autotuner.autotune()

    # Save plots
    param_fig.write_html('params_importance.html')
    hist_fig.write_html('optimization_history.html')

    return autotuner.get_optimization_summary()