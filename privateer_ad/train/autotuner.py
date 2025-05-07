from dataclasses import dataclass

import optuna

from privateer_ad.config.hparams_config import AutotuneParams
from privateer_ad.train.train import TrainPipeline
from privateer_ad.config import setup_logger, update_config

logger = setup_logger('autotuner')

@dataclass
class AutotuneConfig:
   study_name: str = 'study_1'
   storage: str = 'optuna.db'
   n_trials: int = 10
   timeout: int = 3600 * 8  # 8 hours
   target: str = 'f1-score'
   direction: str = 'maximize'
   kwargs: dict = None


class ModelAutoTuner:
    """Automatic hyperparameter tuning for LSTM model."""

    def __init__(self, config=None):
        self.config = config

        self.autotune_config = AutotuneConfig() if not self.config else update_config(AutotuneConfig, self.config)
        storage = self.autotune_config.storage
        if not storage.endswith('.db'):
            storage = '.'.join([storage, 'db'])
        storage = '-'.join([self.autotune_config.study_name, storage])
        storage = 'sqlite:///' + storage
        logger.info(f'Study: {self.autotune_config.study_name} | '
                    f'target: {self.autotune_config.target} direction: {self.autotune_config.direction} |')

        self.study = optuna.create_study(
            study_name=self.autotune_config.study_name,
            direction=self.autotune_config.direction,
            storage=storage,
            load_if_exists=True
        )

    def objective(self, trial: optuna.Trial) -> float:
        """Objective function for optimization."""
        logger.info(f'Setting parameters for trial {trial.number}.')
        for param in AutotuneParams.params:
            suggest = trial.__getattribute__(f'suggest_{param.type}')
            if param.type == 'float':
                self.__getattribute__('kwargs')[param.name] = suggest(name=param.name,
                                                                      low=param.low,
                                                                      high=param.high,
                                                                      step=param.step,
                                                                      log=param.log)
            elif param.type == 'uniform':
                self.__getattribute__('kwargs')[param.name] = suggest(name=param.name,
                                                                      low=param.low,
                                                                      high=param.high)
            elif param.type == 'loguniform':
                self.__getattribute__('kwargs')[param.name] = suggest(name=param.name,
                                                                      low=param.low,
                                                                      high=param.high)
            elif param.type == 'discrete_uniform':
                self.__getattribute__('kwargs')[param.name] = suggest(name=param.name,
                                                                      low=param.low,
                                                                      high=param.high,
                                                                      q=param.q)
            elif param.type == 'int':
                self.__getattribute__('kwargs')[param.name] = suggest(name=param.name,
                                                                      low=param.low,
                                                                      high=param.high,
                                                                      step=param.step,
                                                                      log=param.log)
            elif param.type == 'categorical':
                self.__getattribute__('kwargs')[param.name] = suggest(name=param.name,
                                                                      choices=param.choices)
            else:
                raise NotImplementedError(f'Parameter type `{param.type}`not implemented.')

        train_pipeline = TrainPipeline('-'.join(['trial', str(trial.number)]), nested=True)
        report = train_pipeline.train_eval()
        return report[self.autotune_config.target]

    def autotune(self):
        """Run hyperparameter tuning."""

        self.study.optimize(
            self.objective,
            n_trials=self.autotune_config.n_trials,
            timeout=self.autotune_config.timeout,
            show_progress_bar=True
        )
        logger.info('Best trial:')
        trial = self.study.best_trial
        logger.info(f'    Value: {trial.value:.5f}')
        params_logs = [f'    {key}: {value}' for key, value in trial.params.items()]
        _log = '\n'.join(params_logs)
        logger.info(f'   Params: {_log}')
        param_importance_fig = optuna.visualization.plot_param_importances(self.study)
        optimization_hist_fig = optuna.visualization.plot_optimization_history(self.study, target_name=self.autotune_config.target)
        return param_importance_fig, optimization_hist_fig
