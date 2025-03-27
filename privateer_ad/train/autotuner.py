import optuna

from train.train import train
from utils import set_config
from config import AutotuneConfig, logger


class ModelAutoTuner:
    """Automatic hyperparameter tuning for LSTM model."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.autotune_config = set_config(AutotuneConfig, self.kwargs)

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

    @staticmethod
    def set_params(trial: optuna.Trial):
        logger.info(f'Setting parameters for trial {trial.number}.')
        autotune_params = {'seq_len': trial.suggest_int('seq_len', 1, 60),
                           'd_model': trial.suggest_categorical('d_model', [16, 32, 64, 128]),
                           'nhead': trial.suggest_categorical('nhead', [1, 2, 4, 8]),
                           'num_layers': trial.suggest_int('num_layers', 1, 6),
                           'dropout': trial.suggest_float('dropout', 0.0, 0.35),
                           'run_name': '-'.join(['trial', str(trial.number)])}
        autotune_params.update(autotune_params)
        return autotune_params

    def objective(self, trial: optuna.Trial) -> float:
        """Objective function for optimization."""
        self.kwargs.update(self.set_params(trial))
        report = train(**self.kwargs)
        return report[self.autotune_config.target]

    def autotune(self):
        """Run hyperparameter tuning."""

        self.study.optimize(
            self.objective,
            n_trials=self.autotune_config.n_trials,
            timeout=self.autotune_config.timeout,
            show_progress_bar=True
        )
        print("Best trial:")
        trial = self.study.best_trial
        print(f"  Value: {trial.value:.5f}")
        print("  Params: ")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")

        param_importance_fig = optuna.visualization.plot_param_importances(self.study)
        optimization_hist_fig = optuna.visualization.plot_optimization_history(self.study, target_name=self.autotune_config.target)
        return param_importance_fig, optimization_hist_fig
