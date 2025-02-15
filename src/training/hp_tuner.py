import optuna

from src.cli.train import main as train_main
from src.cli.utils import filter_config_kwargs
from src.config import AutotuneConfig


class ModelAutoTuner:
    """Automatic hyperparameter tuning for LSTM model."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.autotune_config = AutotuneConfig(**filter_config_kwargs(AutotuneConfig, self.kwargs))

        storage = self.autotune_config.storage
        if not storage.endswith('.db'):
            storage = '.'.join([storage, 'db'])
        storage = '-'.join([self.autotune_config.study_name, storage])
        storage = 'sqlite:///' + storage
        self.study = optuna.create_study(
            study_name=self.autotune_config.study_name,
            direction=self.autotune_config.direction,
            storage=storage,
            load_if_exists=True
        )

    def set_params(self, trial: optuna.Trial):
        autotune_params = {'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True),
                           'batch_size': trial.suggest_categorical('batch_size', [2048, 4096]),
                           'seq_len': trial.suggest_categorical('seq_len', [16, 32, 64, 128]),
                           'd_model': trial.suggest_categorical('embedding_dim', [64, 128]),
                           'num_layers': trial.suggest_int('num_layers', 1, 4),
                           'run_name': '-'.join(['trial', str(trial.number)])}

        if autotune_params['d_model'] == 4:
            autotune_params['num_heads'] = 4
        else:
            autotune_params['num_heads'] = trial.suggest_categorical('num_heads', [4, 8, 16])
        autotune_params.update(autotune_params)
        return autotune_params

    def objective(self, trial: optuna.Trial) -> float:
        """Objective function for optimization."""
        self.kwargs.update(self.set_params(trial))
        metrics = train_main(**self.kwargs)
        return metrics[self.autotune_config.target]

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
