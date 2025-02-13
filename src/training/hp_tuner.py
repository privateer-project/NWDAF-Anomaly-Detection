import optuna

from src.cli.train import main as train_main


class ModelTuner:
    """Automatic hyperparameter tuning for LSTM model."""

    def __init__(self,kwargs, *, study_name, storage='sqlite:///optuna.db', target='val_loss', direction='minimize', n_trials: int = 100, timeout: int = 3600 * 8):
        self.target = target
        self.n_trials = n_trials
        self.timeout = timeout
        self.study = optuna.create_study(
            study_name=study_name,
            direction=direction,
            storage=storage,
            load_if_exists=True
        )
        self.kwargs= kwargs

    def set_params(self, trial: optuna.Trial):
        hparams = {'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
                   'batch_size': trial.suggest_categorical('batch_size', [256, 512, 1024, 2048]),
                   'window_size': trial.suggest_int('window_size', 12, 48),
                   'hidden_size1': trial.suggest_categorical('hidden_size1', [32, 64, 128]),
                   'hidden_size2': trial.suggest_categorical('hidden_size2', [32, 64, 128]),
                   'dropout': trial.suggest_float('dropout_rate', 0.1, 0.5),
                   'd_model': trial.suggest_categorical('embedding_dim', [64, 128]),
                   'num_layers': trial.suggest_int('num_layers', 1, 4)}
        if hparams['d_model'] == 4:
            hparams['n_head'] = 4
        else:
            hparams['n_head'] = trial.suggest_categorical('num_heads', [4, 8])
        self.kwargs.update(hparams)
        return self.kwargs

    def objective(self, trial: optuna.Trial) -> float:
        """Objective function for optimization."""
        kwargs = self.set_params(trial)
        metrics = train_main(**kwargs)
        return metrics[self.target]

    def autotune(self):
        """Run hyperparameter tuning."""

        self.study.optimize(
            self.objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=True
        )

        # Print results
        print("Best trial:")
        trial = self.study.best_trial
        print(f"  Value: {trial.value:.5f}")
        print("  Params: ")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")
        # Save study visualization
        param_importance_fig = optuna.visualization.plot_param_importances(self.study)
        optimization_hist_fig = optuna.visualization.plot_optimization_history(self.study, target_name=self.target)

        return param_importance_fig, optimization_hist_fig