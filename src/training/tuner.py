from datetime import datetime

import optuna
import mlflow

from src import config
from src.config import *
from src.training.trainer import ModelTrainer


class ModelTuner:
    """Automatic hyperparameter tuning for LSTM model."""

    def __init__(
            self,
            n_trials: int = 20,
            timeout: int = 3600 * 8,  # 8 hours
    ):
        self.n_trials = n_trials
        self.timeout = timeout

        # Initialize configs
        self.paths = Paths()
        self.metadata = MetaData()
        self.diff_privacy_config = DifferentialPrivacyConfig()
        self.mlflow_config = MLFlowConfig()
        self.study_name = self.mlflow_config.experiment_name
        self.partition_config = PartitionConfig()

        if self.mlflow_config.track:
            mlflow.set_tracking_uri(self.mlflow_config.server_address)
            mlflow.set_experiment(self.mlflow_config.experiment_name)
            run_name = 'autotuning-tfad'
            run_id = None
            runs = mlflow.search_runs(experiment_names=[self.mlflow_config.experiment_name],
                                      filter_string=f"tags.mlflow.runName = '{run_name}'",
                                      output_format="list")
            if runs:
                run_id = runs[0].info.run_id
            mlflow.start_run(run_id=run_id, run_name=run_name)

        self.nested_run_name = datetime.now().strftime("%Y%m%d-%H%M%S")
        # Create study directory
        self.study_dir = self.paths.studies.joinpath(self.study_name)
        self.run_dir = self.study_dir.joinpath(self.nested_run_name)
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def objective(self, trial: optuna.Trial) -> float:
        """Objective function for optimization."""
        # Hyperparameters to tune
        hparams = HParams(
            epochs=200,  # Fixed number of epochs for tuning
            learning_rate=trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
            batch_size=trial.suggest_categorical("batch_size", [256, 512, 1024, 2048]),
            window_size=trial.suggest_int("window_size", 12, 48)
        )
        model_config = getattr(config, hparams.model + 'Config')()

        if hparams.model == 'LSTMAutoencoder':
            model_config.input_size = 4  # Will be updated based on data
            model_config.hidden_size1 = trial.suggest_categorical("hidden_size1", [32, 64, 128])
            model_config.hidden_size2 = trial.suggest_categorical("hidden_size2", [32, 64, 128])
            model_config.num_layers = trial.suggest_int("num_layers", 1, 3)
            model_config.dropout = trial.suggest_float("dropout_rate", 0.1, 0.5)

        if hparams.model == 'TransformerAD':
            model_config.d_input = 4  # Will be updated based on data
            model_config.seq_len = hparams.window_size
            model_config.d_model = trial.suggest_categorical("embedding_dim", [64, 128])
            if model_config.d_model == 4:
                model_config.n_head = 4
            else:
                model_config.n_head = trial.suggest_categorical("num_heads", [4, 8])
            model_config.n_layers = trial.suggest_int("num_layers", 1, 4)
            model_config.dropout = trial.suggest_float("dropout_rate", 0.1, 0.3)

        optimizer_config = OptimizerConfig()
        nested_run_name = datetime.now().strftime("%Y%m%d-%H%M%S")
        mlflow.start_run(run_name=nested_run_name, nested=True)
        # Create and train model
        trainer = ModelTrainer(
            paths=self.paths,
            metadata=self.metadata,
            hparams=hparams,
            model_config=model_config,
            optimizer_config=optimizer_config,
            diff_privacy_config=self.diff_privacy_config,
            mlflow_config=self.mlflow_config,
            partition_config=self.partition_config,
        )
        trainer.training()
        trainer.evaluate_model(self.run_dir)
        mlflow.end_run()

        # Return validation loss
        return trainer.best_checkpoint['val_loss']

    def tune(self):
        """Run hyperparameter tuning."""
        study = optuna.create_study(
            study_name=self.study_name,
            # sampler=optuna.samplers.GPSampler(),
            direction="minimize",
            storage=f"sqlite:///{self.study_dir}/{self.study_name}.db",
            load_if_exists=True
        )

        study.optimize(
            self.objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=True
        )

        # Print results
        print("Best trial:")
        trial = study.best_trial
        print(f"  Value: {trial.value:.5f}")
        print("  Params: ")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")
        # Save study visualization
        try:
            import plotly
            fig = optuna.visualization.plot_param_importances(study)
            fig.write_html(str(self.study_dir.joinpath("param_importances.html")))
            fig.write_image(str(self.study_dir.joinpath("param_importances.png")))

            fig = optuna.visualization.plot_optimization_history(study, target_name='val_loss')
            fig.write_html(str(self.study_dir.joinpath("optimization_history.html")))
            fig.write_image(str(self.study_dir.joinpath("optimization_history.png")))
        except (ImportError, AttributeError):
            print("Plotly not available for visualization")

        mlflow.log_artifact(str(self.study_dir.joinpath("param_importances.html")))
        mlflow.log_artifact(str(self.study_dir.joinpath("param_importances.png")))
        mlflow.log_artifact(str(self.study_dir.joinpath("optimization_history.html")))
        mlflow.log_artifact(str(self.study_dir.joinpath("optimization_history.png")))
        mlflow.end_run()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Autotune LSTM model')
    parser.add_argument('--n_trials', type=int, default=100,
                        help='Number of trials for optimization')
    parser.add_argument('--timeout', type=int, default=3600 * 8,
                        help='Timeout in seconds')

    args = parser.parse_args()

    tuner = ModelTuner(
        n_trials=args.n_trials,
        timeout=args.timeout,
    )

    tuner.tune()