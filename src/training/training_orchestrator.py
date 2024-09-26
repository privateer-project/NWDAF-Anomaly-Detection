from typing import Type, Dict, Any
import torch
import torch.nn as nn
from hyperopt import fmin, tpe, Trials
import pandas as pd

from src.utils.config import Config
from src.utils.tuning_utils import search_space
from src.utils.training_utils import *
from src.utils.evaluation_utils import *
from src.utils.logging_utils import *
from src.data.dataloader import *
from src.data.preprocessing import *


class DictToObject:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                setattr(self, key, DictToObject(value))
            else:
                setattr(self, key, value)


class TrainingOrchestrator:
    def __init__(self, model_class: Type[nn.Module], config: dict, experiment_id):
        if not model_class or not hasattr(model_class, "__call__"):
            raise ValueError(
                "model_class must be a callable class reference to a PyTorch model"
            )
        if not config:
            raise ValueError("Config path must be provided")

        self.config = Config.from_json(config)
        self.model_class = model_class

        # -------------------------
        # Data Configuration
        # -------------------------
        self.data_config = self.config.data
        self.benign_train_path = self.data_config.benign_train_path
        self.benign_test_path = self.data_config.benign_test_path
        self.malicious_path = self.data_config.malicious_path
        self.preprocessing_method = self.data_config.preprocessing.preprocessing_method
        self.feature_columns = self.data_config.feature_columns
        self.attack_periods = self.data_config.attacks
        self.malicious_imeisv = self.data_config.malicious_imeisv

        # -------------------------
        # Dataloader Configuration
        # -------------------------
        self.dataloaders_config = self.config.dataloaders
        self.time_window_length = self.dataloaders_config.time_window_length
        self.step_size = self.dataloaders_config.step_size
        self.test_step_size = self.dataloaders_config.test_step_size
        self.train_batch_size = self.dataloaders_config.train_batch_size
        self.test_batch_size = self.dataloaders_config.test_batch_size

        # -------------------------
        # Training Setup
        # -------------------------
        self.train_setup = self.config.train_setup
        self.num_epochs = self.train_setup.num_epochs
        self.lr = self.train_setup.lr
        self.optimizer_type = self.train_setup.optimizer_config.type
        self.optimizer_params = self.train_setup.optimizer_config.params
        self.criterion_name = self.train_setup.criterion
        self.early_stopping_enabled = self.train_setup.early_stopping_config.enabled
        self.patience = self.train_setup.early_stopping_config.patience
        self.min_delta = self.train_setup.early_stopping_config.min_delta

        # -------------------------
        # Tuning Configuration
        # -------------------------
        self.tuning_config = self.config.tuning
        self.tuning_enabled = self.tuning_config.enabled
        self.max_evals = self.tuning_config.max_evals

        # -------------------------
        # Model Parameters
        # -------------------------
        self.model_params = (
            self.config.model_params
        )  # Keep model params as a nested dictionary

        # -------------------------
        # Mlflow Config
        # -------------------------
        self.experiment_id = experiment_id

        # Load data and apply preprocessing
        self._load_data()
        self._apply_preprocessing()

    def _load_data(self):
        """Loads the benign_train, benign_test, and malicious datasets using the paths provided in the CFG."""
        self.benign_train_data = pd.read_csv(self.benign_train_path)
        self.benign_test_data = pd.read_csv(self.benign_test_path)
        self.malicious_data = pd.read_csv(self.malicious_path)

    def _get_optimizer(self) -> torch.optim.Optimizer:
        optimizer_params = {
            **self.optimizer_params.__dict__,
            "lr": self.lr,
        }
        if self.optimizer_type == "Adam":
            return torch.optim.Adam(self.model.parameters(), **optimizer_params)
        elif self.optimizer_type == "AdamW":
            return torch.optim.AdamW(self.model.parameters(), **optimizer_params)
        else:
            raise ValueError(f"Unsupported optimizer type: {self.optimizer_type}")

    def _get_criterion(self) -> nn.Module:
        if self.criterion_name == "MSE":
            return nn.MSELoss()
        elif self.criterion_name == "L1":
            return nn.L1Loss()
        else:
            raise ValueError(f"Unsupported loss function type: {self.criterion_name}")

    def _apply_preprocessing(self):
        # Scaling
        self.benign_train_data, self.fitted_scaler = scale_ts(
            self.benign_train_data, self.feature_columns
        )

        self.benign_test_data, _ = scale_ts(
            self.benign_test_data,
            self.feature_columns,
            self.fitted_scaler,
        )

        self.malicious_data, _ = scale_ts(
            self.malicious_data,
            self.feature_columns,
            self.fitted_scaler,
        )

        # Smoothing and Differencing
        if self.preprocessing_method in ["scaled_smoothed", "scaled_smoothed_diff"]:
            rolling_window_len = self.data_config.preprocessing.rolling_window_len
            self.benign_train_data = smooth_ts(
                self.benign_train_data, self.feature_columns, rolling_window_len
            )
            self.benign_test_data = smooth_ts(
                self.benign_test_data, self.feature_columns, rolling_window_len
            )
            self.malicious_data = smooth_ts(
                self.malicious_data, self.feature_columns, rolling_window_len
            )

        if self.preprocessing_method == "scaled_smoothed_diff":
            self.benign_train_data = apply_diff(
                self.benign_train_data, self.feature_columns
            )
            self.benign_test_data = apply_diff(
                self.benign_test_data, self.feature_columns
            )
            self.malicious_data = apply_diff(self.malicious_data, self.feature_columns)

    def _create_data_loaders(self):
        self.train_data_loader, self.val_data_loader, self.mal_data_loader = (
            create_ds_loader(
                self.benign_train_data,
                self.malicious_data,
                self.time_window_length,
                self.step_size,
                self.feature_columns,
                self.train_batch_size,
            )
        )

        self.benign_test_data_loader, self.mal_test_data_loader = (
            create_test_ds_loaders(
                self.benign_test_data,
                self.malicious_data,
                self.time_window_length,
                self.test_step_size,
                self.feature_columns,
                self.test_batch_size,
            )
        )

    def build_model(self, device: torch.device):
        model_params = {
            "input_dim": len(self.feature_columns),
            "output_dim": len(self.feature_columns),
            **self.model_params.__dict__,  # Use the nested model_params directly
        }
        # print("Model params inside the build model")
        # print(model_params)
        self.model = self.model_class(**model_params).to(device)

        mlflow.log_params(model_params)

    def train_model(self, device: torch.device):
        early_stopping = None
        if self.early_stopping_enabled:
            early_stopping = EarlyStopping(self.patience, self.min_delta)

        self.criterion = self._get_criterion()
        self.optimizer = self._get_optimizer()

        self.history = self.model.train_model(
            num_epochs=self.num_epochs,
            early_stopping=early_stopping,
            train_data_loader=self.train_data_loader,
            val_data_loader=self.val_data_loader,
            mal_data_loader=self.mal_data_loader,
            device=device,
            criterion=self.criterion,
            optimizer=self.optimizer,
        )

        log_training_history(self.history)

    def eval_model(self, device):
        benign_test_losses, mal_test_losses = self.model.evaluate_model(
            self.criterion,
            self.benign_test_data_loader,
            self.mal_test_data_loader,
            device,
        )

        fpr, tpr, thresholds, roc_auc, optimal_threshold = calculate_threshold(
            benign_test_losses, mal_test_losses
        )

        accuracy, precision, recall, f1, tp_rate, tn_rate, fp_rate, fn_rate = infer(
            benign_test_losses, mal_test_losses, optimal_threshold
        )

        evaluation_metrics = {
            "roc_auc": roc_auc,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp_rate": tp_rate,
            "tn_rate": tn_rate,
            "fp_rate": fp_rate,
            "fn_rate": fn_rate,
        }

        log_evaluation_metrics(evaluation_metrics)
        log_plot_scatter_plot_rec_loss(benign_test_losses, mal_test_losses)
        log_plot_roc_curve(fpr, tpr, thresholds, roc_auc)

    def tune_model(self, device):
        def objective(params):
            params_obj = DictToObject(params)
            # -------------------------
            # Model Parameters
            # -------------------------
            # Update the model parameters using params_obj
            for attr_name, attr_value in vars(params_obj).items():
                print(f"{attr_name}, {attr_value}")
                setattr(self, attr_name, attr_value)

            mlflow.start_run(nested=True, experiment_id=self.experiment_id)

            try:
                mlflow.log_params(
                    {
                        "time_window_length": self.time_window_length,
                        "step_size": self.step_size,
                        "learning_rate": self.lr,
                        "optimizer_type": self.optimizer_type,
                        "criterion": self.criterion_name,
                    }
                )

                self._create_data_loaders()
                self.build_model(device)
                self.train_model(device)
                self.eval_model(device)
            finally:
                mlflow.end_run()

            return {"loss": min(self.history.val_losses), "status": "ok"}

        trials = Trials()
        best_params = fmin(
            fn=objective,
            space=search_space,
            algo=tpe.suggest,
            max_evals=self.max_evals,
            trials=trials,
        )
        return best_params
