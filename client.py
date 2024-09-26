import warnings

warnings.filterwarnings("ignore")

import argparse
import flwr as fl
from flwr.common.logger import log
from logging import INFO, DEBUG

import torch
import torch.nn as nn
from collections import OrderedDict

import pandas as pd

from src.utils.config import Config
from src.utils.training_utils import *
from src.utils.evaluation_utils import *
from src.data.dataloader import *
from src.data.preprocessing import *
from src.models.rae import *


class DictToObject:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                setattr(self, key, DictToObject(value))
            else:
                setattr(self, key, value)


class FlowerClient(fl.client.NumPyClient):
    def __init__(
        self,
        model,
        train_data_loader,
        val_data_loader,
        mal_data_loader,
        benign_test_data_loader,
        mal_test_data_loader,
    ):
        self.model = model
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        self.mal_data_loader = mal_data_loader
        self.benign_test_data_loader = benign_test_data_loader
        self.mal_test_data_loader = mal_test_data_loader

        self.local_epochs = 3
        self.criterion = nn.L1Loss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.001)

    def get_parameters(self, config=None):
        log(INFO, f"Getting model parameters")
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters, config=None):
        log(INFO, f"Setting model parameters")
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config=None):
        log(INFO, f"Training the model locally")
        self.set_parameters(parameters)

        self.history = self.model.train_model(
            num_epochs=self.local_epochs,
            early_stopping=None,
            train_data_loader=self.train_data_loader,
            val_data_loader=self.val_data_loader,
            mal_data_loader=self.mal_data_loader,
            device=self.device,
            criterion=self.criterion,
            optimizer=self.optimizer,
        )

        return self.get_parameters(), len(self.train_data_loader), {}

    def evaluate(self, parameters, config=None):
        log(INFO, f"Evaluating the model locally")
        self.set_parameters(parameters)

        benign_test_losses, mal_test_losses = self.model.evaluate_model(
            self.criterion,
            self.benign_test_data_loader,
            self.mal_test_data_loader,
            self.device,
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

        return accuracy, len(self.benign_test_data_loader), evaluation_metrics


def main() -> None:
    # Parse command line argument `partition`
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--client-id",
        type=int,
        default=0,
        choices=range(0, 10),
        required=False,
        help="Specifies the artificial data partition of CIFAR10 to be used. \
        Picks partition 0 by default",
    )

    args = parser.parse_args()

    client_id = args.client_id

    fl.common.logger.configure(
        identifier="myFlowerExperiment", filename=f"log_flwr_client_{client_id}.txt"
    )

    benign_train_data = pd.read_csv(
        f"data/partitioned/client_{client_id}/benign_train.csv"
    )
    benign_test_data = pd.read_csv(
        f"data/partitioned/client_{client_id}/benign_test.csv"
    )
    malicious_data = pd.read_csv(f"data/partitioned/client_{client_id}/malicious.csv")

    feature_columns = ["ul_bitrate"]

    benign_train_data, fitted_scaler = scale_ts(benign_train_data, feature_columns)

    benign_test_data, _ = scale_ts(
        benign_test_data,
        feature_columns,
        fitted_scaler,
    )

    malicious_data, _ = scale_ts(
        malicious_data,
        feature_columns,
        fitted_scaler,
    )

    train_data_loader, val_data_loader, mal_data_loader = create_ds_loader(
        benign_train_data,
        malicious_data,
        120,
        40,
        feature_columns,
        32,
    )

    benign_test_data_loader, mal_test_data_loader = create_test_ds_loaders(
        benign_test_data,
        malicious_data,
        120,
        10,
        features=feature_columns,
        batch_size=1,
    )

    rae_model = LSTMAutoencoder(
        input_dim=len(feature_columns),
        hidden_dim1=50,
        hidden_dim2=100,
        output_dim=len(feature_columns),
        dropout=0.2,
        layer_norm_flag=False,
        num_layers=2,
    )

    flower_client = FlowerClient(
        rae_model,
        train_data_loader,
        val_data_loader,
        mal_data_loader,
        benign_test_data_loader,
        mal_test_data_loader,
    )

    fl.client.start_client(
        server_address="10.160.3.174:8080",
        client=flower_client.to_client(),
    )


if __name__ == "__main__":
    main()
