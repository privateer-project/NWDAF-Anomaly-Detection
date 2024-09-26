# -*- coding: utf-8 -*-
""" Model config in json format """

# src/configs/config.py

from pathlib import Path

config_dir = Path(__file__).resolve().parent
base_dir = config_dir.parent.parent

print(base_dir / "data/raw/nwdaf_classic/amari_ue_data.csv")

CFG = {
    "data": {
        # Defaults for tabular_data-extraction.py module
        "amari_ue_classic_input_filepath": str(
            base_dir / "data/raw/nwdaf_classic/amari_ue_data.csv"
        ),  # Default input file path for 'amari_ue classic'
        "amari_ue_mini_input_filepath": str(
            base_dir / "data/raw/nwdaf_mini/amari_ue_data.csv"
        ),  # Default input file path for 'amari_ue mini'
        "amari_ue_output_filepath": str(
            base_dir / "data/interim/amari_ue_labeled.csv"
        ),  # Default output file path for 'amari_ue'
        "enb_counters_input_filepath": "path/to/enb_counters_input.csv",  # Default input file path for 'enb_counters'
        "enb_counters_output_filepath": "path/to/enb_counters_output.csv",  # Default output file path for 'enb_counters'
        "attacks": [
            ("2024-03-23 21:26:00", "2024-03-23 22:23:00"),
            ("2024-03-23 22:56:00", "2024-03-23 23:56:00"),
        ],  # Default attack periods
        "malicious_imeisv": [
            "8642840401612300",
            "8642840401624200",
            "8642840401594200",
            "3557821101183501",
            "8609960480859057",
            "8628490433231157",
            "8609960468879057",
            "8609960480666910",
        ],  # Default list of malicious IMEIs or phone numbers
        "benign_train_path": str(base_dir / "data/interim/benign_train.csv"),
        "benign_test_path": str(base_dir / "data/interim/benign_test.csv"),
        "malicious_path": str(base_dir / "data/interim/malicious.csv"),
        "feature_columns": [
            # "dl_bitrate",
            "ul_bitrate",
            # "cell_x_dl_retx",
            # "cell_x_dl_tx",
            # "cell_x_ul_retx",
            # "cell_x_ul_tx",
            # "ul_total_bytes_non_incr",
            # "dl_total_bytes_non_incr",
        ],
        "preprocessing": {
            "preprocessing_method": "scaled_smoothed",
            "rolling_window_len": 10,
        },
    },
    "dataloaders": {
        "time_window_length": 120,
        "step_size": 40,
        "test_step_size": 10,
        "train_batch_size": 32,
        "test_batch_size": 5,
    },
    "train_setup": {
        "num_epochs": 30,
        "lr": 0.001,
        "optimizer_config": {
            "type": "AdamW",
            "params": {"betas": [0.9, 0.999], "eps": 1e-08, "weight_decay": 0.01},
        },
        "criterion": "L1",
        "early_stopping_config": {"enabled": True, "patience": 5, "min_delta": 0.001},
    },
    "tuning": {
        "enabled": True,
        "max_evals": 50,
    },
    "model_params": {
        "hidden_dim1": 50,
        "hidden_dim2": 100,
        "num_layers": 2,
        "dropout": 0.2,
        "layer_norm_flag": False,
    },
    "mlflow_config": {"enabled": True, "experiment_name": "nwdaf_amari_ue_lstm_ae_01"},
}
