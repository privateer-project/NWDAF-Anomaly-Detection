# -*- coding: utf-8 -*-
""" main.py """

# standard
import argparse
import os
import time
import logging

# external
import mlflow
import warnings
import torch

# internal
from src.configs.config import CFG
from src.models.rae import LSTMAutoencoder
from src.utils.logging_utils import connect_to_mlflow

from src.logger.logger import setup_logger
from src.training.training_orchestrator import TrainingOrchestrator

# Create a logger instance
logger = setup_logger()

warnings.filterwarnings("ignore")


def main():
    # Configure the device (GPU/CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Connect to MLflow
    mlflow_cfg = CFG.get("mlflow_config", {})
    if mlflow_cfg.get("enabled", False):
        experiment_id, run_id = connect_to_mlflow(mlflow_cfg)
        print(f"Connected to MLflow. Run ID: {run_id}")
    else:
        print("MLflow tracking is disabled in the configuration.")

    # Initialize the Training Orchestrator with the model class and configuration
    orchestrator = TrainingOrchestrator(
        model_class=LSTMAutoencoder,
        config=CFG,
        experiment_id=experiment_id,
    )

    # Building the model
    # orchestrator.build_model(device=device)

    # Start training the model
    # orchestrator.train_model(device=device)

    # Evaluate the model
    # orchestrator.eval_model(device=device)

    # If tuning is enabled, start the hyperparameter tuning process
    if orchestrator.tuning_enabled:
        print("Starting model tuning...")
        best_params = orchestrator.tune_model(device=device)
        print(f"Best hyperparameters found: {best_params}")

    # End MLflow run
    if mlflow_cfg.get("enabled", False):
        mlflow.end_run()


if __name__ == "__main__":
    main()
