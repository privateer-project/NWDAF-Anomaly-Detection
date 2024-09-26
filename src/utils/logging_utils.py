from typing import Type, Dict, Any
import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import mlflow
from mlflow.tracking import MlflowClient
import tensorflow as tf
from dotenv import load_dotenv
from src.logger.logger import setup_logger

from src.utils.training_utils import TrainingHistory
from src.utils.visualization_utils import *

# Create a logger instance
logger = setup_logger()


def connect_to_mlflow(mlflow_cfg):
    # Load environment variables from .env file
    # load_dotenv()

    # mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    mlflow_tracking_uri = "http://127.0.0.1:8080"

    if mlflow_tracking_uri:
        mlflow.set_tracking_uri(mlflow_tracking_uri)

    experiment_name = mlflow_cfg["experiment_name"]
    experiment_id = retrieve_mlflow_experiment_id(experiment_name, create=True)
    mlflow.start_run(experiment_id=experiment_id)
    # Get the run ID
    run_id = mlflow.active_run().info.run_id
    logger.info(f"MLflow run {run_id} (stored in local_artifacts/logs.log)")
    return experiment_id, run_id


def retrieve_mlflow_experiment_id(name, create=False):
    experiment_id = None
    if name:
        existing_experiment = MlflowClient().get_experiment_by_name(name)
        if existing_experiment and existing_experiment.lifecycle_stage == "active":
            experiment_id = existing_experiment.experiment_id
        else:
            if create:
                experiment_id = mlflow.create_experiment(name)
            else:
                raise Exception(
                    f'Experiment "{name}" not found in {mlflow.get_tracking_uri()}'
                )

    if experiment_id is not None:
        experiment = MlflowClient().get_experiment(experiment_id)
        logger.info("Experiment name: {}".format(experiment.name))
        logger.info("Experiment_id: {}".format(experiment.experiment_id))
        logger.info("Artifact Location: {}".format(experiment.artifact_location))
        logger.info("Lifecycle_stage: {}".format(experiment.lifecycle_stage))

    return experiment_id


def log_training_history(history: TrainingHistory):
    # Log metrics (train, val, mal losses, epochs)
    for epoch, (train_loss, val_loss, mal_loss) in enumerate(
        zip(history.train_losses, history.val_losses, history.mal_losses), start=1
    ):
        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("val_loss", val_loss, step=epoch)
        mlflow.log_metric("mal_loss", mal_loss, step=epoch)

    mlflow.log_metric("epochs_trained", history.epochs_trained)

    gradient_norms_file = "gradient_norms.json"
    with open(gradient_norms_file, "w") as f:
        json.dump(history.gradient_norms, f)
    mlflow.log_artifact(gradient_norms_file)

    model_weights_file = "model_weights.pth"
    torch.save(history.model_weights, model_weights_file)
    mlflow.log_artifact(model_weights_file)

    log_plot_train_val_loss(history.train_losses, history.val_losses)


def log_evaluation_metrics(metrics: Dict[str, float], step: int = None):
    """
    Logs evaluation metrics using the MLflow Tracking API.

    Args:
        metrics (Dict[str, float]): Dictionary of metric names and their values.
        step (int, optional): The step or epoch to associate with the logged metrics. Defaults to None.
    """
    for metric_name, value in metrics.items():
        mlflow.log_metric(metric_name, value, step=step)


def write_confusion_matrix_to_md(conf_matrix: np.ndarray):
    with open("confusion_matrix.md", "w") as file:
        if conf_matrix.shape == (1, 1):
            file.write(f"|       | {conf_matrix[0, 0]} |\n")
            file.write(f"|-------|-------|\n")
            file.write(f"| True 0| {conf_matrix[0, 0]} |\n")
        else:
            file.write("|       | Predicted 0 | Predicted 1 |\n")
            file.write("|-------|-------------|-------------|\n")

            for i in range(conf_matrix.shape[0]):
                file.write(
                    f"| True {i} | {conf_matrix[i, 0]}        | {conf_matrix[i, 1]}        |\n"
                )


def log_plot_train_val_loss(
    train_losses, val_losses, artifact_path="train_val_loss.png"
):
    plot_train_val_loss(train_losses, val_losses)
    plt.savefig(artifact_path)
    mlflow.log_artifact(artifact_path)
    os.remove(artifact_path)


def log_plot_scatter_plot_rec_loss(
    benign_data_mse_losses,
    mal_data_mse_losses,
    artifact_path="scatter_plot_rec_loss.png",
):
    plot_scatter_plot_rec_loss(benign_data_mse_losses, mal_data_mse_losses)
    plt.savefig(artifact_path)
    mlflow.log_artifact(artifact_path)
    os.remove(artifact_path)


def log_plot_roc_curve(fpr, tpr, thresholds, roc_auc, artifact_path="roc_curve.png"):
    plot_roc_curve(fpr, tpr, thresholds, roc_auc)
    plt.savefig(artifact_path)
    mlflow.log_artifact(artifact_path)
    os.remove(artifact_path)
