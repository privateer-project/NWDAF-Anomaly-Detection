import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_train_val_loss(train_losses, val_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Over Epochs")
    plt.legend()
    plt.show()


def plot_scatter_plot_rec_loss(benign_data_mse_losses, mal_data_mse_losses):
    index_benign = list(range(len(benign_data_mse_losses)))
    index_mal = list(range(len(mal_data_mse_losses)))

    plt.figure(figsize=(10, 6))

    plt.scatter(index_benign, benign_data_mse_losses, color="blue", label="Benign")
    plt.scatter(index_mal, mal_data_mse_losses, color="red", label="")

    plt.title("Scatter Plot of Benign & Malicious MSE loss")
    plt.xlabel("batch num")
    plt.ylabel("MSE loss")
    plt.legend()

    plt.show()


def plot_roc_curve(fpr, tpr, thresholds, roc_auc):
    plt.figure(figsize=(7, 5.5))

    plt.plot(fpr, tpr, "b", label="AUC = %0.2f" % roc_auc)

    plt.plot([0, 1], [0, 1], "r--")

    plt.xlim([0, 1])
    plt.ylim([0, 1])

    plt.title("Receiver Operating Characteristic")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")

    # Annotate 20% of the thresholds
    num_annotations = max(1, int(len(thresholds) * 0.2))
    indices_to_annotate = np.linspace(
        0, len(thresholds) - 1, num=num_annotations, dtype=int
    )

    for i in indices_to_annotate:
        plt.annotate(
            f"{thresholds[i]:.4f}",
            (fpr[i], tpr[i]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=8,
        )

    plt.show()
