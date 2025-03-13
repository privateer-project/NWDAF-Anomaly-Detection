import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import ConfusionMatrixDisplay, roc_curve, auc


class Visualizer:
    def __init__(self):
        self.figures = {}

        self.colors = {
            'correct': '#2E8B57',  # dark green
            'wrong': '#8B0000',  # dark red
            'benign': 'green',
            'malicious': 'red'
        }

    def plot_training_history(self, history):
        """Plot training and validation metrics"""
        for metric in history.history:
            if not metric.startswith('val_'):
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(history.history[metric], label='Train')
                if f'val_{metric}' in history.history:
                    ax.plot(history.history[f'val_{metric}'], label='Validation')
                ax.set_title(f'{metric}')
                ax.set_xlabel('Epoch')
                ax.set_ylabel(metric)
                ax.legend()
                self.figures[f'{metric}_history'] = fig

    def visualize(self, y_true, y_pred, scores=None, threshold=None, target_names=None, prefix=''):
        """Generate visualizations figures"""
        # Create confusion matrix visualizations
        self.figures[f'{prefix}_confusion_matrix'] = self._create_confusion_matrix(y_true, y_pred, target_names, prefix)
        self.figures[f'{prefix}_roc_curve'] = self._create_roc_curve(y_true, y_pred, scores, target_names, prefix)
        # Create distribution plot if we have scores and threshold
        if scores is not None and threshold is not None:
            self.figures[f'{prefix}_distribution'] = self._create_distribution_plot(scores, y_true, threshold,
                                                                                    prefix=prefix)

    @staticmethod
    def _create_roc_curve(y_true, y_pred, scores, class_names, prefix=''):
        """Create ROC curves for multiclass classification using One-vs-Rest approach.

        Args:
            y_true: True labels (integer class indices)
            y_pred: Predicted probabilities for each class
            class_names: List of class names
            prefix: Prefix for plot title
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        if scores is not None:
            # Calculate ROC curve
            fpr, tpr, _ = roc_curve(y_true, scores)
            roc_auc = auc(fpr, tpr)
            label = f'classifier (AUC = {roc_auc:.2f})'
            plt.plot(fpr, tpr, lw=2, label=label)
            plt.title(f'{prefix}-ROC Curve', fontsize=14, pad=20)
        else:
            # Convert predictions to one-hot if they aren't already
            y_pred = np.eye(len(class_names))[y_pred]
            # Plot ROC curve for each class
            for i, class_name in enumerate(class_names):
                # Create binary labels for current class (one-vs-rest)
                y_true_binary = (y_true == i).astype(int)
                if y_pred.shape[-1] >= 2:
                    y_pred_binary = y_pred[:, i]
                else:
                    y_pred_binary = y_pred

                # Calculate ROC curve
                fpr, tpr, _ = roc_curve(y_true_binary, y_pred_binary)
                roc_auc = auc(fpr, tpr)

                plt.plot(fpr, tpr, lw=2, label=f'{class_name} (AUC = {roc_auc:.2f})')
                plt.title(f'{prefix}-ROC Curves (One-vs-Rest)', fontsize=14, pad=20)

        plt.plot([0, 1], [0, 1], 'k--', lw=2)

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        return fig

    @staticmethod
    def _create_confusion_matrix(y_true, y_pred, class_names, prefix=''):
        """Create basic confusion matrix visualizations"""
        # Compute confusion matrix

        cm = ConfusionMatrixDisplay.from_predictions(y_true=y_true,
                                                     y_pred=y_pred,
                                                     normalize='true',
                                                     display_labels=class_names,
                                                     colorbar=False,
                                                     cmap='inferno'
                                                     )
        cm.figure_.suptitle(f'{prefix}-Confusion Matrix')
        return cm.figure_

    def _create_distribution_plot(self, scores, y_true, threshold, prefix=''):
        """Create an enhanced distribution plot comparing benign and malicious scores.

        Args:
            scores: Array of anomaly scores
            y_true: Array of true labels (0 for benign, 1 for malicious)
            threshold: Detection threshold value
            prefix: Prefix for the plot title
        """
        # Create figure with appropriate size and DPI for clarity
        fig, ax = plt.subplots(figsize=(12, 6), dpi=100)

        # Create pandas DataFrame for easier plotting
        df = pd.DataFrame({
            'Score': scores,
            'Category': np.where(y_true == 0, 'Benign', 'Malicious')
        })

        # Plot distributions using seaborn for better statistics
        sns.kdeplot(
            data=df,
            x='Score',
            hue='Category',
            fill=True,
            alpha=0.5,
            palette={
                'Benign': self.colors['benign'],
                'Malicious': self.colors['malicious']
            },
            common_norm=False  # Normalize each distribution separately
        )

        # Add threshold line with annotation
        plt.axvline(
            x=threshold,
            color='black',
            linestyle='--',
            label=f'threshold ({threshold:.3f})'
        )



        plt.fill_between(
            [plt.xlim()[0], threshold],
            y2=plt.ylim()[1],
            y1=plt.ylim()[0],
            color='green',
            alpha=0.1,
            label='Benign Region'
        )

        plt.fill_between(
            [threshold, plt.xlim()[1]],
            y2=plt.ylim()[1],
            y1=plt.ylim()[0],
            color='red',
            alpha=0.1,
            label='Anomaly Region'
        )

        # Enhance plot aesthetics
        plt.xlabel('Anomaly Score', fontsize=12)
        plt.ylabel('Density', fontsize=12)
        plt.title(f'{prefix}-Score Distribution', fontsize=14, pad=20)
        plt.legend(title='Categories', title_fontsize=11, fontsize=10)

        # Add grid for better readability
        plt.grid(True, alpha=0.3, linestyle='--')

        # Adjust layout to prevent label clipping
        plt.tight_layout()

        return fig

    def close_figures(self):
        """Cleanup figures"""
        for fig in self.figures.values():
            plt.close(fig)
