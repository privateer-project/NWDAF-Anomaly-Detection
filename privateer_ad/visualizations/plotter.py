import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, roc_curve, auc


class Visualizer:
    def __init__(self):
        sns.set_theme('paper', 'white')
        self.figures = {}

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
        self.figures[f'{prefix}_confusion_matrix'] = self.create_confusion_matrix(y_true, y_pred, target_names)
        self.figures[f'{prefix}_roc_curve'] = self.create_roc_curve(y_true, y_pred, scores, target_names)
        # Create distribution plot if we have scores and threshold
        if scores is not None and threshold is not None:
            self.figures[f'{prefix}_distribution'] = self.create_distribution_plot(scores, y_true, threshold)

    @staticmethod
    def create_roc_curve(y_true, y_pred, scores, class_names):
        """Create ROC curves for multiclass classification using One-vs-Rest approach.

        Args:
            y_true: True labels (integer class indices)
            y_pred: Predicted probabilities for each class
            class_names: List of class names
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        if scores is not None:
            # Calculate ROC curve
            fpr, tpr, _ = roc_curve(y_true, scores)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f'Anomaly Detector (AUC = {roc_auc:.2f})')
            plt.title(f'ROC Curve', fontsize=14, pad=20)
        else:
            # Convert predictions to one-hot if they aren't already
            y_pred = np.eye(len(class_names))[y_pred]
            # Plot ROC curve for each class
            for i, class_name in enumerate(class_names):
                # Create binary labels for current class (one-vs-rest)
                y_true_binary = np.where(y_true == i, 1, 0)
                if y_pred.shape[-1] >= 2:
                    y_pred_binary = y_pred[:, i]
                else:
                    y_pred_binary = y_pred

                # Calculate ROC curve
                fpr, tpr, _ = roc_curve(y_true_binary, y_pred_binary)
                roc_auc = auc(fpr, tpr)

                plt.plot(fpr, tpr, lw=2, label=f'{class_name} (AUC = {roc_auc:.2f})')
            plt.title(f'ROC Curves (One-vs-Rest)', fontsize=14, pad=20)

        plt.plot([0, 1], [0, 1], 'k--', lw=2)

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        return fig

    @staticmethod
    def create_confusion_matrix(y_true, y_pred, class_names):
        """Create basic confusion matrix visualizations using seaborn"""

        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        # Normalize by true labels (rows) - equivalent to normalize='true'
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        # Create the plot
        plt.figure(figsize=(8, 6))

        # Create heatmap with seaborn
        sns.heatmap(cm_normalized,
                    annot=True,  # Show values in cells
                    fmt='.2f',  # Format as 2 decimal places
                    xticklabels=class_names,  # X-axis labels
                    yticklabels=class_names,  # Y-axis labels
                    cbar=False,  # No colorbar
                    square=True)  # Square cells

        # Set labels and title
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.tight_layout()

        return plt.gcf()  # Return current figure
    def create_distribution_plot(self, scores, y_true, threshold):
        """Create an enhanced distribution plot comparing benign and malicious scores.

        Args:
            scores: Array of anomaly scores
            y_true: Array of true labels (0 for benign, 1 for malicious)
            threshold: Detection threshold value
        """
        # Create figure with appropriate size and DPI for clarity
        fig, ax = plt.subplots(figsize=(12, 6), dpi=100)

        # Create pandas DataFrame for easier plotting
        df = pd.DataFrame({
            'Score': scores,
            'Category': np.where(y_true == 0, 'Benign', 'Malicious')
        })

        # Plot histograms using seaborn for better statistics
        sns.histplot(
            data=df,
            x='Score',
            hue='Category',
            alpha=0.5,
            palette={
                'Benign': 'green',
                'Malicious': 'red'
            },
            bins=100,  # Adjust number of bins as needed
            stat='density',  # Use density to normalize distributions
            common_norm=False,  # Normalize each distribution separately
            kde=True,  # Explicitly disable KDE overlay
            kde_kws= {'bw_adjust': 1.6}
        )

        # Add threshold line with annotation
        plt.axvline(
            x=threshold,
            color='black',
            linestyle='--',
            linewidth=2,
            label=f'Threshold ({threshold:.3f})'
        )

        # Get current axis limits for background regions
        x_min, x_max = plt.xlim()
        y_min, y_max = plt.ylim()

        # Fill background regions
        plt.fill_between(
            [x_min, threshold],
            y2=y_max,
            y1=y_min,
            color='green',
            alpha=0.1,
            label='Benign Region'
        )

        plt.fill_between(
            [threshold, x_max],
            y2=y_max,
            y1=y_min,
            color='red',
            alpha=0.1,
            label='Anomaly Region'
        )
        # Enhance plot aesthetics
        plt.xlabel('Anomaly Score', fontsize=12)
        plt.ylabel('Density', fontsize=12)
        plt.title(f'Score Distribution', fontsize=14, pad=20)
        plt.legend(title='Categories', title_fontsize=11, fontsize=10)

        # Add grid for better readability
        plt.grid(True, alpha=0.3, linestyle='--')

        # Adjust layout to prevent label clipping
        plt.tight_layout()

        return fig

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Cleanup figures"""
        for fig in self.figures.values():
            plt.close(fig)
