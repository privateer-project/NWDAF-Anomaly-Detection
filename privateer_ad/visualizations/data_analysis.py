import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from privateer_ad.config import MetaData, PathsConf
from privateer_ad.etl.transform import DataProcessor


# Load data
def pca_plots(processed_df, pca, dim=(0, 1), labels=None):
    pca_features_df = processed_df[processed_df.columns[processed_df.columns.str.contains('pca')]]
    pca_features_df['attack'] = processed_df['attack']

    components_df = pd.DataFrame(pca.components_,
                                 columns=labels,
                                 index=['PCA' + str(x) for x in range(1, pca.n_components_ + 1)])
    explained_variance_ratio = pca.explained_variance_ratio_

    # Correlation Circle
    correlation_circle_path = paths.analysis.joinpath('correlation_circle.png')
    if not correlation_circle_path.exists():
        fig, ax = plt.subplots(figsize=(8, 8))
        for i, (x, y) in enumerate(zip(pca.components_[dim[0]], pca.components_[dim[1]])):
            ax.arrow(0, 0, x, y, head_width=0.05, head_length=0.05, fc='k', ec='k')
            if labels is not None:
                ax.text(x, y, labels[i], fontsize=12)
        circle = plt.Circle((0, 0), 1, color='blue', fill=False)
        ax.add_artist(circle)
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_xlabel(f'PC{dim[0] + 1}')
        ax.set_ylabel(f'PC{dim[1] + 1}')
        ax.set_title('Correlation Circle')
        plt.grid()
        plt.savefig(correlation_circle_path)
        plt.show()
        plt.close()

    # Plotting Explained Variance
    explained_variance_path = paths.analysis.joinpath('explained_variance_ratio.png')
    if not explained_variance_path.exists():
        plt.figure(figsize=(8, 8))
        plt.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, alpha=0.5, align='center', label='Individual explained variance')
        plt.step(range(1, len(explained_variance_ratio) + 1), np.cumsum(explained_variance_ratio), where='mid', label='Cumulative explained variance')
        plt.ylabel('Explained variance ratio')
        plt.xlabel('Principal component index')
        plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig(explained_variance_path)
        plt.show()
        plt.close()

    # PCA Components Heatmap
    contributions_path = paths.analysis.joinpath('contribution_to_explained_variance.png')
    if not contributions_path.exists():
        plt.figure(figsize=(12, 10))
        sns.heatmap(components_df, annot=True, cmap='viridis')
        plt.title('Feature Contributions to Explained Variance')
        plt.savefig(contributions_path)
        plt.show()
        plt.close()

    # Pairplot of Principal Components
    pairplot_path = paths.analysis.joinpath('pairplot.png')
    if not pairplot_path.exists():
        # Very slow do it once.
        sns.pairplot(pca_features_df, hue='attack')
        plt.savefig(pairplot_path)
        plt.show()
        plt.close()

    # Interactive 3D Scatter Plot of Principal Components
    principal_components_path = paths.analysis.joinpath('principal_components.html')
    if not principal_components_path.exists():
        fig = px.scatter_3d(pca_features_df, x='pca0', y='pca1', z='pca2', color='attack', title='3D Scatter Plot of Principal Components')
        fig.show()
        fig.write_html(principal_components_path)

if __name__ == '__main__':
    paths = PathsConf()
    metadata = MetaData()
    dp = DataProcessor()
    processed_df = dp.preprocess_data(paths.raw_dataset, use_pca=True)

    os.makedirs(paths.analysis, exist_ok=True)
    pca_plots(processed_df=processed_df,
              pca=dp.load_pca(),
              labels=metadata.get_input_features())
