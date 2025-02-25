import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from src.config import PathsConf, MetaData
from src.data_utils.transform import DataProcessor

paths = PathsConf()
os.makedirs(paths.analysis, exist_ok=True)
metadata = MetaData()
# Load data
df = pd.read_csv('/home/sse/projects/NWDAF-Anomaly-Detection/data/raw/amari_ue_data_merged_with_attack_number.csv')

dp = DataProcessor(metadata=metadata, paths=paths)

processed_df = dp.process_data(df)
cleaned_df = dp.clean_data(df)
scaled_df = dp.scale_features(cleaned_df)
pca = dp.get_pca()

pca_data = dp.apply_pca(scaled_df)
pca_data = pca_data[pca_data.columns[pca_data.columns.str.contains('pca')]]

# Explained Variance by Principal Components
explained_variance = pca.explained_variance_
explained_variance_ratio = pca.explained_variance_ratio_

# Plotting Explained Variance
plt.figure(figsize=(8, 5))
plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.5, align='center', label='Individual explained variance')
plt.step(range(1, len(explained_variance_ratio) + 1), np.cumsum(explained_variance_ratio), where='mid', label='Cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal component index')
plt.legend(loc='best')
plt.tight_layout()
plt.savefig(os.path.join(paths.analysis, 'explained_variance_ratio.png'))
plt.show()
plt.close()

# PCA Components Heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(pca_data, cmap='viridis', yticklabels=['PCA'+str(x) for x in range(1, pca.n_components_ + 1)], xticklabels=list(metadata.get_input_features()))
plt.xlabel('Feature')
plt.ylabel('Principal Component')

plt.savefig(os.path.join(paths.analysis, 'pca_heatmap.png'))
plt.show()
plt.close()

# Pairplot of Principal Components
pca_df = pd.DataFrame(data=pca_data, columns=['PCA'+str(x) for x in range(1, pca.n_components_ + 1)])
sns.pairplot(pca_df)

plt.savefig(os.path.join(paths.analysis, 'pairplot.png'))
plt.show()
plt.close()

# Feature Contributions to Explained Variance
contributions = pd.DataFrame(pca.components_, columns=metadata.get_input_features(), index=['PCA'+str(x) for x in range(1, pca.n_components_ + 1)])
plt.figure(figsize=(10, 8))
sns.heatmap(contributions, annot=True, cmap='viridis')
plt.title('Feature Contributions to Explained Variance')

plt.savefig(os.path.join(paths.analysis, 'contribution_to_explained_variance.png'))
plt.show()
plt.close()

# Interactive 3D Scatter Plot of Principal Components
fig = px.scatter_3d(pca_df, x='PCA1', y='PCA2', z='PCA3', color='PCA1', title='3D Scatter Plot of Principal Components')
plt.savefig(os.path.join(paths.analysis, 'principal_components'))
plt.show()
plt.close()

# Correlation Circle
def correlation_circle(pca, dim=(1, 2), labels=None):
    fig, ax = plt.subplots(figsize=(8, 8))
    for i, (x, y) in enumerate(zip(pca.components_[dim[0] - 1], pca.components_[dim[1] - 1])):
        ax.arrow(0, 0, x, y, head_width=0.05, head_length=0.05, fc='k', ec='k')
        if labels is not None:
            ax.text(x, y, labels[i], fontsize=12)
    circle = plt.Circle((0, 0), 1, color='blue', fill=False)
    ax.add_artist(circle)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_xlabel(f'PC{dim[0]}')
    ax.set_ylabel(f'PC{dim[1]}')
    ax.set_title('Correlation Circle')
    plt.grid()
    plt.savefig(os.path.join(paths.analysis, 'corr_circle.png'))
    plt.show()
    plt.close()

correlation_circle(pca, labels=cleaned_df.columns)