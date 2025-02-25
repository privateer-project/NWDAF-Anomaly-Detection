import os

from src.config import PathsConf


def get_dataset_path(dataset_name: str) -> str:
    """Get full path for a dataset file."""
    return str(PathsConf.processed.joinpath(f"{dataset_name}.csv"))

def check_existing_datasets():
    for mode in ['train', 'val', 'test']:
        path = get_dataset_path(mode)
        if os.path.exists(path):
            raise FileExistsError(f'File {path} exists.')

def get_scaler_path(scaler_name: str) -> str:
    """Get full path for a scaler file."""
    return str(PathsConf.scalers.joinpath(f"{scaler_name}.scaler"))


# def biplot(score, coeff, labels=None):
#     plt.figure(figsize=(10, 8))
#     xs = score[:, 0]
#     ys = score[:, 1]
#     n = coeff.shape[0]
#     scalex = 1.0 / (xs.max() - xs.min())
#     scaley = 1.0 / (ys.max() - ys.min())
#     plt.scatter(xs * scalex, ys * scaley, alpha=0.5)
#
#     for i in range(n):
#         plt.arrow(0, 0, coeff[i, 0], coeff[i, 1], color='r', alpha=0.5)
#         if labels is None:
#             plt.text(coeff[i, 0] * 1.15, coeff[i, 1] * 1.15, f"Var{i + 1}", color='g', ha='center', va='center')
#         else:
#             plt.text(coeff[i, 0] * 1.15, coeff[i, 1] * 1.15, labels[i], color='g', ha='center', va='center')
#
#     plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%})")
#     plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%})")
#     plt.grid(True)
#     plt.title('Biplot of PCA')
#     plt.show()

# plt.figure(figsize=(10, 6))
# plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o', linestyle='--')
# plt.xlabel('Number of Components')
# plt.ylabel('Cumulative Explained Variance')
# plt.title('Explained Variance by Principal Components')
# plt.grid(True)
# plt.show()
# plt.figure(figsize=(10, 6))
# plt.bar(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_, alpha=0.7,
#         align='center')
# plt.xlabel('Principal Component')
# plt.ylabel('Explained Variance Ratio')
# plt.title('Scree Plot')
# plt.grid(True)
# plt.show()
# plt.figure(figsize=(12, 8))
#
# sns.heatmap(pca.components_,
#             cmap='viridis',
#             yticklabels=[f"PC{i + 1}" for i in range(pca.n_components_)],
#             xticklabels=self.metadata.get_input_features(),
#             annot=True,
#             fmt=".2f")
# plt.xlabel('Features')
# plt.ylabel('Principal Components')
# plt.title('PCA Components Heatmap')
# plt.show()
#
# biplot(scores, pca.components_.T, labels=self.metadata.get_input_features())
# pca_df = pd.DataFrame(scores, columns=[f"PC{i + 1}" for i in range(pca.n_components_)])
# sns.pairplot(pca_df.iloc[:, :3], diag_kind='kde')
# plt.suptitle('Pairplot of Principal Components', y=1.02)
# plt.show()
# feature_contributions = np.abs(pca.components_) * pca.explained_variance_ratio_[:, np.newaxis]
#
# # Plot feature contributions
# plt.figure(figsize=(12, 8))
# sns.heatmap(feature_contributions.T,
#             cmap='viridis',
#             yticklabels=self.metadata.get_input_features(),
#             xticklabels=[f"PC{i + 1}" for i in range(pca.n_components_)],
#             annot=True,
#             fmt=".2f")
# plt.xlabel('Principal Components')
# plt.ylabel('Features')
# plt.title('Feature Contributions to Explained Variance')
# plt.show()
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(pca_df['PC1'], pca_df['PC2'], pca_df['PC3'], alpha=0.5)
# ax.set_xlabel('PC1')
# ax.set_ylabel('PC2')
# ax.set_zlabel('PC3')
# ax.set_title('3D Scatter Plot of Principal Components')
# plt.show()
#
# def correlation_circle(pca, features):
#     plt.figure(figsize=(8, 8))
#     for i, feature in enumerate(features):
#         plt.arrow(0, 0, pca.components_[0, i], pca.components_[1, i], color='r', alpha=0.5)
#         plt.text(pca.components_[0, i] * 1.15, pca.components_[1, i] * 1.15, feature, color='g', ha='center',
#                  va='center')
#     circle = plt.Circle((0, 0), 1, color='blue', fill=False)
#     plt.gca().add_artist(circle)
#     plt.xlim(-1, 1)
#     plt.ylim(-1, 1)
#     plt.xlabel('PC1')
#     plt.ylabel('PC2')
#     plt.title('Correlation Circle')
#     plt.grid(True)
#     plt.show()
# correlation_circle(pca, self.metadata.get_input_features())
