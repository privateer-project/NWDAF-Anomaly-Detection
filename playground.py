import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from torch.utils.data import DataLoader

from src.config import MetaData, PathsConf, HParams
from src.data_utils.load import DataLoaderFactory

def extract_features_and_labels(dataloader: DataLoader) -> (np.ndarray, np.ndarray):
    """Extract features and labels from a DataLoader."""
    features = []
    labels = []

    for batch in dataloader:
        x, y = batch
        # Flatten the time series data
        x_flat = x['encoder_cont'].view(x['encoder_cont'].size(0), -1).numpy()
        features.append(x_flat)
        labels.append(y[0].numpy())

    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)

    return features, labels


def train_random_forest(dataloader: DataLoader):
    """Train a Random Forest model."""
    # Extract features and labels
    X_train, y_train = extract_features_and_labels(dataloader)
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train)

    # Initialize and train the model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = rf_model.predict(X_test)
    print("Random Forest Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

    return rf_model


def train_xgboost(dataloader: DataLoader):
    """Train an XGBoost model."""
    # Extract features and labels
    X_train, y_train = extract_features_and_labels(dataloader)
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train)

    # Initialize and train the model
    xgb_model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    xgb_model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = xgb_model.predict(X_test)
    print("XGBoost Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    return xgb_model


if __name__ == '__main__':
    metadata = MetaData()
    paths = PathsConf()
    hparams = HParams()

    # Initialize the DataLoaderFactory
    loader_factory = DataLoaderFactory(metadata, paths, hparams)

    # Get the dataloaders
    dataloaders = loader_factory.get_dataloaders()
    # Train Random Forest
    print("Training Random Forest...")
    rf_model = train_random_forest(dataloaders['test'])

    # Train XGBoost
    print("Training XGBoost...")
    xgb_model = train_xgboost(dataloaders['test'])
