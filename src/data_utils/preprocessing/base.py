from src.config import Paths, MetaData


class BasePreprocessor:
    """Base class for preprocessing functionality."""

    def __init__(self, metadata: MetaData, paths: Paths):
        self.features = metadata.features
        self.attacks = metadata.attacks
        self.devices = metadata.devices
        self.paths = paths
        self.ensure_directories()

    def ensure_directories(self):
        """Create necessary directories if they don't exist."""
        for path in [self.paths.scalers, self.paths.processed]:
            path.mkdir(parents=True, exist_ok=True)

    def get_scaler_path(self, scaler_name: str) -> str:
        """Get full path for a scaler file."""
        return str(self.paths.scalers.joinpath(f"{scaler_name}.scaler"))

    def get_data_path(self, dataset_name: str) -> str:
        """Get full path for a dataset file."""
        return str(self.paths.processed.joinpath(f"{dataset_name}.csv"))