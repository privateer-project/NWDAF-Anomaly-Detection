from glob import glob
from typing import Optional, Dict
import os

import joblib
import pandas as pd
from pandas import DataFrame
from pytorch_forecasting import TimeSeriesDataSet
from torch.utils.data import DataLoader

from src.config import FeatureInfo
from src.data_handling.transform import DataProcessor
from src.config import ProjectPaths, MetaData, PartitionConfig, HParams
from src.data_handling.utils import partition_data, get_dataset_path


class DataLoaderFactory:
    """Creates DataLoaders for training and evaluation."""

    def __init__(self, metadata: MetaData, paths: ProjectPaths, hparams: HParams):
        self.processor = DataProcessor(metadata, paths)
        self.dataset_creator = TSDatasetCreator(metadata.features, paths)
        self.hparams = hparams

    def get_dataloaders(self, **kwargs) -> Dict[str, DataLoader]:
        """Get train, validation and test dataloaders."""
        datasets = {}
        for split in ['train', 'val', 'test']:
            train = False
            if split == 'train':
                train = True
            datasets[split] = self.get_single_dataloader(split, train, **kwargs)
        return datasets

    def get_single_dataloader(self, path: str, train: bool = True, **kwargs) -> DataLoader:
        """Get a single dataloader."""
        dataset = self.dataset_creator.get_dataset(path, self.hparams.window_size,)
        return dataset.to_dataloader(train=train, batch_size=self.hparams.batch_size, **kwargs)


class TSDatasetCreator:
    """Handles creation of time series datasets."""

    def __init__(self, features: Dict[str, FeatureInfo], paths: ProjectPaths):
        self.features = features
        self.paths = paths

    def create_dataset(self, df: DataFrame, window_size: int, use_existing_scalers: bool =True) -> TimeSeriesDataSet:
        """Create a TimeSeriesDataSet from the preprocessed DataFrame."""
        scalers = {}
        if use_existing_scalers:
            scalers = self._load_scalers()
        df.reset_index(inplace=True)
        df['time_idx'] = df.groupby('imeisv')['_time'].cumcount()

        input_features = [feat for feat, info in self.features.items() if info.input]

        return TimeSeriesDataSet(
            data=df,
            time_idx='time_idx',
            target='attack',
            group_ids=['imeisv'],
            max_encoder_length=window_size,
            min_encoder_length=window_size,
            min_prediction_length=1,
            max_prediction_length=1,
            time_varying_unknown_reals=input_features,
            scalers=scalers,
            target_normalizer=None,
            allow_missing_timesteps=False
        )

    def get_dataset(self,
                    path: str,
                    window_size: int = 12,
                    partition_config: Optional[PartitionConfig] = None) -> TimeSeriesDataSet:
        """Load and process a dataset from a file path."""
        if path in ['train', 'val', 'test']:
            path = get_dataset_path(path)
        try:
            df = pd.read_csv(path)
        except FileNotFoundError:
            raise FileNotFoundError(f"File {path} not found.")

        if partition_config:
            df = partition_data(df, partition_config)

        return self.create_dataset(df, window_size)

    def setup_scalers(self, window_size):
        ds = self.create_dataset(pd.read_csv(get_dataset_path('train')),
                                 window_size=window_size,
                                 use_existing_scalers=False)
        for name, scaler in ds.get_parameters()['scalers'].items():
            joblib.dump(scaler, self._get_scaler_path(name))

    def _load_scalers(self) -> Dict:
        scalers = {}
        scaler_files = glob(str(self.paths.scalers.joinpath('*.scaler')))
        for scaler_path in scaler_files:
            col = os.path.splitext(os.path.basename(scaler_path))[0]
            scalers[col] = joblib.load(scaler_path)
        return scalers

    def _get_scaler_path(self, scaler_name: str) -> str:
        """Get full path for a scaler file."""
        return str(self.paths.scalers.joinpath(f"{scaler_name}.scaler"))


if __name__ == '__main__':
    metadata = MetaData()
    paths = ProjectPaths()
    hparams = HParams()
    ts_creator = TSDatasetCreator(metadata.features, paths)
    ts_creator.setup_scalers(hparams.window_size)