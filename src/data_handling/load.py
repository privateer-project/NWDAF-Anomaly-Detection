from glob import glob
from typing import Dict
import os

import joblib
import pandas as pd
from pandas import DataFrame
from pytorch_forecasting import TimeSeriesDataSet
from torch.utils.data import DataLoader

from src.config import *
from src.data_handling.utils import partition_data, get_dataset_path

class NWDAFDataloader:
    """Handles creation of time series datasets."""

    def __init__(self,
                 features: Dict[str, FeatureInfo],
                 hparams: HParams,
                 paths: ProjectPaths,
                 partition_config: PartitionConfig = None):
        self.input_features = [feature for feature in features if features[feature].is_input]

        self.hparams = hparams
        self.paths = paths
        self.partition_config = partition_config

    def get_dataloaders(self, **kwargs) -> Dict[str, DataLoader]:
        """Get train, validation and test dataloaders."""
        datasets = {}
        for split in ['train', 'val', 'test']:
            train = False
            if split == 'train':
                train = True
            datasets[split] = self.get_single_dataloader(path=split, train=train, **kwargs)
        return datasets

    def get_single_dataloader(self, path: str, train: bool = True, **kwargs) -> DataLoader:
        """Get a single dataloader."""
        dataset = self.load_dataset(path=path)
        return dataset.to_dataloader(train=train, batch_size=self.hparams.batch_size, **kwargs)

    def load_dataset(self,
                     path: str,
                     ) -> TimeSeriesDataSet:
        """Load and process a dataset from a file path."""
        if path in ['train', 'val', 'test']:
            path = get_dataset_path(path)
        try:
            df = pd.read_csv(path)
        except FileNotFoundError:
            raise FileNotFoundError(f"File {path} not found.")

        if self.partition_config:
            df = partition_data(df, self.partition_config)

        return self.setup_ts_dataset(df, use_existing_scalers=True)

    def setup_scalers(self):
        train_df = pd.read_csv(get_dataset_path('train'))
        ds = self.setup_ts_dataset(train_df, use_existing_scalers=False)
        for name, scaler in ds.get_parameters()['scalers'].items():
            scaler_path = self._get_scaler_path(name)
            os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
            joblib.dump(scaler, scaler_path)

    def setup_ts_dataset(self, df: DataFrame, use_existing_scalers: bool =True) -> TimeSeriesDataSet:
        """Create a TimeSeriesDataSet from the preprocessed DataFrame."""
        scalers = {}
        if use_existing_scalers:
            scalers = self._load_scalers()
        df.reset_index(drop=True, inplace=True)
        df = df.sort_values(by=['_time'])
        df['time_idx'] = df.groupby('imeisv')['_time'].cumcount()

        return TimeSeriesDataSet(
            data=df,
            time_idx='time_idx',
            target='attack',
            group_ids=['imeisv'],
            max_encoder_length=self.hparams.seq_len,
            min_encoder_length=self.hparams.seq_len,
            min_prediction_length=1,
            max_prediction_length=1,
            time_varying_unknown_reals=self.input_features,
            scalers=scalers,
            target_normalizer=None,
            allow_missing_timesteps=False
        )

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
    hhparams = HParams()
    ppaths = ProjectPaths()
    features = MetaData().features
    ts_creator = NWDAFDataloader(features=features, hparams=hhparams, paths=ppaths)
    ts_creator.setup_scalers()
