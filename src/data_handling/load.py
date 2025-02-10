from glob import glob
from typing import Optional, Tuple, Dict
import os

import joblib
import pandas as pd
from pandas import DataFrame
from pytorch_forecasting import TimeSeriesDataSet
from torch.utils.data import DataLoader

from config import FeatureInfo
from data_handling.utils import partition_data
from config import Paths, MetaData, PartitionConfig, HParams

class DataLoaderFactory:
    """Creates DataLoaders for training and evaluation."""

    def __init__(self, metadata: MetaData, paths: Paths, hparams: HParams):
        self.paths = paths
        self.creator = TSDatasetCreator(metadata.features, paths)
        self.hparams = hparams

    def get_dataloaders(self, window_size: int = 12,
                        partition_config: Optional[PartitionConfig] = None,
                        train: bool = True
                        ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Get train, validation and test dataloaders."""
        datasets = {}
        for split in ['train', 'val', 'test']:
            df = pd.read_csv(self.paths.processed.joinpath(f'{split}.csv'))

            if partition_config:
                df = partition_data(df, partition_config)

            dataset = self.creator.create_dataset(df, window_size)
            datasets[split] = dataset.to_dataloader(
                train=train and split == 'train',
                batch_size=self.hparams.batch_size,
                num_workers=os.cpu_count(),
                pin_memory=True,
                prefetch_factor=self.hparams.batch_size * 100,
                persistent_workers=True
            )

        return datasets['train'], datasets['val'], datasets['test']

    def get_single_dataloader(self, split: str, window_size: int = 12,
                              partition_config: Optional[PartitionConfig] = None,
                              train: bool = True, **kwargs) -> DataLoader:
        """Get a single dataloader."""
        df = pd.read_csv(self.paths.processed.joinpath(f'{split}.csv'))

        if partition_config:
            df = partition_data(df, partition_config)

        dataset = self.creator.create_dataset(df, window_size)
        return dataset.to_dataloader(
            train=train and split == 'train',
            batch_size=self.hparams.batch_size, **kwargs)

class TSDatasetCreator:
    """Handles creation of time series datasets."""

    def __init__(self, features: Dict[str, FeatureInfo], paths: Paths):
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

    def _load_scalers(self) -> Dict:
        scalers = {}
        scaler_files = glob(str(self.paths.scalers.joinpath('*.scaler')))
        for scaler_path in scaler_files:
            col = os.path.splitext(os.path.basename(scaler_path))[0]
            scalers[col] = joblib.load(scaler_path)
        return scalers
