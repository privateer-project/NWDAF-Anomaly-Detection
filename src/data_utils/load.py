import os
from typing import Dict

import pandas as pd
from pandas import DataFrame
from pytorch_forecasting import TimeSeriesDataSet
from torch.utils.data import DataLoader

from src.config import HParams, logger
from src.data_utils.utils import get_dataset_path

class NWDAFDataloader:
    """Handles creation of time series datasets."""

    def __init__(self, batch_size, seq_len):
        self.batch_size = batch_size
        self.seq_len = seq_len

    def get_dataloaders(self) -> Dict[str, DataLoader]:
        """Get train, validation and test dataloaders."""
        dataloader_params = {'num_workers': os.cpu_count(),
                             'pin_memory': True,
                             'prefetch_factor': self.batch_size * 100,
                             'persistent_workers': True}
        dataloaders = {}
        logger.info('Loading datasets...')
        for split in ['train', 'val', 'test']:
            path = get_dataset_path(split)
            try:
                df = pd.read_csv(path)
            except FileNotFoundError:
                raise FileNotFoundError(f"File {path} not found.")
            if split in ('train', 'val'):
                df = df[df['attack'] == 0]
            df = df.sort_values(by=['_time'])

            dataset = self.get_ts_dataset(df)
            dataloaders[split] = dataset.to_dataloader(train=split == 'train',
                                                       batch_size=self.batch_size,
                                                       **dataloader_params)
        logger.info('Finished loading datasets.')
        return dataloaders

    def get_ts_dataset(self, df: DataFrame) -> TimeSeriesDataSet:
        """Create a TimeSeriesDataSet from the preprocessed DataFrame."""
        input_columns = [column for column in df.columns if column.startswith('pca')]
        df['time_idx'] = df.groupby('imeisv')['_time'].cumcount()
        return TimeSeriesDataSet(
            data=df,
            time_idx='time_idx',
            target='attack',
            group_ids=['imeisv'],
            max_encoder_length=self.seq_len,
            max_prediction_length=1,
            time_varying_unknown_reals=input_columns,
            scalers=None,
            target_normalizer=None,
            allow_missing_timesteps=False)
