import os
from copy import deepcopy

import joblib
import pandas as pd
from pandas import DataFrame
from datasets import Dataset
from pytorch_forecasting import TimeSeriesDataSet
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from flwr_datasets.partitioner import PathologicalPartitioner
from torch.utils.data import DataLoader

import sys
sys.path.append(os.path.join(sys.path[0], 'common_libraries'))

from privateer_ad import logger
from privateer_ad.config import PathsConf, MetaData
from privateer_ad.etl.utils import check_existing_datasets, get_dataset_path


class DataProcessor:
    """
    Main class responsible for transforming and preparing datasets for modeling.
    It supports dataset splitting, cleaning, normalization, partitioning, and loader creation.
    """

    def __init__(self, partition=False):
        self.metadata = MetaData()
        self.paths = PathsConf()
        self.scaler_path = os.path.join(self.paths.scalers, 'scaler.pkl')
        self.pca = None
        self.scaler = None
        self.input_features = self.metadata.get_input_features()
        self.drop_features = self.metadata.get_drop_features()
        self.partition = partition

    def prepare_datasets(self, raw_dataset_path, train_size: float = 0.8) -> None:
        """
        Main method that loads raw CSV, processes it, and saves train/val/test datasets.
        """
        check_existing_datasets()
        path = get_dataset_path(raw_dataset_path)
        raw_df = self._read_csv(path)
        logger.info(f'Loaded data from {path}')

        datasets = self.split_data(raw_df, train_size=train_size)

        self.setup_scaler(self.clean_data(datasets['train']))

        os.makedirs(self.paths.processed, exist_ok=True)
        logger.info('Saving processed datasets...')
        for k, df in datasets.items():
            save_path = get_dataset_path(k)
            processed_df = self.preprocess_data(df)
            logger.warning(f'{k} size: {len(processed_df)}')
            processed_df.to_csv(save_path, index=False)
            logger.info(f'{k} saved to {save_path}')

    def split_data(self, df, train_size=0.8):
        """
        Split data by device ID into stratified train/val/test subsets.
        """
        train_dfs, val_dfs, test_dfs = [], [], []
        for device, device_info in self.metadata.devices.items():
            device_df = df.loc[df['imeisv'] == device_info.imeisv]
            logger.debug(f'Device {device}: pre-split attack/benign counts')

            device_in_attacks = device_df['attack_number'].isin(device_info.in_attacks)
            device_df.loc[device_in_attacks, 'attack'] = 1
            device_df.loc[~device_in_attacks, 'attack'] = 0

            device_train_df, df_tmp = train_test_split(device_df, train_size=train_size,
                                                       stratify=device_df['attack_number'], random_state=42)
            device_val_df, device_test_df = train_test_split(df_tmp, test_size=0.5,
                                                             stratify=df_tmp['attack_number'], random_state=42)
            train_dfs.append(device_train_df)
            val_dfs.append(device_val_df)
            test_dfs.append(device_test_df)

        datasets = {
            'train': pd.concat(train_dfs),
            'val': pd.concat(val_dfs),
            'test': pd.concat(test_dfs)
        }

        for k, df in datasets.items():
            logger.debug(f'{k}: attack/benign counts and shape {df.shape}')
        return datasets

    def clean_data(self, df):
        """
        Remove missing values, duplicates, and unnecessary columns.
        """
        if '_time' in df.columns:
            df = df[df['_time'] != '<NA>']
        df = df.drop(columns=self.drop_features, errors='ignore')
        df = df.drop_duplicates().dropna(axis='rows').reset_index(drop=True)
        return df

    def preprocess_data(self, df, partition_id=0) -> DataFrame:
        """
        Perform partitioning, cleaning, scaling, and sorting on the dataset.
        """
        if self.partition:
            df = self.get_partition(df, partition_id=partition_id)
        df = self.clean_data(df)
        df = self.apply_scale(df)
        return df.sort_values(by=['_time']).reset_index(drop=True)

    def get_partition(self, df: DataFrame, partition_id=0) -> DataFrame:
        """
        Partition the dataset using federated learning scheme based on 'cell' column.
        """
        num_partitions = df['cell'].nunique()
        logger.info(f'Loading partition {partition_id + 1}/{num_partitions}')

        if not (0 <= partition_id < num_partitions):
            raise ValueError(f'Invalid partition_id {partition_id}')

        partitioner = PathologicalPartitioner(
            num_partitions=num_partitions,
            num_classes_per_partition=1,
            partition_by='cell',
            class_assignment_mode='deterministic')

        partitioner.dataset = Dataset.from_pandas(df)
        return partitioner.load_partition(partition_id).to_pandas(batched=False)[df.columns]

    def setup_scaler(self, df):
        """
        Fit and save a scaler using only benign (non-attack) data.
        """
        benign_df = df[df['attack'] == 0].copy()
        os.makedirs(os.path.dirname(self.scaler_path), exist_ok=True)
        scaler = StandardScaler()
        scaler.fit(benign_df[self.input_features])
        joblib.dump(scaler, self.scaler_path)

    def load_scaler(self):
        """
        Load the previously saved scaler.
        """
        if self.scaler is None:
            self.scaler = joblib.load(self.scaler_path)
        return self.scaler

    def apply_scale(self, df):
        """
        Apply standard scaling to numeric input features.
        """
        self.load_scaler()
        df.loc[:, self.input_features] = self.scaler.transform(df[self.input_features])
        return df

    def _read_csv(self, path):
        """
        Read a CSV dataset with proper dtypes and datetime parsing.
        """
        path = get_dataset_path(path)
        dtypes = deepcopy(self.metadata.get_features_dtypes())
        dtypes.pop('_time', None)
        try:
            return pd.read_csv(path, dtype=self.metadata.get_features_dtypes(), parse_dates=['_time'])
        except FileNotFoundError:
            raise FileNotFoundError(f'File {path} not found.')

    def get_dataloader(self, path, batch_size=1024, seq_len=6, partition_id=0, only_benign=False) -> DataLoader:
        """
        Build a DataLoader from the CSV path (used for train/test/val).
        """
        params = {
            'train': path == 'train',
            'batch_size': batch_size,
            'num_workers': os.cpu_count(),
            'pin_memory': True,
            'prefetch_factor': 10000,
            'persistent_workers': True
        }
        df = self.preprocess_data(df=self._read_csv(path), partition_id=partition_id)
        df = df.astype({'attack': int})
        if only_benign and 'attack' in df.columns:
            df = df[df['attack'] == 0]

        df['time_idx'] = df.groupby('imeisv')['_time'].cumcount()
        return TimeSeriesDataSet(
            data=df,
            time_idx='time_idx',
            target='attack',
            group_ids=['imeisv'],
            max_encoder_length=seq_len,
            max_prediction_length=1,
            time_varying_unknown_reals=self.input_features,
            scalers=None,
            target_normalizer=None,
            allow_missing_timesteps=False,
            predict_mode=False
        ).to_dataloader(**params)

    def get_dataloader2(self, data, batch_size=1024, seq_len=6, partition_id=0, only_benign=False) -> DataLoader:
        """
        Build a DataLoader from already loaded dataframe.
        """
        params = {
            'batch_size': batch_size,
            'num_workers': os.cpu_count(),
            'pin_memory': True,
            'prefetch_factor': 10000,
            'persistent_workers': True
        }
        df = self.preprocess_data(df=data, partition_id=partition_id)
        df = df.astype({'attack': int})
        if only_benign and 'attack' in df.columns:
            df = df[df['attack'] == 0]

        df['time_idx'] = df.groupby('imeisv')['_time'].cumcount()
        return TimeSeriesDataSet(
            data=df,
            time_idx='time_idx',
            target='attack',
            group_ids=['imeisv'],
            max_encoder_length=seq_len,
            max_prediction_length=1,
            time_varying_unknown_reals=self.input_features,
            scalers=None,
            target_normalizer=None,
            allow_missing_timesteps=False,
            predict_mode=False
        ).to_dataloader(**params)


if __name__ == '__main__':
    dp = DataProcessor()
    paths = PathsConf()
    dp.prepare_datasets(raw_dataset_path=paths.raw_dataset)
    exit()