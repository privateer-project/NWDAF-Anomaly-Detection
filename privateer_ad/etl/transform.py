import logging
from copy import deepcopy

import joblib
import pandas as pd

from datasets import Dataset
from pytorch_forecasting import TimeSeriesDataSet

from sklearn.model_selection import train_test_split
from flwr_datasets.partitioner import PathologicalPartitioner

from privateer_ad.config import DataConfig, PathConfig, MetadataConfig
from privateer_ad.etl.utils import get_dataset_path, check_existing_datasets


class DataProcessor:
    """Main class orchestrating data transform and loading."""

    def __init__(self, data_config: DataConfig | None = None):
        """
        Initialize DataProcessor.

        Args:
            data_config: Optional data configuration override
        """
        logging.info('Initializing DataProcessor...')

        # Extract feature configurations
        self.metadata_config = MetadataConfig()
        self.input_features = self.metadata_config.get_input_features()
        self.features_dtypes = self.metadata_config.get_features_dtypes()
        self.drop_features = self.metadata_config.get_drop_features()

        self.data_config = data_config or DataConfig()
        self.paths_config = PathConfig()
        self.scalers_dir = self.paths_config.scalers_dir
        # Initialize processing components

        self.scalers = self.get_dataset('train', only_benign=True).get_parameters()['scalers']

        for feature_name in self.input_features:
            if not self.scalers_dir.joinpath(feature_name + '.pkl').exists():
                self.paths_config.scalers_dir.mkdir(parents=True, exist_ok=True)

                joblib.dump(self.scalers[feature_name], self.scalers_dir.joinpath(feature_name + '.pkl'))

    def prepare_datasets(self, raw_dataset_path=None) -> None:
        """Prepare complete datasets from raw data."""
        raw_dataset_path = raw_dataset_path or self.paths_config.raw_dataset

        check_existing_datasets()

        raw_df = self.read_ds(raw_dataset_path)
        logging.info(f'Loaded data from {raw_dataset_path}')

        # Use configuration for train size
        datasets = self.split_data(raw_df)

        # Ensure processed directory exists
        self.paths_config.processed_dir.mkdir(parents=True, exist_ok=True)

        logging.info('Save datasets...')
        for k, df in datasets.items():
            save_path = get_dataset_path(k)
            df.to_csv(save_path, index=False)
            logging.info(f'{k} saved at {save_path}')

    def read_ds(self, path):
        dtypes = deepcopy(self.features_dtypes)
        dtypes.pop('_time', None)
        try:
            return pd.read_csv(get_dataset_path(path), dtype=dtypes, parse_dates=['_time'])
        except FileNotFoundError:
            raise FileNotFoundError(f'File {path} not found.')

    def split_data(self, df):
        train_dfs = []
        val_dfs = []
        test_dfs = []
        for device, device_info in self.metadata_config.devices.items():
            device_df = df.loc[df['imeisv'] == device_info.imeisv]
            logging.debug(f'Before: Device - {device}, attack samples - {len(device_df[device_df["attack"] == 1])}'
                          f' benign samples - {len(device_df[device_df["attack"] == 0])}')

            device_in_attacks = device_df['attack_number'].isin(device_info.in_attacks)
            logging.debug(f'Sum  - {sum(device_in_attacks)} Len - {len(device_in_attacks)}')
            device_df.loc[device_in_attacks, 'attack'] = 1
            device_df.loc[~device_in_attacks, 'attack'] = 0
            logging.debug(f'After: Device - {device}, attack samples - {len(device_df[device_df["attack"] == 1])} '
                          f'benign samples - {len(device_df[device_df["attack"] == 0])}')
            device_train_df, df_tmp = train_test_split(device_df,
                                                       train_size=self.data_config.train_size,
                                                       stratify=device_df['attack_number'],
                                                       random_state=42)
            device_val_df, device_test_df = train_test_split(df_tmp,
                                                             # Test is the rest percentage after getting train and val
                                                             test_size=1. - self.data_config.train_size - self.data_config.val_size,
                                                             stratify=df_tmp['attack_number'],
                                                             random_state=42)
            train_dfs.append(device_train_df)
            val_dfs.append(device_val_df)
            test_dfs.append(device_test_df)
        df_train = pd.concat(train_dfs)
        df_val = pd.concat(val_dfs)
        df_test = pd.concat(test_dfs)
        datasets = {'train': df_train, 'val': df_val, 'test': df_test}

        for k, df in datasets.items():
            logging.debug(f'Dataset {k} attack length: {len(df[df["attack"] == 1])} '
                          f'benign length: {len(df[df["attack"] == 0])} '
                          f'{k} shape: {df.shape}')
        return datasets

    def clean_data(self, df):
        if '_time' in df.columns:
            df = df[df['_time'] != '<NA>']
        df = df.drop(columns=self.drop_features, errors='ignore')
        df = df.drop_duplicates()
        df = df.dropna(axis='rows')
        df.reset_index(drop=True, inplace=True)
        return df

    def load_scalers(self):
        scalers = {}
        for feature_name in self.input_features:
            scaler_path = self.scalers_dir.joinpath(feature_name + '.pkl')
            if not self.scalers_dir.joinpath(feature_name + '.pkl').exists():
                raise  FileNotFoundError(f'Scaler for feature {feature_name} not found.')
            scalers[feature_name] = joblib.load(scaler_path)
        return scalers

    def scale_data(self, df) -> pd.DataFrame:
        scalers = self.load_scalers()
        for feature_name in self.input_features:
            df[feature_name] = scalers[feature_name].transform(df[feature_name].values)
        return df

    def get_partition(self, df: pd.DataFrame) -> pd.DataFrame:
        """Partition data."""
        if self.data_config.partition_id == -1:
            logging.info(f'No partitioning.')
            return df
        elif self.data_config.partition_id >= self.data_config.num_partitions:
            raise ValueError(f'partition_id ({self.data_config.partition_id}) '
                             f'is greater than num_partitions ({self.data_config.num_partitions})')

        # Set num_partitions as the number of unique values in partition_by column. E.g. partition_by='cell' will set num_partitions=3
        self.data_config.num_partitions = len(df[self.data_config.partition_by].unique())

        partitioner = PathologicalPartitioner(
            num_partitions=self.data_config.num_partitions,
            partition_by=self.data_config.partition_by,
            num_classes_per_partition=self.data_config.num_classes_per_partition,
            class_assignment_mode='deterministic',
            shuffle=False)

        partitioner.dataset = Dataset.from_pandas(df)
        logging.info(f'Get partition: {self.data_config.partition_id + 1}/{self.data_config.num_partitions}')
        partitioned_df = partitioner.load_partition(self.data_config.partition_id).to_pandas(batched=False)
        return partitioned_df[df.columns]

    def get_dataset(self, data_path, only_benign: bool = False) -> TimeSeriesDataSet:
        if self.data_config.num_workers <= 1:
            self.data_config.prefetch_factor = None
            self.data_config.persistent_workers = False
        logging.info(f'Get {data_path} dataloader with '
                     f'batch_size: {self.data_config.batch_size}, '
                     f'seq_len: {self.data_config.seq_len}, '
                     f'num_features: {len(self.input_features)}'
                     f'partition_id: {self.data_config.partition_id}, '
                     f'only_benign: {only_benign}')

        df = self.read_ds(data_path)
        df = self.get_partition(df)
        df = self.clean_data(df)
        df = df.sort_values(by=['_time']).reset_index(drop=True)
        if only_benign:
            if 'attack' in df.columns:
                df['attack'] = df['attack'].astype(int)
                df = df[df['attack'] == 0]
            else:
                logging.warning('Cannot get benign data. No column named `attack` in dataset.')

        # Create time index for each device
        df['time_idx'] = df.groupby('imeisv')['_time'].cumcount()
        if data_path == 'train':
            scalers = None
        else:
            scalers = self.scalers
        ts_ds = TimeSeriesDataSet(data=df,
                                  time_idx='time_idx',
                                  target='attack',
                                  group_ids=['imeisv'],
                                  max_encoder_length=self.data_config.seq_len,
                                  max_prediction_length=1,
                                  time_varying_known_reals=self.input_features,
                                  allow_missing_timesteps=False,
                                  predict_mode=False,
                                  scalers=scalers
                                  )
        return ts_ds

    def get_dataloader(self, path, only_benign: bool = False, train: bool = True):
        return self.get_dataset(path, only_benign).to_dataloader(
            train=train,
            batch_size=self.data_config.batch_size,
            num_workers=self.data_config.num_workers,
            pin_memory=self.data_config.pin_memory,
            prefetch_factor=self.data_config.prefetch_factor,
            persistent_workers=self.data_config.persistent_workers)


if __name__ == '__main__':
    dp = DataProcessor()
    # dp.prepare_datasets()
    # import torch
    dl = dp.get_dataloader('train', train=False)
    for i, sample in enumerate(dl):
        print(sample[0]['encoder_cont'][0, 0])
        break
    exit()