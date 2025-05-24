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

from privateer_ad import logger
from privateer_ad.config import get_data_config, get_paths, get_metadata
from .utils import get_dataset_path, check_existing_datasets


class DataProcessor:
    """Main class orchestrating data transform and loading."""

    def __init__(self, partition=False, data_config=None, paths_config=None, metadata_config=None):
        """
        Initialize DataProcessor.

        Args:
            partition: Whether to enable data partitioning
            data_config: Optional data configuration override
            paths_config: Optional paths configuration override
            metadata_config: Optional metadata configuration override
        """
        logger.info('Initializing DataProcessor...')
        self.data_config = data_config or get_data_config()
        self.paths_config = paths_config or get_paths()
        self.metadata_config = metadata_config or get_metadata()

        self.partition = partition

        # Setup paths
        self.scaler_path = self.paths_config.scalers_dir / 'scaler.pkl'

        # Initialize processing components
        self.scaler = None

        # Extract feature configurations
        self.input_features = self.metadata_config.get_input_features()
        self.drop_features = self.metadata_config.get_drop_features()

    def prepare_datasets(self, raw_dataset_path=None) -> None:
        """Prepare complete datasets from raw data."""
        if raw_dataset_path is None:
            raw_dataset_path = self.paths_config.raw_dataset

        check_existing_datasets()

        raw_df = self._read_csv(raw_dataset_path)
        logger.info(f'Loaded data from {raw_dataset_path}')

        # Use configuration for train size
        datasets = self.split_data(raw_df, train_size=self.data_config.train_size)

        self.setup_scaler(self.clean_data(datasets['train']))

        # Ensure processed directory exists
        self.paths_config.processed_dir.mkdir(parents=True, exist_ok=True)

        logger.info('Save datasets...')
        for k, df in datasets.items():
            save_path = get_dataset_path(k)
            processed_df = self.preprocess_data(df)
            logger.warning(f'Save size {k}: {len(processed_df)}')
            processed_df.to_csv(save_path, index=False)
            logger.info(f'{k} saved at {save_path}')

    def split_data(self, df, train_size=0.8):
        train_dfs = []
        val_dfs = []
        test_dfs = []
        for device, device_info in self.metadata_config.devices.items():
            device_df = df.loc[df['imeisv'] == device_info.imeisv]
            logger.debug(f'Before: Device - {device}, attack samples - {len(device_df[device_df["attack"] == 1])}'
                        f' benign samples - {len(device_df[device_df["attack"] == 0])}')

            device_in_attacks = device_df['attack_number'].isin(device_info.in_attacks)
            logger.debug(f'Sum  - {sum(device_in_attacks)} Len - {len(device_in_attacks)}')
            device_df.loc[device_in_attacks, 'attack'] = 1
            device_df.loc[~device_in_attacks, 'attack'] = 0
            logger.debug(f'After: Device - {device}, attack samples - {len(device_df[device_df["attack"] == 1])} '
                        f'benign samples - {len(device_df[device_df["attack"] == 0])}')
            device_train_df, df_tmp = train_test_split(device_df,
                                                       train_size=train_size,
                                                       stratify=device_df['attack_number'],
                                                       random_state=42)
            device_val_df, device_test_df = train_test_split(df_tmp,
                                                             test_size=0.5,
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
            logger.debug(f'Dataset {k} attack length: {len(df[df["attack"] == 1])} '
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

    def preprocess_data(self, df, partition_id=0) -> DataFrame:
        if self.partition:
            df = self.get_partition(df, partition_id=partition_id)
        df = self.clean_data(df)
        df = self.apply_scale(df)
        df = df.sort_values(by=['_time']).reset_index(drop=True)
        return df

    def get_partition(self, df: DataFrame, partition_id=0) -> DataFrame:
        """Partition data."""
        num_partitions = df['cell'].unique().shape[0]
        logger.info(f'Get partition: {partition_id + 1}/{num_partitions}')

        if partition_id < 0:
            raise ValueError(f'partition_id ({partition_id}) < 0')
        elif partition_id >= num_partitions:
            raise ValueError(f'partition_id ({partition_id}) is greater than num_partitions ({num_partitions})')
        partitioner = PathologicalPartitioner(
            num_partitions=num_partitions,
            num_classes_per_partition=1,
            partition_by='cell',
            class_assignment_mode='deterministic')
        partitioner.dataset = Dataset.from_pandas(df)
        partitioned_df = partitioner.load_partition(partition_id).to_pandas(batched=False)
        return partitioned_df[df.columns]

    def setup_scaler(self, df):
        benign_df = df[df['attack'] == 0].copy()
        os.makedirs(os.path.dirname(self.scaler_path), exist_ok=True)

        scaler = StandardScaler()
        scaler.fit(benign_df[self.input_features])
        joblib.dump(scaler, self.scaler_path)

    def load_scaler(self):
        if self.scaler is None:
            self.scaler = joblib.load(self.scaler_path)
        return self.scaler

    def apply_scale(self, df):
        self.load_scaler()
        transformed = self.scaler.transform(df[self.input_features])
        df.loc[:, self.input_features] = transformed
        return df

    def _read_csv(self, path):
        path = get_dataset_path(path)
        dtypes = deepcopy(self.metadata_config.get_features_dtypes())
        dtypes.pop('_time', None)
        try:
            return pd.read_csv(path,
                               dtype=self.metadata_config.get_features_dtypes(),
                               parse_dates=['_time'])
        except FileNotFoundError:
            raise FileNotFoundError(f'File {path} not found.')

    def get_dataloader(self,
                       path,
                       batch_size=None,
                       seq_len=None,
                       partition_id=0,
                       only_benign=None) -> DataLoader:
        logger.info(f'Get {path} dataloader with '
                    f'batch_size={batch_size}, '
                    f'seq_len={seq_len}, '
                    f'partition_id={partition_id},'
                    f' only_benign={only_benign}')
        # Use configuration defaults if not provided
        if batch_size is None:
            batch_size = self.data_config.batch_size if hasattr(self.data_config, 'batch_size') else 1024
        if only_benign is None:
            only_benign = self.data_config.only_benign_for_training

        dataloader_params = {
            'train': path == 'train',
            'batch_size': batch_size,
            'num_workers': self.data_config.num_workers,
            'pin_memory': self.data_config.pin_memory,
            'prefetch_factor': 10000,
            'persistent_workers': True
        }

        logger.info(f'Loading {path} dataset...')
        df = self.preprocess_data(
            df=self._read_csv(path),
            partition_id=partition_id
        )
        df = df.astype({'attack': int})

        if only_benign:
            if 'attack' in df.columns:
                df = df[df['attack'] == 0]
            else:
                logger.warning('Cannot get benign data. No column named `attack` in dataset.')

        # Create time index for each device
        df['time_idx'] = df.groupby('imeisv')['_time'].cumcount()

        dataloader = TimeSeriesDataSet(
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
            predict_mode=False,
        ).to_dataloader(**dataloader_params)

        logger.info(f'Finished loading {path} dataset.')
        return dataloader


if __name__ == '__main__':
    dp = DataProcessor()
    dp.prepare_datasets()
    # for i, sample in enumerate(dl):
    #     print("i[0]['encoder_cont']", sample[0]['encoder_cont'])
    #     print("i[0]['encoder_cont']", sample[0]['encoder_cont'].shape)
    #     if i > 2:
    #         break
    exit()
