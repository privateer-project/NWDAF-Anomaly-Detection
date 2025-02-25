import os
import joblib

import pandas as pd
from pandas import DataFrame
from datasets import Dataset
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from flwr_datasets.partitioner import PathologicalPartitioner

from src.config import MetaData, PathsConf, logger
from src.data_utils.utils import check_existing_datasets, get_dataset_path


class DataProcessor:
    """Main class orchestrating data transform and loading."""
    def __init__(self, metadata, paths):
        self.metadata = metadata
        self.paths = paths
        self.scaler_path = os.path.join(paths.scalers, 'scaler.pkl')
        self.pca_path = os.path.join(paths.scalers, 'pca.pkl')

    def clean_data(self, df):
        df = df.drop(columns=self.metadata.get_drop_features())
        df = df.drop_duplicates()
        df = df.dropna(axis='rows')
        df.reset_index(drop=True, inplace=True)
        return df

    def split_data(self, df, val_size=0.1, test_size=0.1):
        train_dfs = []
        val_dfs = []
        test_dfs = []

        for device, device_info in self.metadata.devices.items():
            device_df: DataFrame = df.loc[df['imeisv'] == int(device_info.imeisv)]
            logger.info(f'Device: {device}, attack length: {len(device_df[device_df['attack'] == 1])}'
                  f' benign length: {len(device_df[device_df['attack'] == 0])}')
            device_df.loc[device_df['attack_number'].isin(device_info.in_attacks), 'attack'] = 1
            device_df.loc[~device_df['attack_number'].isin(device_info.in_attacks), 'attack'] = 0
            logger.info(f'Device: {device}, attack length: {len(device_df[device_df['attack'] == 1])}'
                  f' benign length: {len(device_df[device_df['attack'] == 0])}')
            df_train, df_tmp = train_test_split(device_df, test_size=val_size + test_size,
                                                stratify=device_df['attack_number'], random_state=42)
            df_val, df_test = train_test_split( df_tmp, test_size=val_size / (test_size + val_size),
                                                stratify=df_tmp['attack_number'], random_state=42)
            train_dfs.append(df_train)
            val_dfs.append(df_val)
            test_dfs.append(df_test)
        df_train = pd.concat(train_dfs)
        df_val = pd.concat(val_dfs)
        df_test = pd.concat(test_dfs)
        for i, df in enumerate([df_train, df_val, df_test]):
            logger.info(f'Dataset {i} attack length: {len(df[df['attack'] == 1])}'
                  f' benign length: {len(df[df['attack'] == 0])}')
        return {'train': df_train, 'val': df_val, 'test': df_test}

    def setup_scaler(self, df):
        """
        Scaling and normalizing numeric values, imputing missing values, clipping outliers,
         and adjusting values that have skewed distributions.
        """
        scaler = StandardScaler()
        benign_df = df[df['attack_number'] == 0].copy()
        scaler.fit(benign_df[self.metadata.get_input_features()])
        os.makedirs(os.path.dirname(self.scaler_path), exist_ok=True)
        joblib.dump(scaler, self.scaler_path)

    def scale_features(self, df):
        """Scale numeric features using standardization (mean=0, std=1)."""
        scaler = joblib.load(self.scaler_path)
        scaled_df = df.copy()
        scaled_df.loc[:, self.metadata.get_input_features()] = scaler.transform(scaled_df[self.metadata.get_input_features()])
        return scaled_df

    def setup_pca(self, df, n_components=10):
        """Reduce feature dimensions using PCA."""
        pca = PCA(n_components=n_components)
        benign_df = df[df['attack_number'] == 0].copy()
        pca.fit(benign_df[self.metadata.get_input_features()])
        os.makedirs(os.path.dirname(self.pca_path), exist_ok=True)
        joblib.dump(pca, self.pca_path)
        return pca

    def get_pca(self):
        return joblib.load(self.pca_path)

    def apply_pca(self, df):
        pca = self.get_pca()
        pca_features = pca.transform(df[self.metadata.get_input_features()])
        pca_df = pd.DataFrame(pca_features, columns=pca.get_feature_names_out())
        df = df.drop(columns=self.metadata.get_input_features())
        df = df.assign(**pca_df.to_dict(orient='list'))
        return df

    def get_partition(self, df: DataFrame, partition_id, num_partitions, num_classes_per_partition) -> DataFrame:
        """Partition data based on provided configuration."""
        partitioner = PathologicalPartitioner(
            num_partitions=num_partitions,
            num_classes_per_partition=num_classes_per_partition,
            partition_by='imeisv',
            class_assignment_mode='first-deterministic'
        )
        partitioner.dataset = Dataset.from_pandas(df)
        partitioned_df = partitioner.load_partition(partition_id).to_pandas(batched=False)
        return partitioned_df[df.columns]

    def process_data(self, df: DataFrame) -> DataFrame:
        df = self.clean_data(df)
        df = self.scale_features(df)
        df = self.apply_pca(df).sort_values('_time')
        df = df.sort_values('_time').reset_index(drop=True)
        return df

    def prepare_datasets(self, val_size: float = 0.1, test_size: float = 0.1) -> None:
        """Prepare complete datasets from raw data."""
        check_existing_datasets()
        raw_df = pd.read_csv(self.paths.raw_dataset)
        logger.info(f'Raw shape: {raw_df.shape}')
        cleaned_df = self.clean_data(raw_df)
        datasets = self.split_data(cleaned_df, val_size=val_size, test_size=test_size)
        [logger.info(f'{key} shape: {df.shape}') for key, df in datasets.items()]

        self.setup_scaler(datasets['train'])
        datasets['train'] = self.scale_features(datasets['train'])
        datasets['val'] = self.scale_features(datasets['val'])
        datasets['test'] = self.scale_features(datasets['test'])
        [logger.info(f'Scaled {key} shape: {df.shape}') for key, df in datasets.items()]

        self.setup_pca(datasets['train'], n_components=7)
        datasets['train'] = self.apply_pca(datasets['train']).sort_values('_time')
        datasets['val'] = self.apply_pca(datasets['val']).sort_values('_time')
        datasets['test'] = self.apply_pca(datasets['test']).sort_values('_time')
        [logger.info(f'PCA {key} shape: {df.shape}') for key, df in datasets.items()]
        os.makedirs(self.paths.processed, exist_ok=True)
        logger.info('Save datasets...')
        for key, df in datasets.items():
            df.to_csv(get_dataset_path(key), index=False)
        logger.info(f'Datasets saved at {self.paths.processed}')

if __name__ == '__main__':
    paths = PathsConf()
    metadata = MetaData()
    dp = DataProcessor(metadata, paths)
    dp.prepare_datasets()
