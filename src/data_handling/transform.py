from typing import Dict

import pandas as pd
from pandas import DataFrame

from src.config import MetaData, ProjectPaths
from src.data_handling.utils import check_existing_datasets, save_dataset, split_normal_data


class DataProcessor:
    """Main class orchestrating data transform and loading."""
    def __init__(self, metadata: MetaData, paths: ProjectPaths):
        self.paths = paths
        self.devices = metadata.devices
        self.attacks = metadata.attacks
        self.features = metadata.features

    def prepare_data(self, train_ratio: float = 0.8, force: bool = False) -> None:
        """Prepare complete datasets from raw data."""
        check_existing_datasets(force)
        self.paths.processed.mkdir(parents=True, exist_ok=True)
        df = pd.read_csv(self.paths.raw_dataset)
        processed_df = self.preprocess_data(df)
        dfs = self.split_data(processed_df, train_ratio)
        [save_dataset(name, df) for name, df in dfs.items()]

    def preprocess_data(self, df: DataFrame) -> DataFrame:
        """Clean and preprocess the input DataFrame."""
        if not isinstance(df, DataFrame):
            raise TypeError("Input must be a pandas DataFrame")
        if df.empty:
            raise ValueError("DataFrame cannot be empty")
        df = self._clean_dataset(df)
        df = self._apply_datatypes(df)
        df = self._process_features(df)
        return df

    def split_data(self, df: DataFrame, train_size: float = 0.8) -> Dict[str, DataFrame]:
        """Split data into train, validation, and test sets."""
        attack_dfs = []
        for attack, params in self.attacks.items():
            attack_number_df = df.loc[df['attack_number'] == str(attack)]
            devices = [self.devices[str(device)].imeisv for device in params.devices]
            participating_devices_df = attack_number_df.loc[attack_number_df['imeisv'].isin(devices)]
            attack_dfs.append(participating_devices_df)

        attack_df = pd.concat(attack_dfs)
        benign_df = df[df['attack_number'] == '0']

        train_df, val_df, test_df = split_normal_data(benign_df, train_size)
        test_df = pd.concat([test_df, attack_df])
        return {'train': train_df, 'val': val_df, 'test': test_df}

    def _clean_dataset(self, df: DataFrame) -> DataFrame:
        drop_features = [name for name, info in self.features.items() if info.drop and name in df.columns]
        df.drop(columns=drop_features, axis='columns', inplace=True)
        df.dropna(inplace=True)
        df.drop_duplicates(inplace=True)
        return df

    def _apply_datatypes(self, df: DataFrame) -> DataFrame:
        return df.astype({feature: self.features[feature].dtype for feature in df.columns if feature in self.features})

    def _process_features(self, df: DataFrame) -> DataFrame:
        features = [feature for feature, params in self.features.items() if feature in df.columns and params.input]
        for feature in features:
            for proc in self.features[feature].process:
                if proc == 'delta':
                    df[feature] = df.groupby('imeisv')[feature].diff().fillna(0)
        df.drop_duplicates(inplace=True)
        return df

if __name__ == '__main__':
    dp = DataProcessor(MetaData(), ProjectPaths())
    dp.prepare_data()