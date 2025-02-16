from typing import Dict

import pandas as pd
from pandas import DataFrame

from src.config import MetaData, ProjectPaths
from src.data_handling.utils import check_existing_datasets, split_data, get_dataset_path


class DataProcessor:
    """Main class orchestrating data transform and loading."""
    def __init__(self, metadata: MetaData, paths: ProjectPaths):
        self.paths = paths
        self.devices = metadata.devices
        self.attacks = metadata.attacks
        self.features = metadata.features

    def load_dataset(self, path):
        df = pd.read_csv(path,
                         dtype={feature: feature_info.dtype for feature, feature_info in self.features.items()})
        return df
    def prepare_datasets(self, train_size: float = 0.8, force: bool = False) -> None:
        """Prepare complete datasets from raw data."""
        check_existing_datasets(force)
        # dtypes = {feature: feature_info.dtype for feature, feature_info in self.features.items()}
        self.paths.processed.mkdir(parents=True, exist_ok=True)
        df = self.load_dataset(self.paths.raw_dataset)

        processed_df = self.preprocess_dataset(df)
        dfs = self.split_dataset(processed_df, train_size)
        [df.to_csv(get_dataset_path(name), index=False) for name, df in dfs.items()]

    def preprocess_dataset(self, df: DataFrame) -> DataFrame:
        """Clean and preprocess the input DataFrame."""
        if not isinstance(df, DataFrame):
            raise TypeError("Input must be a pandas DataFrame")
        if df.empty:
            raise ValueError("DataFrame cannot be empty")
        df = self._clean_dataset(df)
        df = self._process_features(df)
        return df

    def split_dataset(self, df: DataFrame, train_size: float = 0.8) -> Dict[str, DataFrame]:
        """Split data into train, validation, and test sets."""
        devices_benign = []
        devices_malicious = []
        for device, device_info in self.devices.items():
            device_df: DataFrame = df.loc[df['imeisv'] == device_info.imeisv]
            device_df.reset_index(drop=True, inplace=True)
            device_attack_df = device_df.loc[device_df['attack_number'].isin(device_info.in_attacks)]
            device_attack_df.loc[:, 'attack'] = 1
            # Split benign - malicious per device.
            # If participated in an attack, no matter if performed or not an attack,
            # its behaviour is considered malicious.
            device_benign_df = device_df[~device_df.index.isin(device_attack_df.index)]
            device_benign_df.loc[:, 'attack'] = 0
            devices_benign.append(device_benign_df)
            devices_malicious.append(device_attack_df)

        benign_df = pd.concat(devices_benign)
        malicious_df = pd.concat(devices_malicious)

        benign_train_df, benign_val_df, benign_test_df = split_data(benign_df, train_size)

        train_df = benign_train_df
        val_df = benign_val_df
        test_df = pd.concat([benign_test_df, malicious_df])

        train_df = train_df.sort_values('_time').reset_index(drop=True)
        val_df = val_df.sort_values('_time').reset_index(drop=True)
        test_df = test_df.sort_values('_time').reset_index(drop=True)

        return {'train': train_df, 'val': val_df, 'test': test_df}

    def _clean_dataset(self, df: DataFrame) -> DataFrame:
        valid_features = [feature for feature in df.columns if feature in self.features]

        to_drop_features = [feature for feature in valid_features if self.features[feature].drop == True]
        df = df.drop(columns=to_drop_features, axis='columns')
        df.dropna(inplace=True)
        df.drop_duplicates(inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    def _process_features(self, df: DataFrame) -> DataFrame:
        procs = {feature: self.features[feature].process for feature in df.columns}
        for feature, procs in procs.items():
            for proc in procs:
                if proc == 'delta':
                    df[feature] = df.groupby('imeisv')[feature].diff().fillna(0)
        df.drop_duplicates(inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

if __name__ == '__main__':
    dp = DataProcessor(MetaData(), ProjectPaths())
    dp.prepare_datasets(force=True)
