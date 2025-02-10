import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
import pandas as pd
from pandas import DataFrame
from pytorch_forecasting import TimeSeriesDataSet
from sklearn.model_selection import train_test_split

from config import MetaData, Paths, FeatureInfo, PartitionConfig
from data_handling.utils import partition_data
from data_handling.load import TSDatasetCreator

class DataCleaner:
    """Handles data cleaning and transform operations."""
    def __init__(self, features: Dict[str, FeatureInfo]):
        self.features = features

    def preprocess_data(self, df: DataFrame) -> DataFrame:
        """Clean and preprocess the input DataFrame."""
        if not isinstance(df, DataFrame):
            raise TypeError("Input must be a pandas DataFrame")
        if df.empty:
            raise ValueError("DataFrame cannot be empty")

        df = self._drop_columns(df)
        df = self._apply_datatypes(df)

        df.dropna(inplace=True)
        df.dropna(thresh = 2, axis = 'columns', inplace = True)

        df = self._process_features(df)
        return df

    def _apply_datatypes(self, df: DataFrame) -> DataFrame:
        return df.astype({feature: self.features[feature].dtype for feature in df.columns if feature in self.features})

    def _drop_columns(self, df: DataFrame) -> DataFrame:
        drop_features = [name for name, info in self.features.items() if info.drop and name in df.columns]
        df.drop(columns=drop_features, axis='columns', inplace=True)
        return df

    def _process_features(self, df: DataFrame) -> DataFrame:
        def robust_normalize_ewm(x):
            ewm_mean = x.ewm(alpha=0.1).mean()
            ewm_std = x.ewm(alpha=0.1).std()
            # Handle cases where std is 0 or close to 0
            mask = ewm_std < 1e-8
            result = (x - ewm_mean) / ewm_std.where(~mask, 1.0)
            # If both mean and std are 0, return 0
            result = result.where(~(mask & (ewm_mean == 0)), 0)
            return result

        features = [feature for feature, params in self.features.items() if feature in df.columns and params.input]

        df[features] = df.groupby('imeisv')[features].transform(robust_normalize_ewm).bfill()
        for feature in features:
            for proc in self.features[feature].process:
                if proc == 'delta':
                    df[feature] = df.groupby('imeisv')[feature].diff().fillna(0)
        df.drop_duplicates(inplace=True)
        return df


class DataProcessor:
    """Main class orchestrating data transform and loading."""

    def __init__(self, metadata: MetaData, paths: Paths):
        self.features = metadata.features
        self.attacks = metadata.attacks
        self.devices = metadata.devices
        self.paths = paths
        self._ensure_directories()
        self.raw_dataset = self.paths.raw_dataset
        self.cleaner = DataCleaner(self.features)
        self.splitter = DataSplitter(metadata)
        self.ts_creator = TSDatasetCreator(self.features, paths)

    def prepare_datasets(self, train_ratio: float = 0.8, force: bool = False) -> None:
        """Prepare complete datasets from raw data."""
        self._check_existing_files(force)
        df = pd.read_csv(self.raw_dataset)
        processed_df = self.cleaner.preprocess_data(df)
        train_df, val_df, test_df = self.splitter.split(processed_df, train_ratio)

        self._save_datasets(train_df, val_df, test_df)
        self._create_scalers(train_df)

    def _ensure_directories(self):
        """Create necessary directories if they don't exist."""
        for path in [self.paths.scalers, self.paths.processed]:
            path.mkdir(parents=True, exist_ok=True)

    def _get_scaler_path(self, scaler_name: str) -> str:
        """Get full path for a scaler file."""
        return str(self.paths.scalers.joinpath(f"{scaler_name}.scaler"))

    def _get_data_path(self, dataset_name: str) -> str:
        """Get full path for a dataset file."""
        return str(self.paths.processed.joinpath(f"{dataset_name}.csv"))

    def _check_existing_files(self, force: bool):
        for mode in ['train', 'val', 'test']:
            path = self._get_data_path(mode)
            if os.path.exists(path) and not force:
                raise FileExistsError(
                    f'File {path} exists. Set force=True to overwrite.'
                )

    def _save_datasets(self, train_df: DataFrame, val_df: DataFrame, test_df: DataFrame):
        train_df.to_csv(self._get_data_path('train'), index=True)
        val_df.to_csv(self._get_data_path('val'), index=True)
        test_df.to_csv(self._get_data_path('test'), index=True)

    def _create_scalers(self, train_df: DataFrame):
        ds = self.ts_creator.create_dataset(train_df, window_size=1, use_existing_scalers=False)
        for name, scaler in ds.get_parameters()['scalers'].items():
            joblib.dump(scaler, self._get_scaler_path(name))

    def get_dataset(self, path: str, partition_config: Optional[PartitionConfig] = None,
                    window_size: int = 12) -> TimeSeriesDataSet:
        """Load and process a dataset from a file path."""
        if path in ['train', 'val', 'test']:
            path = self._get_data_path(path)

        try:
            df = pd.read_csv(path)
        except FileNotFoundError:
            raise FileNotFoundError(f"File {path} not found.")

        df = self.cleaner.preprocess_data(df)
        if partition_config:
            df = partition_data(df, partition_config)

        return self.ts_creator.create_dataset(df, window_size)

    def process_new_data(self, input_path: str, window_size: int, use_existing_scalers: bool = True) -> DataFrame:
        """Process a new CSV file using existing transform pipeline and scalers.

        Args:
            input_path: Path to input CSV file
            window_size: window_size
            use_existing_scalers: If True, use existing scalers from training.
                                If False, create new ones from provided data.

        Raises:
            FileNotFoundError: If input file not found or no scalers exist when use_existing_scalers=True
        """
        try:
            df = pd.read_csv(input_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Input file {input_path} not found.")
        processed_df = self.cleaner.preprocess_data(df)
        scalers_exist = any(Path(self._get_scaler_path(feat)).exists()
                            for feat in self.features)
        if use_existing_scalers and not scalers_exist:
            raise FileNotFoundError(
                "No existing scalers found. Either train the model first or set use_existing_scalers=False"
            )
        if not use_existing_scalers:
            print("Creating new scalers from provided data.")
            # Reset any existing scalers in TimeSeriesDatasetCreator
        dataset = self.ts_creator.create_dataset(processed_df, window_size=window_size)
        scaled_data = dataset.data.copy()
        return scaled_data


class DataSplitter:
    """Handles dataset splitting operations."""
    def __init__(self, metadata: MetaData):
        self.attacks = metadata.attacks
        self.devices = metadata.devices

    def split(self, df: DataFrame, train_size: float = 0.8) -> Tuple[DataFrame, ...]:
        """Split data into train, validation, and test sets."""
        attack_dfs = []
        normal_on_attack_dfs = []
        for attack, params in self.attacks.items():
            devices = [self.devices[str(device)].imeisv for device in params.devices]
            participating_devices_df = df.loc[df['imeisv'].isin(devices)]
            not_participating_devices_df = df.loc[~df['imeisv'].isin(devices)]
            attack_number_df = participating_devices_df.loc[participating_devices_df['attack_number'] == str(attack)]
            ben_during_attack_df = not_participating_devices_df.loc[not_participating_devices_df['attack_number'] == str(attack)]

            attack_dfs.append(attack_number_df)
            normal_on_attack_dfs.append(ben_during_attack_df)
        attack_df = pd.concat(attack_dfs)
        normal_on_attack_df = pd.concat(normal_on_attack_dfs)
        normal_on_attack_df['attack'] = 0
        normal_on_attack_df['attack_number'] = '0'

        benign_df = df[df['attack_number'] == '0']
        normal_df = pd.concat([benign_df, normal_on_attack_df])

        train_df, val_df, test_df = self._split_normal_data(normal_df, train_size)
        test_df = pd.concat([test_df, attack_df])
        df.drop_duplicates(inplace=True)
        train_df.drop_duplicates(inplace=True)
        val_df.drop_duplicates(inplace=True)
        test_df.drop_duplicates(inplace=True)
        return train_df, val_df, test_df

    @staticmethod
    def _split_normal_data(normal_data: DataFrame, train_size: float) -> Tuple[DataFrame, ...]:
        train_df = DataFrame()
        val_df = DataFrame()
        test_df = DataFrame()

        for _, normal_per_device in normal_data.groupby('imeisv'):
            train_dev, val_test_dev = train_test_split(
                normal_per_device,
                shuffle=False,
                train_size=train_size,
                random_state=42
            )
            val_dev, test_dev = train_test_split(
                val_test_dev,
                shuffle=False,
                test_size=0.5,
                random_state=42)

            train_df = pd.concat([train_df, train_dev])
            val_df = pd.concat([val_df, val_dev])
            test_df = pd.concat([test_df, test_dev])
        return train_df, val_df, test_df
