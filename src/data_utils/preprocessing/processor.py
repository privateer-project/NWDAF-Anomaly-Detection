import os

from pathlib import Path
from typing import Optional
import pandas as pd
from pandas import DataFrame
import joblib

from data_utils.data_loading.partitioner import PartitionConfig
from src.config import Paths, MetaData
from base import BasePreprocessor
from cleaner import DataCleaner
from splitter import  DataSplitter
from ..data_loading import TimeSeriesDatasetCreator, DataPartitioner
from pytorch_forecasting import TimeSeriesDataSet


class DataProcessor(BasePreprocessor):
    """Main class orchestrating data preprocessing and loading."""

    def __init__(self, metadata: MetaData, paths: Paths):
        super().__init__(metadata, paths)
        self.raw_dataset = self.paths.raw_dataset
        self.cleaner = DataCleaner(self.features)
        self.splitter = DataSplitter(metadata)
        self.partitioner = DataPartitioner()
        self.ts_creator = TimeSeriesDatasetCreator(self.features, paths)

    def prepare_datasets(self, train_ratio: float = 0.8, force: bool = False) -> None:
        """Prepare complete datasets from raw data."""
        self._check_existing_files(force)
        df = pd.read_csv(self.raw_dataset)
        processed_df = self.cleaner.preprocess_data(df)
        train_df, val_df, test_df = self.splitter.split(processed_df, train_ratio)

        self._save_datasets(train_df, val_df, test_df)
        self._create_scalers(train_df)

    def _check_existing_files(self, force: bool):
        for mode in ['train', 'val', 'test']:
            path = self.get_data_path(mode)
            if os.path.exists(path) and not force:
                raise FileExistsError(
                    f'File {path} exists. Set force=True to overwrite.'
                )

    def _save_datasets(self, train_df: DataFrame, val_df: DataFrame, test_df: DataFrame):
        train_df.to_csv(self.get_data_path('train'), index=True)
        val_df.to_csv(self.get_data_path('val'), index=True)
        test_df.to_csv(self.get_data_path('test'), index=True)

    def _create_scalers(self, train_df: DataFrame):
        ds = self.ts_creator.create_dataset(train_df, window_size=1, use_existing_scalers=False)
        for name, scaler in ds.get_parameters()['scalers'].items():
            joblib.dump(scaler, self.get_scaler_path(name))

    def get_dataset(self, path: str, partition_config: Optional[PartitionConfig] = None, window_size: int = 12) -> TimeSeriesDataSet:
        """Load and process a dataset from a file path."""
        if path in ['train', 'val', 'test']:
            path = self.get_data_path(path)

        try:
            df = pd.read_csv(path)
        except FileNotFoundError:
            raise FileNotFoundError(f"File {path} not found.")

        df = self.cleaner.preprocess_data(df)
        if partition_config:
            df = self.partitioner.partition_data(df, partition_config)

        return self.ts_creator.create_dataset(df, window_size)

    def process_new_data(self, input_path: str, window_size: int, use_existing_scalers: bool = True) -> DataFrame:
        """Process a new CSV file using existing preprocessing pipeline and scalers.

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
        scalers_exist = any(Path(self.get_scaler_path(feat)).exists()
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

if __name__ == '__main__':
    DataProcessor(MetaData(), Paths()).prepare_datasets(force=True)
