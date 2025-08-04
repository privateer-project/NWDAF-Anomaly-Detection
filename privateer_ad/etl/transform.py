import logging
from copy import deepcopy
from pathlib import Path

import joblib
import pandas as pd

from datasets import Dataset
from pytorch_forecasting import TimeSeriesDataSet

from sklearn.model_selection import train_test_split
from flwr_datasets.partitioner import PathologicalPartitioner
from torch.utils.data import DataLoader

from privateer_ad.config import DataConfig, PathConfig, MetadataConfig
from privateer_ad.etl.utils import get_dataset_path, check_existing_datasets


class DataProcessor:
    """
    Central data processing engine for the PRIVATEER anomaly detection framework.

    This class orchestrates the complete data pipeline from raw network traffic data
    through preprocessing, scaling, partitioning, and time series dataset preparation.
    The implementation follows privacy-preserving principles by establishing scalers
    exclusively from benign training samples while supporting federated learning
    scenarios through configurable data partitioning strategies.

    The processor handles temporal aggregation of network metrics, maintains proper
    train-validation-test splits with stratification on attack patterns, and provides
    seamless integration with PyTorch Lightning forecasting frameworks.

    Attributes:
        metadata_config (MetadataConfig): Configuration containing feature definitions
                                        and device specifications
        input_features (list): Names of numerical features used for model training
        features_dtypes (dict): Data type specifications for each feature column
        drop_features (list): Feature names to exclude from processing pipeline
        data_config (DataConfig): Processing parameters including batch sizes and
                                sequence lengths
        paths_config (PathConfig): File system path configuration for datasets
                                 and artifacts
        scalers_dir (Path): Directory path for persisting feature scaling objects
        scalers (dict): Feature-wise scaling transformations fitted on benign data
    """
    def __init__(self, data_config: DataConfig | None = None):
        """
        Initialize the data processing pipeline with configuration parameters.

        Establishes the processing environment by loading metadata configurations,
        setting up feature specifications, and preparing path structures for
        dataset management. The initialization process prepares all necessary components
         for subsequent processing operations.

        Args:
            data_config (DataConfig, optional): Custom data processing configuration.
                                              If None, uses default configuration
                                              with standard batch sizes and
                                              sequence parameters.
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
        self.scalers = None
        self.load_scalers()
        if not self.scalers:
            self.setup_scalers()

    def read_ds(self, path: str) -> pd.DataFrame:
        """
        Load dataset from specified path with proper data type handling.

        This method handles CSV file loading with explicit data type specifications
        to ensure consistent data representation across processing stages.

        Args:
            path (str): Dataset identifier or file path for loading. Can be
                       a dataset name that gets resolved from a preprocessed path
                       or a direct file path.

        Returns:
            pd.DataFrame: Loaded dataset with properly typed columns and
                         parsed temporal indexing.
        """
        dtypes = deepcopy(self.features_dtypes)
        dtypes.pop('_time', None)
        df = pd.read_csv(get_dataset_path(path), dtype=dtypes, parse_dates=['_time'])
        return df

    def get_partition(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract federated learning partition based on pathological data distribution.

        This method implements data partitioning strategies essential for federated
        learning scenarios where data must be distributed across multiple participants
        with realistic non-IID characteristics. The pathological partitioning approach
        creates realistic data heterogeneity by limiting the number of classes available
        to each partition, simulating real-world federated scenarios.

        The partitioning strategy uses specified columns (typically geographical or
        organizational identifiers) to create meaningful data distributions that
        reflect practical federated deployment scenarios. When partition_id is -1,
        the method returns the complete dataset for centralized training approaches.

        Args:
            df (pd.DataFrame): Complete dataset for partitioning with appropriate
                             grouping columns and class labels for distribution
                             control.

        Returns:
            pd.DataFrame: Subset of the original dataset corresponding to the
                         specified partition with maintained column structure
                         and realistic data distribution characteristics.

        Raises:
            ValueError: When partition_id exceeds the available number of partitions
                       or partitioning configuration is invalid.
        """

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

    def split_data(self, df: pd.DataFrame, seed: int = 42) -> dict[str, pd.DataFrame]:
        """Splits dataset into train/validation/test sets with stratified sampling per device.

        This method performs a stratified split of the input dataset, ensuring that each device's
        data is split proportionally based on attack types while maintaining the original time
        ordering within each split. The splitting is done per device to ensure balanced
        representation across all devices in each dataset split.

        Args:
            df (pd.DataFrame): Input DataFrame containing network traffic data with columns:
                - 'imeisv': Device identifier for grouping
                - 'attack': Binary attack indicator (0=benign, 1=attack)
                - 'attack_number': Attack type identifier for stratification
                - '_time': Timestamp for chronological ordering
            seed (int, optional): Random seed for reproducible splits. Defaults to 42.

        Returns:
            dict: Dictionary containing three DataFrames with keys:
                - 'train': Training dataset
                - 'val': Validation dataset
                - 'test': Test dataset
                Each DataFrame is sorted by '_time' and has reset indices.

        Note:
            - Uses stratified sampling based on 'attack_number' to maintain attack type distribution
            - Split ratios are determined by self.data_config.train_size and self.data_config.val_size
            - Test size is calculated as the remainder (1 - train_size - val_size)
            - Logs debug information about sample counts before and after splitting
            - All splits use the same random seed for consistency across train/val/test creation

        Example:
            >>> processor = DataProcessor()
            >>> datasets = processor.split_data(df)
            >>> print(datasets['train'].shape, datasets['val'].shape, datasets['test'].shape)
            (8000, 50) (1000, 50) (1000, 50)
        """
        train_dfs = []
        val_dfs = []
        test_dfs = []
        for device, device_info in self.metadata_config.devices.items():
            device_df = df.loc[df['imeisv'] == device_info.imeisv]
            logging.debug(f'Before: Device - {device}, attack samples - {len(device_df[device_df["attack"] == 1])}'
                          f' benign samples - {len(device_df[device_df["attack"] == 0])}')

            device_train_df, df_tmp = train_test_split(device_df,
                                                       train_size=self.data_config.train_size,
                                                       stratify=device_df['attack_number'],
                                                       random_state=seed)
            device_val_df, device_test_df = train_test_split(df_tmp,
                                                             test_size=1. - self.data_config.train_size - self.data_config.val_size,
                                                             stratify=df_tmp['attack_number'],
                                                             random_state=seed)
            train_dfs.append(device_train_df)
            val_dfs.append(device_val_df)
            test_dfs.append(device_test_df)
        df_train = pd.concat(train_dfs)
        df_train = df_train.sort_values(by=['_time']).reset_index(drop=True)

        df_val = pd.concat(val_dfs)
        df_val = df_val.sort_values(by=['_time']).reset_index(drop=True)

        df_test = pd.concat(test_dfs)
        df_test = df_test.sort_values(by=['_time']).reset_index(drop=True)

        datasets = {'train': df_train, 'val': df_val, 'test': df_test}

        for k, df in datasets.items():
            logging.debug(f'Dataset {k} attack length: {len(df[df["attack"] == 1])} '
                          f'benign length: {len(df[df["attack"] == 0])} '
                          f'{k} shape: {df.shape}')
        return datasets

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cleans and preprocesses the input DataFrame by removing unwanted columns and invalid data.

        This method performs comprehensive data cleaning operations including:
        1. Dropping specified feature columns that are not needed for analysis
        2. Removing duplicate rows to ensure data uniqueness
        3. Handling NaT (Not a Time) values in the '_time' column with logging
        4. Dropping all rows containing any NaN values
        5. Resetting the DataFrame index for clean sequential indexing

        Args:
            df (pd.DataFrame): Input DataFrame to be cleaned. Expected to potentially
                contain columns specified in self.drop_features and a '_time' column.

        Returns:
            pd.DataFrame: Cleaned DataFrame with:
                - Unwanted columns removed
                - No duplicate rows
                - No NaT values in '_time' column
                - No NaN values in any column
                - Reset sequential index starting from 0

        Side Effects:
            Modifies the DataFrame in-place for the final index reset operation.
            Logs warnings when NaT values are found in the '_time' column.

        Example:
            >>> processor = DataProcessor()
            >>> clean_df = processor.clean_data(df)
            WARNING:root:Found 15 NaT values in _time column.
            >>> print(f"Cleaned: {len(clean_df)} rows from {len(df)} original")
            Cleaned: 9850 rows from 10000 original
        """
        df = df.drop(columns=self.drop_features, errors='ignore')
        df = df.drop_duplicates()
        # Handle NaT values in _time column
        if '_time' in df.columns:
            nat_indexes = df.index[df['_time'].isna()]
            if len(nat_indexes) > 0:
                logging.warning(f'Found {len(nat_indexes)} NaT values in _time column.')
                df = df.dropna(subset=['_time'])
        df = df.dropna(axis='index')
        df = df.reset_index(drop=True)
        return df

    def prepare_datasets(self, raw_dataset_path=None) -> None:
        """
        Transform raw network traffic data into processed train-validation-test splits.

        This method constitutes the primary data preparation pipeline, handling the
        complete transformation from raw CSV data through stratified splitting,
        scaler establishment, and persistence of processed datasets. The splitting
        strategy ensures balanced representation of attack patterns across all subsets
        while maintaining temporal ordering within each partition.

        The scaling establishment phase exclusively uses benign training samples
        to prevent contamination from attack signatures.

        Args:
            raw_dataset_path (str, optional): Path to raw dataset file. If None,
                                            uses the default path from configuration.
                                            Expected format is CSV with timestamped
                                            network metrics.

        Raises:
            FileNotFoundError: When the specified raw dataset path cannot be accessed.
            ValueError: When dataset lacks required columns or contains insufficient
                       samples for stratified splitting.
        """
        raw_dataset_path = raw_dataset_path or self.paths_config.raw_dataset

        check_existing_datasets()

        raw_df = self.read_ds(raw_dataset_path)
        logging.info(f'Loaded data from {raw_dataset_path}')

        # Split data to train/val/test
        datasets = self.split_data(raw_df)

        # Ensure a processed directory exists
        self.paths_config.processed_dir.mkdir(parents=True, exist_ok=True)

        # Setup and save scalers using training data
        self.setup_scalers()

        logging.info('Save datasets...')
        for k, df in datasets.items():
            save_path = get_dataset_path(k)
            df.to_csv(save_path, index=False)
            logging.info(f'{k} saved at {save_path}')

    def setup_scalers(self) -> None:
        """Sets up data scalers using only benign training samples for normalization.

        This method creates and configures scalers for time series data preprocessing by:
        1. Cleaning and sorting the training data chronologically
        2. Filtering to benign samples only (attack == 0) to avoid scaling bias
        3. Aggregating data by time to get mean values per timestep
        4. Creating a temporary TimeSeriesDataSet to extract fitted scalers

        The scalers are fitted exclusively on benign data to ensure that attack patterns
        don't influence the normalization parameters, which could reduce anomaly detection
        effectiveness.

        Side Effects:
            Sets self.scalers attribute with fitted scaling parameters from the
            temporary TimeSeriesDataSet.

        Note:
            - Only benign samples (attack == 0) are used for scaler fitting
            - Data is aggregated by time using _aggregate_by_time() method
            - Creates temporary time_idx and group_id columns for TimeSeriesDataSet compatibility
            - Logs the number of benign vs total samples used for transparency
            - Falls back to using all data if 'attack' column is missing

        Raises:
            May raise exceptions from TimeSeriesDataSet creation if data format is invalid.

        Example:
            >>> processor = DataProcessor()
            >>> processor.setup_scalers()
            INFO:root:Setting up scalers from benign training data only...
            INFO:root:Using 7500 benign samples out of 8000 total training samples for scaling
        """
        logging.info('Setting up scalers from benign training data only...')
        train_df = self.read_ds('train')
        train_df = self.get_partition(train_df)

        # Clean and prepare training data
        clean_train_df = self.clean_data(train_df.copy())
        clean_train_df = clean_train_df.sort_values(by=['_time']).reset_index(drop=True)

        # Filter to benign data only for scaling
        if 'attack' in clean_train_df.columns:
            benign_train_df = clean_train_df[clean_train_df['attack'] == 0].copy()
            logging.info(
                f'Using {len(benign_train_df)} benign samples out of {len(clean_train_df)} total training samples for scaling')
        else:
            logging.warning('No attack column found, using all training data for scaling')
            benign_train_df = clean_train_df

            # Get mean values per timestep for scaler fitting (benign only)
            mean_train_df = self._aggregate_by_time(benign_train_df)

            # Create time index
            mean_train_df['time_idx'] = range(len(mean_train_df))
            mean_train_df['group_id'] = 0
            # Create temporary TimeSeriesDataSet to get scalers
            temp_ts_ds = TimeSeriesDataSet(
                data=mean_train_df,
                time_idx='time_idx',
                target='attack',
                group_ids=['group_id'], # No grouping since we're using aggregated data
                max_encoder_length=self.data_config.seq_len,
                max_prediction_length=1,
                time_varying_known_reals=self.input_features,
                allow_missing_timesteps=False,
                predict_mode=False,
            )

            # Extract scalers
            self.scalers = temp_ts_ds.get_parameters()['scalers']
            self.paths_config.scalers_dir.mkdir(parents=True, exist_ok=True)
            for feature_name in self.scalers:
                scaler_path = self.scalers_dir.joinpath(feature_name + '.pkl')
                joblib.dump(self.scalers[feature_name], scaler_path)
                logging.info(f'Saved scaler for {feature_name} at {scaler_path}')

    def load_scalers(self) -> None:
        """Loads previously saved scalers from disk if not already loaded.

           This method loads serialized scaler objects from pickle files for each input feature.
           The scalers are only loaded if they haven't been loaded already (self.scalers is None).
           Each scaler file is expected to be named after its corresponding feature with a '.pkl'
           extension in the configured scalers directory.

           Side Effects:
               Sets self.scalers attribute to a dictionary mapping feature names to their
               corresponding loaded scaler objects.

           Note:
               - Only loads scalers if self.scalers is currently None (lazy loading)
               - Expects one .pkl file per feature in self.input_features
               - Scaler files must be named exactly as: {feature_name}.pkl
               - Uses joblib.load() for deserialization to match joblib.dump() from _save_scalers()

           Raises:
               FileNotFoundError: If any required scaler file is missing from the scalers directory.

           Example:
               >>> processor.load_scalers()
               >>> print(f"Loaded {len(processor.scalers)} scalers")
               Loaded 25 scalers
               >>> processor.load_scalers()  # Second call does nothing since scalers already loaded
           """

        if self.scalers is None:
            scalers = {}
            for feature_name in self.input_features:
                scaler_path = self.scalers_dir.joinpath(feature_name + '.pkl')
                if not scaler_path.exists():
                    logging.warning(f'Scaler for feature {feature_name} not found at {scaler_path}.')
                    continue
                scalers[feature_name] = joblib.load(scaler_path)
            if len(scalers) > 0:
                self.scalers = scalers
                logging.info(f'Loaded {len(self.scalers)} scalers')

    def scale_data(self, df) -> pd.DataFrame:
        """
        Apply feature scaling transformations to dataset using established parameters.

        This method transforms numerical features using previously fitted scaling
        parameters, ensuring consistent normalization across training and inference
        phases. The scaling application maintains the established statistical
        properties derived from benign training samples.

        Args:
            df (pd.DataFrame): Dataset requiring feature scaling with numerical
                             columns matching the established feature specifications.

        Returns:
            pd.DataFrame: Scaled dataset with transformed numerical features
                         maintaining original structure and non-feature columns.
        """
        if not self.scalers:
            self.load_scalers()
        for feature_name in self.scalers:
            df[feature_name] = self.scalers[feature_name].transform(df[[feature_name]])
        return df

    def get_dataset(self, data_path : str | Path, only_benign: bool = False) -> TimeSeriesDataSet:
        """
        Construct PyTorch Forecasting TimeSeriesDataSet for model training and evaluation.

        This method orchestrates the complete dataset preparation pipeline from raw
        data loading through partitioning, cleaning, aggregation, and time series
        dataset construction. The process integrates established scaling parameters
        and supports both full dataset usage and benign-only filtering for specific
        training scenarios.

        The time series dataset construction includes proper temporal indexing,
        group identification for forecasting frameworks, and integration with
        established scaling parameters. The method handles the complexity of
        transforming aggregated network data into formats suitable for sequence
        modeling approaches used in anomaly detection.

        Args:
            data_path (str | Path): Dataset identifier for loading (train, val, test, or
                           custom path specification).
            only_benign (bool, optional): Flag to filter dataset to benign samples
                                        only, useful for unsupervised training
                                        approaches. Defaults to False.

        Returns:
            TimeSeriesDataSet: Configured dataset object ready for PyTorch Lightning
                              training with proper temporal structuring, scaling
                              integration, and sequence length configuration.
        """

        if self.data_config.num_workers <= 1:
            self.data_config.prefetch_factor = None
            self.data_config.persistent_workers = False
        logging.info(f'Get {data_path} dataloader with '
                     f'batch_size: {self.data_config.batch_size}, '
                     f'seq_len: {self.data_config.seq_len}, '
                     f'num_features: {len(self.input_features)}, '
                     f'partition_id: {self.data_config.partition_id}, '
                     f'only_benign: {only_benign}')

        self.setup_scalers()

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

        # Aggregate by time to get mean values per timestep
        df = self._aggregate_by_time(df)

        # Create time index
        df['time_idx'] = range(len(df))
        df['group_id'] = 0

        ts_ds = TimeSeriesDataSet(data=df,
                                  time_idx='time_idx',
                                  target='attack',
                                  group_ids=['group_id'],
                                  max_encoder_length=self.data_config.seq_len,
                                  max_prediction_length=1,
                                  time_varying_known_reals=self.input_features,
                                  allow_missing_timesteps=False,
                                  predict_mode=False,
                                  scalers=self.scalers
                                  )
        return ts_ds

    def get_dataloader(self, path, only_benign: bool = False, train: bool = True) -> DataLoader:
        """
        Create PyTorch DataLoader with optimized configuration for model training.

        This method wraps the TimeSeriesDataSet construction with DataLoader
        configuration, applying performance optimizations based on the processing
        configuration. The method handles worker process management, memory pinning,
        and prefetching strategies to maximize training throughput while maintaining
        data consistency.

        The DataLoader configuration adapts to single-worker scenarios by disabling
        features that require multiple processes, ensuring compatibility across
        different deployment environments from development to production settings.

        Args:
            path (str): Dataset identifier for loading through the established
                       dataset pipeline.
            only_benign (bool, optional): Flag to restrict dataset to benign samples
                                        for unsupervised learning approaches.
                                        Defaults to False.
            train (bool, optional): Training mode flag affecting DataLoader behavior
                                   such as shuffling and data ordering. Defaults to True.

        Returns:
            DataLoader: Configured PyTorch DataLoader with optimized settings for
                       efficient batch processing and model training integration.
        """
        return self.get_dataset(path, only_benign).to_dataloader(
            train=train,
            batch_size=self.data_config.batch_size,
            num_workers=self.data_config.num_workers,
            pin_memory=self.data_config.pin_memory,
            prefetch_factor=self.data_config.prefetch_factor,
            persistent_workers=self.data_config.persistent_workers)

    def _aggregate_by_time(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform temporal aggregation of network metrics across devices.

        This method consolidates device-level measurements into network-wide
        temporal patterns by computing mean values for each feature at every
        timestamp. The aggregation strategy treats attack presence as a binary
        maximum operation, where any device exhibiting attack behavior during
        a timestamp marks the entire network state as compromised for that period.

        The aggregation process reduces computational complexity while preserving
        essential temporal patterns in network behavior. This approach facilitates
        anomaly detection at the network level rather than device level, aligning
        with the project's focus on infrastructure-wide security analytics.

        Args:
            df (pd.DataFrame): Device-level network measurements with temporal
                             indexing and feature columns for aggregation.

        Returns:
            pd.DataFrame: Temporally aggregated dataset with mean feature values
                         and maximum attack indicators for each timestamp.

        Note:
            The aggregation preserves temporal ordering and maintains consistency
            in attack labeling across the aggregated time series for subsequent
            anomaly detection processing.
        """
        agg_dict = {feature: 'mean' for feature in self.input_features}
        # For attack column, take max (if any device has attack=1, the timestep is considered attack)
        agg_dict['attack'] = 'max'

        aggregated_df = df.groupby('_time').agg(agg_dict).reset_index()

        logging.info(f'Aggregated data shape: {aggregated_df.shape}')
        logging.info(f'Attack samples after aggregation: {len(aggregated_df[aggregated_df["attack"] == 1])}')
        logging.info(f'Benign samples after aggregation: {len(aggregated_df[aggregated_df["attack"] == 0])}')

        return aggregated_df

if __name__ == '__main__':
    dp = DataProcessor()
    dl = dp.get_dataloader('val', train=False)
    for i, sample in enumerate(dl):
        print(sample[0]['encoder_cont'][0, 0])
        break
    exit()
