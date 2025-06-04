import os
from copy import deepcopy

import joblib
import pandas as pd

from pandas import DataFrame
from datasets import Dataset
from pytorch_forecasting import TimeSeriesDataSet
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from privateer_ad.config import PathsConf, MetaData


class DataProcessor:
    """Main class orchestrating data transform and loading."""

    def __init__(self, partition=False):
        self.metadata = MetaData()
        self.paths = PathsConf()
        self.scaler_path = os.path.join(self.paths.scalers, 'scaler.pkl')
        self.pca = None
        self.scaler = None
        self.input_features = self.metadata.get_input_features()
        self.drop_features = self.metadata.get_drop_features()
        self.partition = partition


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

    def setup_scaler(self, df):
        """
        Scaling and normalizing numeric values, imputing missing values, clipping outliers,
         and adjusting values that have skewed distributions.
        """
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
        """Scale numeric features using standardization (mean=0, std=1)."""
        self.load_scaler()
        transformed = self.scaler.transform(df[self.input_features])
        df.loc[:, self.input_features] = transformed
        return df

    def _read_csv(self, path):
        path = path
        dtypes = deepcopy(self.metadata.get_features_dtypes())
        dtypes.pop('_time', None)
        try:
            return pd.read_csv(path,
                               dtype=self.metadata.get_features_dtypes(),
                               parse_dates=['_time'])
        except FileNotFoundError:
            raise FileNotFoundError(f'File {path} not found.')


    def get_dataloader2(self,
                       
                       data,
                       batch_size=1024,
                       seq_len=6,
                       partition_id=0,
                       only_benign=False) -> DataLoader:

        """Get train, validation and test dataloaders."""
        dataloader_params = {
                             'batch_size': batch_size,
                             'num_workers': os.cpu_count(),
                             'pin_memory': True,
                             'prefetch_factor': 10000,
                             'persistent_workers': True}

        df = self.preprocess_data(df=data,
                                  partition_id=partition_id)
        df = df.astype({'attack': int})

        # Create time index for each device
        df['time_idx'] = df.groupby('imeisv')['_time'].cumcount()
        dataloader = TimeSeriesDataSet(data=df,
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
        return dataloader