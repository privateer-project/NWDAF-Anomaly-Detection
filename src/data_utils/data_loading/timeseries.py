from typing import Dict
import os
from glob import glob

from pandas import DataFrame
import joblib
from pytorch_forecasting import TimeSeriesDataSet
from src.config import Paths, FeatureInfo


class TimeSeriesDatasetCreator:
    """Handles creation of time series datasets."""

    def __init__(self, features: Dict[str, FeatureInfo], paths: Paths):
        self.features = features
        self.paths = paths

    def create_dataset(self, df: DataFrame, window_size: int, use_existing_scalers: bool =True) -> TimeSeriesDataSet:
        """Create a TimeSeriesDataSet from the preprocessed DataFrame."""
        scalers = {}
        if use_existing_scalers:
            scalers = self._load_scalers()
        df = self._prepare_time_features(df)
        input_features = [feat for feat, info in self.features.items() if info.input]

        return TimeSeriesDataSet(
            data=df,
            time_idx='time_idx',
            target='attack',
            group_ids=['imeisv'],
            max_encoder_length=window_size,
            min_encoder_length=window_size,
            min_prediction_length=1,
            max_prediction_length=1,
            time_varying_unknown_reals=input_features,
            scalers=scalers,
            target_normalizer=None,
            allow_missing_timesteps=False
        )

    def _load_scalers(self) -> Dict:
        scalers = {}
        scaler_files = glob(str(self.paths.scalers.joinpath('*.scaler')))
        for scaler_path in scaler_files:
            col = os.path.splitext(os.path.basename(scaler_path))[0]
            scalers[col] = joblib.load(scaler_path)
        return scalers

    @staticmethod
    def _prepare_time_features(df: DataFrame) -> DataFrame:
        df.reset_index(inplace=True)
        df['time_idx'] = df.groupby('imeisv')['_time'].cumcount()
        return df
