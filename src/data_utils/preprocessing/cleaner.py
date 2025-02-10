from typing import Dict

from pandas import DataFrame

from src.config import FeatureInfo


class DataCleaner:
    """Handles data cleaning and preprocessing operations."""
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
