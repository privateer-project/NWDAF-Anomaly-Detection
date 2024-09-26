import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler


def scale_ts(df, feature_columns, scaler=None):

    if not scaler:
        scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[feature_columns])

    scaled_df = pd.DataFrame(scaled_data, columns=feature_columns, index=df.index)

    result_df = df[["_time", "imeisv"]].copy()
    result_df = pd.concat([result_df, scaled_df], axis=1)

    return (result_df, scaler)


def smooth_ts(df, feature_columns, rolling_window):

    imeisv_df_for_ma = {}

    for imeisv, imeisv_df in df.groupby("imeisv"):
        imeisv_df[feature_columns] = (
            imeisv_df[feature_columns].rolling(window=rolling_window).mean()
        )

        imeisv_df_for_ma[str(imeisv)] = imeisv_df

    smoothed_df = pd.concat(list(imeisv_df_for_ma.values()))

    return smoothed_df


def apply_diff(df, feature_columns):
    imeisv_df_for_diff = {}

    for imeisv, imeisv_df in df.groupby("imeisv"):
        imeisv_df[feature_columns] = imeisv_df[feature_columns].diff()

        imeisv_df_for_diff[str(imeisv)] = imeisv_df

    diffed_df = pd.concat(list(imeisv_df_for_diff.values()))

    return diffed_df
