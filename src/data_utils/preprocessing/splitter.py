from typing import Tuple

import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import train_test_split

from src.config import MetaData


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
