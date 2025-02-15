import os
from typing import Tuple

import pandas as pd
from pandas import DataFrame
from datasets import Dataset
from flwr_datasets.partitioner import PathologicalPartitioner
from sklearn.model_selection import train_test_split

from src.config import PartitionConfig, ProjectPaths


def partition_data(df: DataFrame, config: PartitionConfig) -> DataFrame:
    """Partition data based on provided configuration."""
    partitioner = PathologicalPartitioner(
        num_partitions=config.num_partitions,
        num_classes_per_partition=config.num_classes_per_partition,
        partition_by='imeisv',
        class_assignment_mode='first-deterministic'
    )
    partitioner.dataset = Dataset.from_pandas(df)
    partitioned_df = partitioner.load_partition(config.partition_id).to_pandas(batched=False)
    return partitioned_df[df.columns]

def get_dataset_path(dataset_name: str) -> str:
    """Get full path for a dataset file."""
    return str(ProjectPaths.processed.joinpath(f"{dataset_name}.csv"))

def save_dataset(name: str, df: DataFrame):
    df.reset_index(drop=True, inplace=True)
    df.to_csv(get_dataset_path(name), index=False)

def check_existing_datasets(force: bool):
    for mode in ['train', 'val', 'test']:
        path = get_dataset_path(mode)
        if os.path.exists(path) and not force:
            raise FileExistsError(
            f'File {path} exists. Set force=True to overwrite.'
            )

def split_data(data: DataFrame, train_size: float) -> Tuple[DataFrame, ...]:
    train_dfs = []
    val_dfs = []
    test_dfs = []
    for _, device_data in data.groupby('imeisv'):
        device_train_df, device_val_test_df = train_test_split(
            device_data,
            shuffle=False,
            train_size=train_size,
            random_state=42
        )
        device_val_df, device_test_df = train_test_split(
            device_val_test_df,
            shuffle=False,
            test_size=0.5,
            random_state=42)

        train_dfs.append(device_train_df)
        val_dfs.append(device_val_df)
        test_dfs.append(device_test_df)
    train_df = pd.concat(train_dfs)
    val_df = pd.concat(val_dfs)
    test_df = pd.concat(test_dfs)
    return train_df, val_df, test_df