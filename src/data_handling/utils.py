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
    df.to_csv(get_dataset_path(name), index=True)

def check_existing_datasets(force: bool):
    for mode in ['train', 'val', 'test']:
        path = get_dataset_path(mode)
        if os.path.exists(path) and not force:
            raise FileExistsError(
            f'File {path} exists. Set force=True to overwrite.'
            )

def split_normal_data(data: DataFrame, train_size: float) -> Tuple[DataFrame, ...]:
    train_df = DataFrame()
    val_df = DataFrame()
    test_df = DataFrame()

    for _, normal_per_device in data.groupby('imeisv'):
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