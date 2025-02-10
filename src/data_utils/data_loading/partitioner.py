from pandas import DataFrame
from datasets import Dataset
from flwr_datasets.partitioner import PathologicalPartitioner

from config import PartitionConfig


class DataPartitioner:
    """Handles data partitioning for federated learning."""

    @staticmethod
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
