from typing import Optional, Tuple
import os
import pandas as pd
from torch.utils.data import DataLoader
from src.config import Paths, MetaData, PartitionConfig, HParams
from src.data_utils.data_loading import TimeSeriesDatasetCreator, DataPartitioner

class DataLoaderFactory:
    """Creates DataLoaders for training and evaluation."""

    def __init__(self, metadata: MetaData, paths: Paths, hparams: HParams):
        self.paths = paths
        self.creator = TimeSeriesDatasetCreator(metadata.features, paths)
        self.partitioner = DataPartitioner()
        self.hparams = hparams

    def get_dataloaders(self, window_size: int = 12,
                        partition_config: Optional[PartitionConfig] = None,
                        train: bool = True
                        ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Get train, validation and test dataloaders."""
        datasets = {}
        for split in ['train', 'val', 'test']:
            df = pd.read_csv(self.paths.processed.joinpath(f'{split}.csv'))

            if partition_config:
                df = self.partitioner.partition_data(df, partition_config)

            dataset = self.creator.create_dataset(df, window_size)
            datasets[split] = dataset.to_dataloader(
                train=train and split == 'train',
                batch_size=self.hparams.batch_size,
                num_workers=os.cpu_count(),
                pin_memory=True,
                prefetch_factor=self.hparams.batch_size * 100,
                persistent_workers=True
            )

        return datasets['train'], datasets['val'], datasets['test']

    def get_single_dataloader(self, split: str, window_size: int = 12,
                              partition_config: Optional[PartitionConfig] = None,
                              train: bool = True, **kwargs) -> DataLoader:
        """Get a single dataloader."""
        df = pd.read_csv(self.paths.processed.joinpath(f'{split}.csv'))

        if partition_config:
            df = self.partitioner.partition_data(df, partition_config)

        dataset = self.creator.create_dataset(df, window_size)
        return dataset.to_dataloader(
            train=train and split == 'train',
            batch_size=self.hparams.batch_size, **kwargs)
