from typing import Dict, Iterator

import numpy as np
import pandas as pd
import torch
import psutil

from sklearn.preprocessing import RobustScaler
from torch.utils.data import IterableDataset
from src.config import FeatureInfo, MetaData, AttackInfo
import time



class TimeSeriesGenerator(IterableDataset):
    """Memory-efficient generator for time series data."""

    def __init__(self, data: np.ndarray, window_size: int, batch_size: int, shuffle: bool = True):
        self.data = data
        self.window_size = window_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_samples = len(data) - window_size + 1

    def __iter__(self) -> Iterator[np.ndarray]:
        while True:
            # Generate indices for the sequences
            if self.shuffle:
                start_indices = np.random.randint(0, self.num_samples, size=self.batch_size)
            else:
                start_indices = np.arange(self.num_samples)

            # Create batches
            for idx in range(0, len(start_indices), self.batch_size):
                batch_indices = start_indices[idx:idx + self.batch_size]
                batch = np.array([
                    self.data[i:i + self.window_size] for i in batch_indices])
                yield torch.FloatTensor(batch)


    # Process each device
    train_datasets = {}
    val_datasets = {}
    test_datasets = {}
    scalers = {}

    for device in df['imeisv'].unique():
        # Get device data
        device_benign = benign_data[benign_data['imeisv'] == device][input_features].values
        device_attack = attack_data[attack_data['imeisv'] == device][input_features].values

        if len(device_benign) == 0:
            continue

        # Split indices
        n_samples = len(device_benign)
        train_idx = int(n_samples * 0.7)
        val_idx = int(n_samples * 0.85)

        # Fit scaler on training data only
        scaler = RobustScaler()
        train_data = device_benign[:train_idx]
        scaler.fit(train_data)
        scalers[device] = scaler

        # Create scaled datasets
        train_scaled = scaler.transform(train_data)
        val_scaled = scaler.transform(device_benign[train_idx:val_idx])
        test_benign_scaled = scaler.transform(device_benign[val_idx:])
        test_attack_scaled = scaler.transform(device_attack) if len(device_attack) > 0 else None

        # Create generators
        if len(train_scaled) >= window_size:
            train_datasets[device] = TimeSeriesGenerator(
                train_scaled, window_size, device_batch_size, shuffle=True
            )

        if len(val_scaled) >= window_size:
            val_datasets[device] = TimeSeriesGenerator(
                val_scaled, window_size, device_batch_size, shuffle=False
            )

        if len(test_benign_scaled) >= window_size:
            test_datasets[f"{device}_benign"] = TimeSeriesGenerator(
                test_benign_scaled, window_size, device_batch_size, shuffle=False
            )

        if test_attack_scaled is not None and len(test_attack_scaled) >= window_size:
            test_datasets[f"{device}_attack"] = TimeSeriesGenerator(
                test_attack_scaled, window_size, device_batch_size, shuffle=False
            )

    return {
        'train_datasets': train_datasets,
        'val_datasets': val_datasets,
        'test_datasets': test_datasets,
        'scalers': scalers,
        'input_features': input_features
    }


class MultiDeviceDataLoader:
    """Combines data from multiple devices into a single stream."""

    def __init__(self, datasets: Dict[str, TimeSeriesGenerator], batch_size: int):
        self.datasets = datasets
        self.batch_size = batch_size
        self.device_iterators = {
            device: iter(dataset)
            for device, dataset in datasets.items()
        }

    def __iter__(self):
        return self

    def __next__(self):
        # Randomly select devices for this batch
        selected_devices = np.random.choice(
            list(self.datasets.keys()),
            size=min(len(self.datasets), self.batch_size),
            replace=True
        )

        # Collect batches from selected devices
        batches = []
        for device in selected_devices:
            try:
                batch = next(self.device_iterators[device])
                batches.append(batch)
            except StopIteration:
                # Reset iterator if exhausted
                self.device_iterators[device] = iter(self.datasets[device])
                batch = next(self.device_iterators[device])
                batches.append(batch)

        # Combine device batches
        return torch.cat(batches)



# Usage example:
def get_data_loaders(prepared_data: dict, batch_size: int = 32):
    """Create data loaders for all splits."""
    return {
        'train': MultiDeviceDataLoader(prepared_data['train_datasets'], batch_size),
        'val': MultiDeviceDataLoader(prepared_data['val_datasets'], batch_size),
        'test': MultiDeviceDataLoader(prepared_data['test_datasets'], batch_size)
    }


def test_data_preparation(features: Dict[str, FeatureInfo], attacks: Dict[str, AttackInfo]):

    """Test the data preparation pipeline."""
    print("Starting data preparation tests...")

    # Load your data and metadata
    df = pd.read_csv('/data/raw/amari_ue_data_merged_with_attack_number.csv')

    # Test 1: Check attack mask creation
    print("\nTest 1: Attack mask creation")
    start_time = time.time()
    print(f"Time taken: {time.time() - start_time:.2f} seconds")
    print(f"Number of attack periods: {attack_mask.sum()}")
    print(f"Percentage of attack data: {(attack_mask.sum() / len(df)) * 100:.2f}%")

    # Test 2: Prepare datasets
    print("\nTest 2: Dataset preparation")
    start_time = time.time()
    prepared_data = prepare_dataset_efficient(
        df,
        features=features,
        attacks=attacks,
        window_size=100,
        device_batch_size=1000
    )
    print(f"Time taken: {time.time() - start_time:.2f} seconds")

    # Print dataset information
    print("\nDataset information:")
    print(f"Number of input features: {len(prepared_data['input_features'])}")
    print(f"Number of devices with train data: {len(prepared_data['train_datasets'])}")
    print(f"Number of devices with test data: {len(prepared_data['test_datasets'])}")

    # Test 3: Data loading and iteration
    print("\nTest 3: Data loading")
    loaders = get_data_loaders(prepared_data, batch_size=32)

    # Test training data loader
    print("\nTesting training data loader...")
    batch_sizes = []
    unique_values = set()

    start_time = time.time()
    for i, batch in enumerate(loaders['train']):
        if i >= 10:  # Test first 10 batches
            break

        batch_sizes.append(batch.shape[0])

        # Check batch properties
        print(f'\nBatch - {i + 1}:')
        print(f'NaNs: {torch.sum(torch.isnan(batch))}')
        print(f"Shape: {batch.shape}")
        print(f"Value range: [{batch.min():.3f}, {batch.max():.3f}]")
        print(f"Mean: {batch.mean():.3f}")
        print(f"Std: {batch.std():.3f}")

        # Add values to set to check for variety
        unique_values.update(batch.numpy().flatten())

    print(f"\nTime to load 10 batches: {time.time() - start_time:.2f} seconds")
    print(f"Average batch size: {np.mean(batch_sizes):.2f}")
    print(f"Number of unique values: {len(unique_values)}")

    # Test 4: Memory usage
    print("\nTest 4: Memory usage")
    process = psutil.Process()
    print(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")

    # Test 5: Scaler consistency
    print("\nTest 5: Scaler consistency")
    for device, scaler in prepared_data['scalers'].items():
        print(f"\nDevice: {device}")
        print(f"Scale mean: {np.mean(scaler.scale_):.3f}")
        print(f"Center mean: {np.mean(scaler.center_):.3f}")

    return prepared_data, loaders


if __name__ == "__main__":
    # Run tests
    metadata = MetaData()
    prepared_data, loaders = test_data_preparation(features=metadata.features, attacks=metadata.attacks)

    # Interactive testing section
    print("\nEnter 'q' to quit, or press Enter to get next batch")
    while True:
        user_input = input()
        if user_input.lower() == 'q':
            break

        try:
            batch = next(iter(loaders['train']))
            print(f"Batch shape: {batch.shape}")
            print(f"Value range: [{batch.min():.3f}, {batch.max():.3f}]")
        except StopIteration:
            print("Reached end of dataset")
