import random
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler


class MultiVarTsDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = [
            seq for seq in sequences 
            if not torch.isnan(torch.tensor(seq, dtype=torch.float)).any()
        ]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        return (
            torch.tensor(sequence, dtype=torch.float), 
            torch.tensor(sequence, dtype=torch.float)
        )
        
def get_rolling_windows(df, window_size, step_size, features):
    windows = []
    for start in range(0, len(df) - window_size + 1, step_size):
        end = start + window_size
        window_df = df[start:end]
        if window_df['imeisv'].nunique() >= 2:
            continue
        windows.append(window_df[features])
    return np.array(windows)

def split_train_test(data_list, train_size=0.8):
    random.shuffle(data_list)

    split_index = int(len(data_list) * train_size)
    
    train_data = data_list[:split_index]
    test_data = data_list[split_index:]
        
    return train_data, test_data

def create_ds_loader(benign_data, malicious_data, window_size, step_size, features, batch_size):
    
    if step_size is None:
        step_size = int(np.round(window_size/3))
    
    benign_data = get_rolling_windows(benign_data, window_size, step_size, features)
    mal_data = get_rolling_windows(malicious_data, window_size, step_size, features)
    
    train_data, val_data = split_train_test(benign_data)
    
    # scaler = StandardScaler()

    # scaler.fit(np.vstack(train_data))
    # train_data_scaled = [scaler.transform(ds) for ds in train_data]
    # val_data_scaled = [scaler.transform(ds) for ds in val_data]

    # mal_data_scaled = [scaler.transform(ds) for ds in mal_data]
    
    train_dataset = MultiVarTsDataset(train_data)
    val_dataset = MultiVarTsDataset(val_data)

    mal_dataset = MultiVarTsDataset(mal_data)

    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_data_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    malicious_data_loader = DataLoader(mal_dataset, batch_size=10, shuffle=False)
    
    return train_data_loader, val_data_loader, malicious_data_loader


def create_test_ds_loaders(benign_data, malicious_data, window_size, step_size, features, batch_size):
    if step_size is None:
        step_size = int(np.round(window_size/3))
    
    benign_data = get_rolling_windows(benign_data, window_size, step_size, features)
    mal_data = get_rolling_windows(malicious_data, window_size, step_size, features)
    
    benign_dataset = MultiVarTsDataset(benign_data)
    mal_dataset = MultiVarTsDataset(mal_data)
    
    benign_test_data_loader = DataLoader(benign_dataset, batch_size=batch_size, shuffle=False)
    mal_test_data_loader = DataLoader(mal_dataset, batch_size=batch_size, shuffle=False)
    
    return benign_test_data_loader, mal_test_data_loader