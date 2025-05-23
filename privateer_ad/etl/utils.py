import os

from privateer_ad.config import get_paths


def get_dataset_path(dataset_path: str) -> str:
    """Get full path for a dataset file."""
    if dataset_path in ['train', 'val', 'test']:
        return str(get_paths().processed_dir.joinpath(f"{dataset_path}.csv"))
    return dataset_path

def check_existing_datasets():
    for mode in ['train', 'val', 'test']:
        path = get_dataset_path(mode)
        if os.path.exists(path):
            raise FileExistsError(f'File {path} exists.')

def get_scaler_path(scaler_name: str) -> str:
    """Get full path for a scaler file."""
    return str(get_paths().scalers_dir.joinpath(f'{scaler_name}.scaler'))
