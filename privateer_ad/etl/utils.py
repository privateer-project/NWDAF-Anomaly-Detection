import os

from privateer_ad.config import PathConfig

def get_dataset_path(dataset_path: str) -> str:
    """Get full path for a dataset file."""
    if dataset_path in ['train', 'val', 'test']:
        return str(PathConfig().processed_dir.joinpath(f"{dataset_path}.csv"))
    return dataset_path

def check_existing_datasets():
    for mode in ['train', 'val', 'test']:
        path = get_dataset_path(mode)
        if os.path.exists(path):
            raise FileExistsError(f'File {path} exists.')
