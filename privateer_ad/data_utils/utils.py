import os

from privateer_ad.config import PathsConf

def get_dataset_path(dataset_name: str) -> str:
    """Get full path for a dataset file."""
    return str(PathsConf.processed.joinpath(f"{dataset_name}.csv"))

def check_existing_datasets():
    for mode in ['train', 'val', 'test']:
        path = get_dataset_path(mode)
        if os.path.exists(path):
            raise FileExistsError(f'File {path} exists.')

def get_scaler_path(scaler_name: str) -> str:
    """Get full path for a scaler file."""
    return str(PathsConf.scalers.joinpath(f'{scaler_name}.scaler'))
