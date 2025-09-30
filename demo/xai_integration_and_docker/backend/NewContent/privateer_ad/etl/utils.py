import os
from pathlib import Path

from privateer_ad.config import PathConfig

def get_dataset_path(dataset_path: str) -> Path:
    """Resolves and validates a dataset path, supporting both shorthand and full paths.

       This function handles two types of input:
       1. Shorthand strings ('train', 'val', 'test') which are mapped to processed CSV files
       2. Full file paths which are converted to Path objects and validated

       Args:
           dataset_path (str): Either a shorthand dataset name ('train', 'val', 'test')
               or a full path to a dataset file.

       Returns:
           Path: A validated Path object pointing to the dataset file.

       Raises:
           FileNotFoundError: If the resolved dataset file does not exist.

       Example:
           >>> get_dataset_path('train')
           PosixPath('/path/to/processed/train.csv')
           >>> get_dataset_path('/custom/path/data.csv')
           PosixPath('/custom/path/data.csv')
       """
    if dataset_path in ['train', 'val', 'test']:
        dataset_path = PathConfig().processed_dir.joinpath(f"{dataset_path}.csv").as_posix()
    if isinstance(dataset_path, str):
        dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f'Dataset file not found: {dataset_path}')
    return dataset_path

def check_existing_datasets():
    """Checks if any of the standard dataset files already exist and raises an error if found.

    This function verifies that none of the processed dataset files (train.csv, val.csv,
    test.csv) exist in the configured processed directory. This is typically used before
    data processing operations to prevent accidental overwriting of existing datasets.

    Raises:
        FileExistsError: If any of the dataset files ('train', 'val', 'test') already
            exist in the processed directory.

    Example:
        >>> check_existing_datasets()  # Passes if no files exist
        >>> check_existing_datasets()  # Raises FileExistsError if train.csv exists
        FileExistsError: File /path/to/processed/train.csv exists.
    """
    for mode in ['train', 'val', 'test']:
        path = get_dataset_path(mode)
        if os.path.exists(path):
            raise FileExistsError(f'File {path} exists.')
