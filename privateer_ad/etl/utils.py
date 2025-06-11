import os

from privateer_ad.config import PathConfig

def get_dataset_path(dataset_path: str) -> str:
    """
    Args:
        dataset_path (str): Either a standard dataset identifier ('train', 'val', 'test')
                           or a direct file path. Standard identifiers are resolved to
                           CSV files in the processed data directory, while other values
                           are returned unchanged.

    Returns:
        str: The complete file system path to the dataset. For standard identifiers,
             this will be a path within the processed data directory with a CSV
             extension. For other inputs, the original string is returned as-is.

    Example:
        Standard dataset access:
        >>> get_dataset_path('train')
        '/path/to/project/data/processed/train.csv'

        Custom file path:
        >>> get_dataset_path('/custom/path/my_data.csv')
        '/custom/path/my_data.csv'

    Note:
        This function does not verify that the returned path actually exists on the
        filesystem. It simply constructs the expected path based on the project's
        configuration and naming conventions.
    """
    if dataset_path in ['train', 'val', 'test']:
        return str(PathConfig().processed_dir.joinpath(f"{dataset_path}.csv"))
    return dataset_path

def check_existing_datasets():
    """
    Verify that standard dataset files do not already exist before processing.
    """
    for mode in ['train', 'val', 'test']:
        path = get_dataset_path(mode)
        if os.path.exists(path):
            raise FileExistsError(f'File {path} exists.')
