import os

import sys
sys.path.append(os.path.join(sys.path[0], 'common_libraries'))

from load_dataset.controllers.transform import DataProcessor

# Define the absolute path to the datasets directory
# This assumes the datasets folder is located at: /load_dataset/datasets
DATASET_DIR = os.path.join(os.path.dirname(__file__),'..', 'datasets')
DATASET_DIR = os.path.abspath(DATASET_DIR)

print(f"Dataset directory set to: {DATASET_DIR}")

def save_csv(file, filename):
    """
    Save an uploaded CSV file to the datasets directory.

    Parameters:
        file (FileStorage): The uploaded file object.
        filename (str): The name to save the file as.

    Returns:
        str: Full path to the saved file.
    """
    path = os.path.join(DATASET_DIR, filename)
    file.save(path)
    return path

def list_csv_files():
    """
    List all files in the datasets directory.

    Returns:
        list: A list of filenames present in the datasets directory.
    """
    return os.listdir(DATASET_DIR)

def delete_csv(filename):
    """
    Delete a specific CSV file from the datasets directory.

    Parameters:
        filename (str): The name of the file to delete.

    Returns:
        bool: True if the file was deleted, False if it did not exist.
    """
    path = os.path.join(DATASET_DIR, filename)
    if os.path.exists(path):
        os.remove(path)
        return True
    return False

def load_csv(filename):
    """
    Load and process a CSV file using DataProcessor.

    This method initializes the DataProcessor, reads the CSV file,
    and returns the processed DataLoader containing the data.

    Parameters:
        filename (str): Name of the CSV file to load.

    Returns:
        DataLoader: Processed data in the form of a PyTorch DataLoader.

    Raises:
        FileNotFoundError: If the specified file does not exist.
    """
    path = os.path.join(DATASET_DIR, filename)
    print(f"Loading dataset from: {path}\n")
    if os.path.exists(path):
        dp = DataProcessor(partition=False)
        data = dp._read_csv(path)
        dataloader  = dp.get_dataloader2(data, seq_len=12, only_benign=False)
        return dataloader
    return path
    #raise FileNotFoundError
