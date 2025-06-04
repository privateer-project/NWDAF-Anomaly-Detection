import os
import torch

import sys
sys.path.append(os.path.join(sys.path[0], 'common_libraries'))

from privateer_ad.models import TransformerAD

# Define the absolute path to the directory where model files are stored
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..','saved_models')
MODEL_DIR = os.path.abspath(MODEL_DIR)
print("caminho....."+MODEL_DIR)
print("caminho2....."+os.path.dirname(__file__))

def save_model(file, filename):
    """
    Save an uploaded PKL model file to the models directory.

    Args:
        file (FileStorage): The uploaded file object (typically via Flask request).
        filename (str): Desired name of the file to save.

    Returns:
        str: Full path to the saved model file.
    """
    path = os.path.join(MODEL_DIR, filename)
    file.save(path)
    return path

def list_model_files():
    """
    List all model files currently stored in the models directory.

    Returns:
        list: A list of all filenames in the models directory.
    """
    return os.listdir(MODEL_DIR)

def delete_model(filename):
    """
    Delete a specific model file from the models directory if it exists.

    Args:
        filename (str): The name of the file to delete.

    Returns:
        bool: True if the file was deleted, False otherwise.
    """
    path = os.path.join(MODEL_DIR, filename)
    if os.path.exists(path):
        os.remove(path)
        return True
    return False

def load_model(filename):
    """
    Load a .pkl model file and initialize it as a TransformerAD instance.

    This function assumes that the model architecture is TransformerAD and the
    state_dict may have a prefix (e.g., from Distributed Data Parallel training).

    Args:
        filename (str): The filename of the model to load.

    Returns:
        TransformerAD: The loaded and initialized model.

    Raises:
        FileNotFoundError: If the file does not exist in the models directory.
    """
    path = os.path.join(MODEL_DIR, filename)
    if os.path.exists(path):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize the model architecture
        model = TransformerAD()

        # Load the state_dict from disk
        state_dict = torch.load(path, map_location=torch.device('cpu'))

        # If model was trained using Distributed Data Parallel, remove the module prefix
        state_dict = {key.removeprefix('_module.'): value for key, value in state_dict.items()}

        # Load the weights into the model
        model.load_state_dict(state_dict)

        # Move the model to the appropriate device (CPU or GPU)
        model = model.to(device)
        return model

    raise FileNotFoundError(f"Model file '{filename}' not found in '{MODEL_DIR}'")
