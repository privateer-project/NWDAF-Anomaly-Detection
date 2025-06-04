import requests
import pickle
import base64
import torch
import io
from torch.utils.data import DataLoader

from xAI_shap.controllers.urls import BASE_URL_DATASET, BASE_URL_MODEL

def send_dataset_request(filename, batch_size=32):
    """
    Sends a request to the dataset service API to load a specific dataset.

    The response is expected to be a base64-encoded pickled object representing a PyTorch DataLoader.

    Args:
        filename (str): Name of the dataset file (without path).
        batch_size (int): Desired batch size for the DataLoader (passed to the backend service).

    Returns:
        dict: A dictionary containing the deserialized PyTorch DataLoader.
              Example: { "dataset_loader": DataLoader(...) }

    Raises:
        Exception: If the request fails or the API returns an error.
    """
    url = f"{BASE_URL_DATASET}/api/datasets/{filename}"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()

        # Decode base64 string and unpickle the DataLoader
        dataloader = pickle.loads(base64.b64decode(data["dataset_bytes_base64"]))
        return {
            "dataset_loader": dataloader
        }
    else:
        raise Exception(f"Failed to load dataset: {response.status_code}, {response.text}")


def send_model_request(model_filename):
    """
    Sends a request to the model service API to retrieve a serialized PyTorch model.

    The model is expected to be a `.pkl` file containing a `state_dict` (trained weights).
    The code assumes the model architecture is `TransformerAD` and will load the weights into it.

    Args:
        model_filename (str): Name of the model file (e.g., "my_model.pkl").

    Returns:
        torch.nn.Module: The loaded model in evaluation mode.

    Notes:
        - The model is loaded on CPU.
        - If the model class `TransformerAD` is not found or doesn't match, `strict=False` allows partial loading.
        - If the request fails, None is returned.

    Raises:
        Exception: If a request-related error occurs.
    """
    url = f"{BASE_URL_MODEL}/api/models/{model_filename}"

    try:
        response = requests.get(url)

        if response.status_code == 200:
            # The response content is the raw binary of the model (.pkl)
            model_bytes = response.content

            # Load state_dict into memory as bytes and convert to dictionary
            state_dict = torch.load(io.BytesIO(model_bytes), map_location="cpu")

            # Dynamically import the model architecture (must match the trained one)
            import sys,os
            sys.path.append(os.path.join(sys.path[0], 'common_libraries'))
            from privateer_ad.models import TransformerAD
            model = TransformerAD()

            # Load weights into model (non-strict in case of mismatches or missing keys)
            model.load_state_dict(state_dict, strict=False)
            model.eval()  # Ensure model is in inference mode

            return model
        else:
            print(f"Error {response.status_code}: {response.text}")
            return None

    except requests.exceptions.RequestException as e:
        print(f"Request error: {e}")
        return None
