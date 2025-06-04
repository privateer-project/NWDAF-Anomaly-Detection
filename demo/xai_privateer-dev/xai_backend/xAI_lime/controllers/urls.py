# === API Endpoint Configuration ===

# Base URL for the Dataset Service API
# This URL should point to the backend service responsible for handling dataset loading and preprocessing.
# Example: A separate Flask API running on port 5000 on local interface 127.0.0.2
BASE_URL_DATASET = "http://manage_datasets:5000"

# Base URL for the Model Service API
# This URL should point to the backend service responsible for serving trained models.
# Example: A separate Flask API running on port 5000 on local interface 127.0.0.3
BASE_URL_MODEL = "http://manage_models:5001"
