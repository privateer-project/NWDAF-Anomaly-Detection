"""
Configuration, constants, logging, and helpers for the XAI backend.
"""

import logging
import os
import sys
from typing import List

# === Logging ===
# Ensure the log file is always written under the backend directory
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
LOG_FILE = os.path.join(BASE_DIR, 'xai_api.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# === Paths and Directories ===
MODELS_DIR = os.path.join(BASE_DIR, 'models')
DATASETS_DIR = os.path.join(BASE_DIR, 'datasets')

DATASET_PATH = os.path.join(DATASETS_DIR, 'startup_dataset.json')
EXAMPLE_INSTANCE = os.path.join(DATASETS_DIR, 'one_instance.json')
MODEL_PATH = os.path.join(MODELS_DIR, 'model.pt')

# Ensure directories exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATASETS_DIR, exist_ok=True)

# Ensure NewContent is importable
new_content_dir = os.path.join(BASE_DIR, 'NewContent')
if new_content_dir not in sys.path:
    sys.path.insert(0, new_content_dir)

# === Allowed file extensions ===
ALLOWED_MODEL_EXTENSIONS = {'pt'}
ALLOWED_DATASET_EXTENSIONS = {'json'}

# === Feature Name Constants ===
XAI_FEATURE_NAMES: List[str] = [
    "dl_bitrate",
    "dl_retx",
    "dl_tx",
    "ul_bitrate",
    "ul_mcs",
    "ul_retx",
    "ul_tx",
    "turbo_decoder_avg",
]


def allowed_model_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_MODEL_EXTENSIONS

def allowed_dataset_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_DATASET_EXTENSIONS