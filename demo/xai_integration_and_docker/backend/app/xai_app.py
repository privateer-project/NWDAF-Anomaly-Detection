"""
Core application object that manages model, dataset, and XAI services.
"""

from typing import Any, Dict, Optional

import json
import torch

from app.config import (
    logger,
    DATASET_PATH,
)
from NewContent.privateer_ad.architectures.transformer_ad import TransformerAD
from NewContent.privateer_ad.config.settings import ModelConfig
from app.lime_timeseries import LimeInTimeSeries
from app.shap_timeseries import ShapTimeSeries


class XAIApplication:
    def __init__(self) -> None:
        self.current_model: Optional[torch.nn.Module] = None
        self.current_dataset: Optional[torch.Tensor] = None
        self.shap_timeseries: Optional[ShapTimeSeries] = None
        self.lime_timeseries: Optional[LimeInTimeSeries] = None

    @staticmethod
    def load_tensor_from_json_data(data: Dict[str, Any], device: Optional[Any] = None) -> torch.Tensor:
        tensor = torch.tensor(data['data'])
        if 'dtype' in data:
            dtype_map = {
                'torch.float32': torch.float32,
                'torch.float64': torch.float64,
                'torch.int32': torch.int32,
                'torch.int64': torch.int64,
                'torch.bool': torch.bool
            }
            tensor = tensor.to(dtype=dtype_map.get(data['dtype'], torch.float32))
        if 'shape' in data:
            tensor = tensor.view(data['shape'])
        target_device = device if device else data.get('device', 'cpu')
        if target_device != 'cpu' and torch.cuda.is_available():
            tensor = tensor.to(target_device)
        if data.get('requires_grad', False):
            tensor.requires_grad_(True)
        return tensor

    def load_model(self, model_path: Optional[str] = None) -> torch.nn.Module:
        if model_path is None:
            from app.config import MODEL_PATH
            model_path = MODEL_PATH

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_config = ModelConfig()
        model = TransformerAD(model_config)

        try:
            state_dict = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)

            if isinstance(state_dict, dict) and 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']

            if hasattr(state_dict, 'state_dict'):
                state_dict = state_dict.state_dict()
            elif not isinstance(state_dict, dict):
                logger.warning(f"Unexpected state_dict type: {type(state_dict)}")
                if hasattr(state_dict, '__dict__'):
                    state_dict = dict(state_dict.__dict__)
                else:
                    raise ValueError(f"Cannot convert state_dict of type {type(state_dict)} to dict")

            cleaned_state_dict = {}
            for key, value in state_dict.items():
                clean_key = key
                for prefix in ['_module.', 'module.']:
                    if clean_key.startswith(prefix):
                        clean_key = clean_key[len(prefix):]
                cleaned_state_dict[clean_key] = value

            model.load_state_dict(cleaned_state_dict, strict=False)
            model = model.to(device)
            model.eval()

        except Exception as e:
            logger.error(f"Error loading model from {model_path}: {str(e)}")
            raise

        return model

    def load_dataset(self, dataset_path: Optional[str] = None) -> torch.Tensor:
        if dataset_path is None:
            dataset_path = DATASET_PATH
        with open(dataset_path, 'r') as f:
            loaded_dict = json.load(f)
        data_tensor = torch.tensor(loaded_dict["data"], dtype=torch.float32)
        data_tensor = data_tensor.view(-1, 12, 8)
        return data_tensor

    def make_prediction(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            output = self.current_model(x)
            return output

    def initialize_services(self) -> None:
        if self.current_model is None or self.current_dataset is None:
            raise ValueError("Model and dataset must be loaded before initializing services")
        self.shap_timeseries = ShapTimeSeries(self.current_model, self.current_dataset)
        first_batch = self.current_dataset[:1]
        output = self.make_prediction(first_batch)
        self.lime_timeseries = LimeInTimeSeries(self.current_model, self.current_dataset, output)

    def load_and_initialize(self) -> None:
        self.current_model = self.load_model()
        self.current_dataset = self.load_dataset()
        self.initialize_services()

    def update_model(self, model_path: str) -> None:
        self.current_model = self.load_model(model_path)
        self.initialize_services()

    def update_dataset(self, dataset_path: str) -> None:
        self.current_dataset = self.load_dataset(dataset_path)
        self.initialize_services()