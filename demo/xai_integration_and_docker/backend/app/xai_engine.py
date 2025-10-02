"""
Core application object that manages model, dataset, and XAI services.
"""

from typing import Any, Dict, Optional

import json
import os
import torch
import numpy as np

from app.config import (
    logger,
    DATASET_PATH,
    XAI_FEATURE_NAMES
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
        self.output: Optional[torch.Tensor] = None
        self.input: Optional[str] = None
        # Caches for last computed results
        self.cached_shap_result: Optional[Dict[str, Any]] = None
        self.cached_lime_result: Optional[Dict[str, Any]] = None
        self.cached_classification_result: Optional[Dict[str, Any]] = None
        # Current file names
        self.current_model_filename: Optional[str] = None
        self.current_dataset_filename: Optional[str] = None

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

        # Check if model file exists
        if not os.path.exists(model_path):
            logger.error(f"Model file not found at {model_path}")
            raise FileNotFoundError(f"Model file not found at {model_path}")

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Create custom model config with 12 timesteps to match existing models
        model_config = ModelConfig()
        model_config.seq_len = 12  # Override default 77 with 12 to match existing models
        model = TransformerAD(model_config)

        try:
            # Try loading with weights_only=True first (safer)
            try:
                state_dict = torch.load(model_path, map_location=torch.device('cpu'), weights_only=True)
            except Exception as weights_only_error:
                # If that fails, try with weights_only=False (less safe but necessary for Opacus models)
                logger.warning(f"Weights-only load failed, falling back to weights_only=False: {weights_only_error}")
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
        
        data = loaded_dict["data"]
        data_tensor = torch.tensor(data, dtype=torch.float32)
        
        # Auto-detect data structure and reshape accordingly
        if data_tensor.dim() == 2:
            # 2D data: reshape to (-1, 12, 8) for time series format
            logger.info(f"Loading 2D dataset with shape {data_tensor.shape}, reshaping to time series format")
            data_tensor = data_tensor.view(-1, 12, 8)
            #data_tensor = data_tensor.view(-1, 77, 32)  # Commented: 77 timesteps, 32 features
        elif data_tensor.dim() == 3:
            # 3D data: already in time series format, keep as is
            logger.info(f"Loading 3D dataset with shape {data_tensor.shape}")
        else:
            # Unexpected format, try to reshape to time series format
            logger.warning(f"Unexpected data dimensions: {data_tensor.shape}, attempting to reshape to time series format")
            data_tensor = data_tensor.view(-1, 12, 8)
            #data_tensor = data_tensor.view(-1, 77, 32)  # Commented: 77 timesteps, 32 features
        
        logger.info(f"Final dataset shape: {data_tensor.shape}")
        return data_tensor

    def make_prediction(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            output = self.current_model(x)
            return output

    def initialize_services(self) -> None:
        if self.current_model is None or self.current_dataset is None:
            raise ValueError("Model and dataset must be loaded before initializing services")
        
        logger.info("Initializing XAI services...")
        self.shap_timeseries = ShapTimeSeries(self.current_model, self.current_dataset)
        first_batch = self.current_dataset[:1]
        output = self.make_prediction(first_batch)
        self.output = output
        self.lime_timeseries = LimeInTimeSeries(self.current_model, self.current_dataset, output)
        

    def load_and_initialize(self) -> None:
        logger.info("Loading model and dataset...")
        
        # Load model and set filename
        from app.config import MODEL_PATH
        self.current_model = self.load_model()
        self.current_model_filename = os.path.basename(MODEL_PATH)
        
        # Load dataset and set filename
        from app.config import DATASET_PATH
        self.current_dataset = self.load_dataset()
        self.current_dataset_filename = os.path.basename(DATASET_PATH)
        
        logger.info("Model and dataset loaded successfully")
        logger.info(f"Loaded model: {self.current_model_filename}")
        logger.info(f"Loaded dataset: {self.current_dataset_filename}")
        self.initialize_services()
        logger.info(f"Cache status - SHAP: {self.cached_shap_result is not None}, LIME: {self.cached_lime_result is not None}")

    def update_model(self, model_path: str) -> None:
        self.current_model = self.load_model(model_path)
        self.current_model_filename = os.path.basename(model_path)
        self.initialize_services()

    def update_dataset(self, dataset_path: str) -> None:
        self.current_dataset = self.load_dataset(dataset_path)
        self.current_dataset_filename = os.path.basename(dataset_path)
        self.initialize_services()

    def compute_lime_for_instance(self, instance: torch.Tensor) -> Dict[str, Any]:
        """Compute LIME explanation for a given instance tensor and return JSON-serializable content."""
        if self.lime_timeseries is None:
            raise ValueError("LIME service is not initialized")
        lime_result = self.lime_timeseries.lime_values_from_instance(instance)
        lime_values = lime_result["lime_values"]
        feature_names = lime_result["feature_names"]

        xai_feature_names = XAI_FEATURE_NAMES
        instance_flat = instance.flatten()

        feature_importance: Dict[str, float] = {}
        if lime_values:
            for feature_idx, importance in lime_values:
                if feature_idx < len(feature_names):
                    feat_name = feature_names[feature_idx]
                    main_feature = feat_name.rsplit('_', 1)[0] if '_' in feat_name else feat_name
                    if main_feature in xai_feature_names:
                        feature_importance[main_feature] = float(importance)

        for feature in xai_feature_names:
            if feature not in feature_importance:
                feature_importance[feature] = 0.0

        # Convert tensor to list for JSON serialization
        instance_flat_list = instance_flat.detach().cpu().numpy().tolist()
        
        # Convert output tensor to list for JSON serialization
        output_list = self.output.tolist() if hasattr(self.output, 'tolist') else self.output
        
        json_content = { 
            "message": "LIME explanation calculated successfully.",
            "sample": {name: float(value) for name, value in zip(xai_feature_names, instance_flat_list[:8])},
            "lime_values": feature_importance,
            "contribution": [{"feature": name, "value": value} for name, value in feature_importance.items()],
            "feature_names": xai_feature_names,
            "input_shape": list(instance.shape),
            "input": self.input,
            "output": output_list,
            "explanation_available": lime_result.get("explanation") is not None
        }
        return json_content

    def compute_shap_for_instance(self, instance: torch.Tensor) -> Dict[str, Any]:
        """Compute SHAP explanation for a given instance tensor and return JSON-serializable content."""
        if self.shap_timeseries is None:
            raise ValueError("SHAP service is not initialized")
        shap_result = self.shap_timeseries.shap_values_from_instance(instance)

        shap_values = shap_result["shap_values"]
        feature_names = XAI_FEATURE_NAMES
        instance_flat = instance.flatten()

        if len(shap_values.shape) > 1:
            shap_vals_flat = shap_values[0]
        else:
            shap_vals_flat = shap_values

        def safe_float(value: Any) -> float:
            if isinstance(value, (np.ndarray, torch.Tensor)):
                return float(value.item())
            return float(value)

        # Convert tensor to list for JSON serialization
        instance_flat_list = instance_flat.detach().cpu().numpy().tolist()
        
        # Convert output tensor to list for JSON serialization
        output_list = self.output.tolist() if hasattr(self.output, 'tolist') else self.output
        
        json_content = {
            "message": "SHAP values calculated successfully.",
            "sample": {name: safe_float(value) for name, value in zip(feature_names, instance_flat_list)},
            "shap_values": {name: safe_float(val) for name, val in zip(feature_names, shap_vals_flat)},
            "contribution": [{"feature": name, "value": safe_float(val)} for name, val in zip(feature_names, shap_vals_flat)],
            "feature_names": feature_names,
            "input": self.input,
            "output": output_list,
            "input_shape": list(instance.shape)
        }
        return json_content

    def compute_lime_from_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        instance = XAIApplication.load_tensor_from_json_data(data)
        self.input = self.get_input_data_string(data)
        
        # Store classification result (input and output only)
        output = self.make_prediction(instance)
        output_list = output.tolist() if hasattr(output, 'tolist') else output
        self.cached_classification_result = {
            "input": self.input,
            "output": output_list
        }
        
        return self.compute_lime_for_instance(instance)

    def compute_shap_from_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        instance = XAIApplication.load_tensor_from_json_data(data)
        self.input = self.get_input_data_string(data)
        
        # Store classification result (input and output only)
        output = self.make_prediction(instance)
        output_list = output.tolist() if hasattr(output, 'tolist') else output
        self.cached_classification_result = {
            "input": self.input,
            "output": output_list
        }
        
        return self.compute_shap_for_instance(instance)

    def get_input_data_string(self, data: Dict[str, Any]) -> str:
        """Return the input data as a string representation.
        
        Args:
            data: The input data dictionary sent by the user
            
        Returns:
            String representation of the input data
        """
        try:
            # Convert the data dictionary to a JSON string
            data_string = json.dumps(data, indent=2, ensure_ascii=False)
            return data_string
        except Exception as e:
            logger.error(f"Error converting data to string: {e}")
            return str(data)
    