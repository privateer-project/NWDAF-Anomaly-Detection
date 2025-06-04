import os
import json
import torch
from xAI_shap.controllers.request_orders import send_dataset_request, send_model_request
from xAI_shap.controllers.calculation_shap import main_calculation_shap, main_calculation_shap_from_tensor
from xAI_shap.controllers.graphics_shap import main_graphics_shap
from xAI_shap.controllers.generate_shap_reports import main_generate_all_shap_reports
from xAI_shap.controllers.predictions import make_prediction, get_feature_names


class ShapService:
    """
    Service class to manage SHAP explainability operations in a stateful manner.

    Responsibilities:
    - Load dataset and model from remote sources
    - Execute predictions on input data
    - Compute SHAP values using KernelExplainer
    - Generate plots and reports to explain predictions
    - Provide access to generated feature and plot names

    This avoids the use of global variables by encapsulating all related state.
    """

    def __init__(self):
        # State attributes to persist between operations
        self.model = None
        self.train_loader = None
        self.feature_names = None
        self.x = None
        self.output = None
        self.result_shap = None
        self.name_dataset = None
        self.name_model = None
        self.instance_index = 0  # Default index for instance explanation

    def load_resources(self, name_dataset: str, name_model: str, batch_size: int = 32):
        """
        Loads the dataset and model from external sources.
        Also retrieves the list of feature names.

        Args:
            name_dataset (str): Name of the dataset to load
            name_model (str): Name of the model to load
            batch_size (int): Batch size for DataLoader
        """
        self.name_dataset = name_dataset
        self.name_model = name_model

        # Request dataset loader from external service
        dataset_response = send_dataset_request(name_dataset, batch_size=batch_size)
        self.train_loader = dataset_response["dataset_loader"]

        # Request model from external service
        self.model = send_model_request(name_model)

        # Load feature names for explanation
        self.feature_names = get_feature_names()

    def run_predictions(self):
        """
        Runs model inference on the first batch of data and stores the input/output tensors.

        Raises:
            ValueError: If model or dataset has not been loaded or if the loader is empty
        """
        if not self.model or not self.train_loader:
            raise ValueError("Model or dataset not loaded. Please call load_resources first.")

        try:
            # Fetch the first batch from the data loader
            first_batch = next(iter(self.train_loader))
            self.x = first_batch[0]['encoder_cont']  # Extract encoder continuous features
            self.output = make_prediction(self.x, self.model)  # Run model prediction
        except StopIteration:
            raise ValueError("The DataLoader is empty.")

    def calculate_shap(self):
        """
        Calculates SHAP values for the selected instance using KernelExplainer.

        Raises:
            ValueError: If prediction outputs or inputs are missing
        """
        if self.x is None or self.output is None or self.model is None:
            raise ValueError("Missing prediction results. Please call run_predictions first.")

        # Perform SHAP value calculation
        self.result_shap = main_calculation_shap(
            x=self.x,
            output=self.output,
            model=self.model,
            instance_X_test=self.instance_index
        )
    
    def calculate_shap_from_tensor(self,tensor):
        """
        Calculates SHAP values for the selected instance using KernelExplainer.

        Raises:
            ValueError: If prediction outputs or inputs are missing
        """
        if self.x is None or self.output is None or self.model is None:
            raise ValueError("Missing prediction results. Please call run_predictions first.")

        # Perform SHAP value calculation
        self.result_shap = main_calculation_shap_from_tensor(
            x=self.x,
            output=self.output,
            model=self.model,
            instance_X_test=tensor
        )

    def generate_graphics(self):
        """
        Generates SHAP visualizations for the selected instance.

        Raises:
            ValueError: If SHAP values haven't been computed yet
        """
        if self.result_shap is None:
            raise ValueError("SHAP values not calculated. Please call calculate_shap first.")

        # Generate visual explanation plots
        main_graphics_shap(self.result_shap, self.instance_index)

    def generate_reports(self):
        """
        Generates full SHAP-based reports for the selected instance.

        Raises:
            ValueError: If SHAP values haven't been computed yet
        """
        if self.result_shap is None:
            raise ValueError("SHAP values not calculated. Please call calculate_shap first.")

        main_generate_all_shap_reports(self.result_shap, self.instance_index)

    def list_features(self):
        """
        Returns the list of feature names used in the model.

        Raises:
            ValueError: If features have not been loaded yet

        Returns:
            List[str]: List of feature names
        """
        if self.feature_names is None:
            raise ValueError("Feature names not available. Please call load_resources first.")

        return self.feature_names

    def get_graphic_names(self):
        """
        Returns the list of SHAP summary graphics generated for the selected instance.

        Raises:
            ValueError: If dataset name has not been set

        Returns:
            List[str]: List of filenames for the generated plots
        """
        if self.name_dataset is None:
            raise ValueError("Dataset not specified. Please call load_resources first.")

        graphics_dir = 'graphics'
        matched_files = [f for f in os.listdir(graphics_dir)
                         if f.startswith(f"shap_summary_{self.instance_index}_") and f.endswith(".png")]
        return matched_files
    
    def load_tensor_from_json_data(self, data, device=None):
        """Load tensor from JSON file with metadata restoration"""
        # Create tensor from data
        tensor = torch.tensor(data['data'])
        
        # Restore dtype
        if 'dtype' in data:
            dtype_map = {
                'torch.float32': torch.float32,
                'torch.float64': torch.float64,
                'torch.int32': torch.int32,
                'torch.int64': torch.int64,
                'torch.bool': torch.bool
            }
            tensor = tensor.to(dtype=dtype_map.get(data['dtype'], torch.float32))
        
        # Restore shape
        tensor = tensor.view(data['shape'])
        
        # Restore device
        target_device = device if device else data.get('device', 'cpu')
        if target_device != 'cpu' and torch.cuda.is_available():
            tensor = tensor.to(target_device)
        
        # Restore requires_grad
        if data.get('requires_grad', False):
            tensor.requires_grad_(True)
        
        return tensor
