import os
import torch
from xAI_lime.controllers.request_orders import send_dataset_request, send_model_request
from xAI_lime.controllers.calculation_lime import main_calculation_lime, main_calculation_lime_from_tensor
from xAI_lime.controllers.graphics_lime import main_graphics_lime
from xAI_lime.controllers.predictions import make_prediction, get_feature_names
from xAI_lime.controllers.generate_lime_reports import main_generate_all_lime_reports

class LimeService:
    """
    Service class to handle LIME explanation logic.
    This includes loading data and models, performing predictions,
    computing LIME explanations, generating graphics, and exporting reports.
    """

    def __init__(self):
        self.model = None                      # Loaded machine learning model
        self.train_loader = None              # DataLoader with the dataset
        self.feature_names = None             # List of feature names used in the model
        self.x = None                         # Input data (features) for LIME
        self.output = None                    # Model predictions (reconstructions)
        self.result_lime = None               # Result of the LIME explanation
        self.name_dataset = None              # Dataset name used for current run
        self.name_model = None                # Model name used for current run
        self.instance_index = 0               # Index of the instance to be explained

    def load_resources(self, name_dataset: str, name_model: str, batch_size: int = 32):
        """
        Load dataset and model from external services.

        Args:
            name_dataset (str): Name or path of the dataset.
            name_model (str): Name or path of the trained model.
            batch_size (int): Batch size used to load the dataset.
        """
        self.name_dataset = name_dataset
        self.name_model = name_model

        dataset_response = send_dataset_request(name_dataset, batch_size=batch_size)
        self.train_loader = dataset_response["dataset_loader"]
        self.model = send_model_request(name_model)
        self.feature_names = get_feature_names()

    def run_predictions(self):
        """
        Run the model on the first batch of the dataset to prepare inputs for LIME.

        Raises:
            ValueError: If model or dataset hasn't been loaded properly.
        """
        if not self.model or not self.train_loader:
            raise ValueError("Model or dataset not loaded. Call load_resources first.")

        try:
            first_batch = next(iter(self.train_loader))
            self.x = first_batch[0]['encoder_cont']      # Extract input features
            self.output = make_prediction(self.x, self.model)
        except StopIteration:
            raise ValueError("The DataLoader is empty.")

    def calculate_lime(self, mode='regression'):
        """
        Calculate LIME explanation for a specific instance from the dataset.

        Args:
            mode (str): Mode of LIME explanation. Can be 'regression' or 'classification'.

        Raises:
            ValueError: If predictions or model are not yet initialized.
        """
        if self.x is None or self.output is None or self.model is None:
            raise ValueError("Missing predictions or model. Call run_predictions first.")

        self.result_lime = main_calculation_lime(
            x=self.x,
            model=self.model,
            output=self.output,
            mode=mode,
            sample_index=self.instance_index
        )
    
    def calculate_lime_from_tensor(self,tensor):
        """
        Calculates SHAP values for the selected instance using KernelExplainer.

        Raises:
            ValueError: If prediction outputs or inputs are missing
        """
        if self.x is None or self.output is None or self.model is None:
            raise ValueError("Missing prediction results. Please call run_predictions first.")

        # Perform SHAP value calculation
        self.result_lime = main_calculation_lime_from_tensor(
            x=self.x,
            output=self.output,
            model=self.model,
            instance_X_test=tensor
        )

    def generate_graphics(self):
        """
        Generate explanation graphics based on the previously calculated LIME results.

        Raises:
            ValueError: If LIME explanation hasn't been computed yet.
        """
        if self.result_lime is None:
            raise ValueError("LIME explanation not computed. Call calculate_lime first.")

        main_graphics_lime(self.result_lime, self.instance_index)

    def list_features(self):
        """
        Return the list of feature names used in the model.

        Returns:
            list[str]: Feature names.

        Raises:
            ValueError: If features were not loaded.
        """
        if self.feature_names is None:
            raise ValueError("Feature names not loaded. Call load_resources first.")
        return self.feature_names

    def get_graphic_names(self):
        """
        List generated graphic filenames for the selected instance.

        Returns:
            list[str]: List of matching PNG file names.

        Raises:
            ValueError: If the dataset was not loaded yet.
        """
        if self.name_dataset is None:
            raise ValueError("Dataset not loaded. Call load_resources first.")

        graphics_dir = 'graphics'
        matched_files = [
            f for f in os.listdir(graphics_dir)
            if f.startswith(f"lime_summary_{self.instance_index}_") and f.endswith(".png")
        ]
        return matched_files

    def generate_reports(self):
        """
        Generate reports (CSV, JSON, HTML, etc.) for the LIME explanation.

        Raises:
            ValueError: If LIME explanation has not been calculated yet.
        """
        if self.result_lime is None:
            raise ValueError("LIME explanation not computed. Call calculate_lime first.")

        main_generate_all_lime_reports(self.result_lime, self.instance_index)
    
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
