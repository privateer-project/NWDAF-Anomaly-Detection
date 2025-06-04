import shap
import torch

from xAI_shap.controllers.predictions import get_feature_names

class ShapTimeSeries:
    def __init__(self,model, dataset):
        self.model = model
        self.initial_dataset = dataset
        self.explainer = None

        self.__init_shap()
    
    def __init_shap(self):
        # Reshape input to 2D (instances, flattened features) for SHAP KernelExplainer
        original_shape = self.initial_dataset.shape
        shape_kernel_explainer = (original_shape[0], original_shape[1] * original_shape[2])
        input_kernel_explainer = self.initial_dataset.detach().numpy().reshape(shape_kernel_explainer)

        # Initialize SHAP KernelExplainer with model prediction function
        self.explainer = shap.KernelExplainer(self.__model_predict_fn(), input_kernel_explainer)

    def __model_predict_fn(self):
        """
        Wraps the PyTorch model into a function that can be used by SHAP's KernelExplainer.

        This function:
        - Reshapes the input into the expected 3D shape (batch_size, sequence_length, num_features)
        - Performs inference using the model without computing gradients
        - Computes the mean squared error between the input and model output per instance
        - Returns a NumPy array with the computed values for SHAP to interpret

        Returns:
            A function that takes a NumPy array and returns model predictions as NumPy
        """

        def predict(x):
            x_tensor = torch.tensor(x, dtype=torch.float32)
            x_tensor = x_tensor.reshape((-1, 12, 8))  # Reshape to match model input
            with torch.no_grad():
                output = self.model(x_tensor)
                mse_output = ((x_tensor - output) ** 2).mean(axis=1).mean(axis=1)
                return mse_output.detach().numpy()

        return predict
    
    def shap_values_from_instance(self,instance):
        # Compute SHAP values for the selected instance
        shap_values = self.explainer.shap_values(instance)

        # Handle possible multi-output models (list of arrays)
        if isinstance(shap_values, list):
            print("[INFO] KernelExplainer returned list, using shap_values[0]")
            shap_values = shap_values[0]

        # Return all relevant objects
        return {
            "shap_values": shap_values,
            "x": instance,
            "explainer": self.explainer,
        }

class ShapFeaturesInTimeSeries:
    def __init__(self,model, dataset):
        self.model = model
        self.initial_dataset = dataset
        self.explainer = None

        self.__init_shap()
    
    def __init_shap(self):
        # Reshape input to 2D (instances, flattened features) for SHAP KernelExplainer
        original_shape = self.initial_dataset.shape
        shape_kernel_explainer = (original_shape[0], original_shape[1] * original_shape[2])
        input_kernel_explainer = self.initial_dataset.detach().numpy().reshape(shape_kernel_explainer)

        # Initialize SHAP KernelExplainer with model prediction function
        self.explainer = shap.KernelExplainer(self.__model_predict_fn(), input_kernel_explainer)

    def __model_predict_fn(self):
        """
        Wraps the PyTorch model into a function that can be used by SHAP's KernelExplainer.

        This function:
        - Reshapes the input into the expected 3D shape (batch_size, sequence_length, num_features)
        - Performs inference using the model without computing gradients
        - Computes the mean squared error between the input and model output per instance
        - Returns a NumPy array with the computed values for SHAP to interpret

        Returns:
            A function that takes a NumPy array and returns model predictions as NumPy
        """

        def predict(x):
            x_tensor = torch.tensor(x, dtype=torch.float32)
            x_tensor = x_tensor.reshape((-1, 12, 8))  # Reshape to match model input
            with torch.no_grad():
                output = self.model(x_tensor)
                mse_output = ((x_tensor - output) ** 2).mean(axis=1)
                return mse_output.detach().numpy()

        return predict
    
    def shap_values_from_instance(self,instance):
        # Compute SHAP values for the selected instance
        shap_values = self.explainer.shap_values(instance)

        # Handle possible multi-output models (list of arrays)
        if isinstance(shap_values, list):
            print("[INFO] KernelExplainer returned list, using shap_values[0]")
            shap_values = shap_values[0]

        # Return all relevant objects
        return {
            "shap_values": shap_values,
            "x": instance,
            "explainer": self.explainer,
        }

class ShapWindowsInTimeSeries:
    def __init__(self,model, dataset):
        self.model = model
        self.initial_dataset = dataset
        self.explainer = None

        self.__init_shap()
    
    def __init_shap(self):
        # Reshape input to 2D (instances, flattened features) for SHAP KernelExplainer
        original_shape = self.initial_dataset.shape
        shape_kernel_explainer = (original_shape[0], original_shape[1] * original_shape[2])
        input_kernel_explainer = self.initial_dataset.detach().numpy().reshape(shape_kernel_explainer)

        # Initialize SHAP KernelExplainer with model prediction function
        self.explainer = shap.KernelExplainer(self.__model_predict_fn(), input_kernel_explainer)

    def __model_predict_fn(self):
        """
        Wraps the PyTorch model into a function that can be used by SHAP's KernelExplainer.

        This function:
        - Reshapes the input into the expected 3D shape (batch_size, sequence_length, num_features)
        - Performs inference using the model without computing gradients
        - Computes the mean squared error between the input and model output per instance
        - Returns a NumPy array with the computed values for SHAP to interpret

        Returns:
            A function that takes a NumPy array and returns model predictions as NumPy
        """

        def predict(x):
            x_tensor = torch.tensor(x, dtype=torch.float32)
            x_tensor = x_tensor.reshape((-1, 12, 8))  # Reshape to match model input
            with torch.no_grad():
                output = self.model(x_tensor)
                mse_output = ((x_tensor - output) ** 2).mean(axis=2)
                return mse_output.detach().numpy()

        return predict
    
    def shap_values_from_instance(self,instance):
        # Compute SHAP values for the selected instance
        shap_values = self.explainer.shap_values(instance)

        # Handle possible multi-output models (list of arrays)
        if isinstance(shap_values, list):
            print("[INFO] KernelExplainer returned list, using shap_values[0]")
            shap_values = shap_values[0]

        # Return all relevant objects
        return {
            "shap_values": shap_values,
            "x": instance,
            "explainer": self.explainer
        }
