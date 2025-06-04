import shap
import torch

from xAI_shap.controllers.predictions import get_feature_names


def model_predict_fn(model):
    """
    Wraps the PyTorch model into a function that can be used by SHAP's KernelExplainer.

    This function:
    - Reshapes the input into the expected 3D shape (batch_size, sequence_length, num_features)
    - Performs inference using the model without computing gradients
    - Computes the mean squared error between the input and model output per instance
    - Returns a NumPy array with the computed values for SHAP to interpret

    Args:
        model: Trained PyTorch model

    Returns:
        A function that takes a NumPy array and returns model predictions as NumPy
    """

    def predict(x):
        x_tensor = torch.tensor(x, dtype=torch.float32)
        x_tensor = x_tensor.reshape((-1, 12, 8))  # Reshape to match model input
        with torch.no_grad():
            output = model(x_tensor)
            mse_output = ((x_tensor - output) ** 2).mean(axis=1).mean(axis=1)
            return mse_output.detach().numpy()

    return predict


def main_calculation_shap(x, output, model, instance_X_test):
    """
    Main function to compute SHAP values for a given input tensor and model using SHAP's KernelExplainer.

    This function:
    - Prepares the input data by reshaping it to 2D for KernelExplainer
    - Initializes the SHAP KernelExplainer with the wrapped model prediction function
    - Computes SHAP values for the specified instance index
    - Handles multi-output SHAP value lists by selecting the first output
    - Loads the corresponding feature names

    Args:
        x (torch.Tensor): Input tensor with shape (n_instances, sequence_length, num_features)
        output (torch.Tensor): Model output (unused here, but may be used in future extensions)
        model (torch.nn.Module): Trained PyTorch model to explain
        instance_X_test (int): Index of the instance to explain

    Returns:
        dict: A dictionary with keys:
            - "shap_values": SHAP values for the selected instance
            - "x": The original input tensor
            - "explainer": The SHAP explainer instance
            - "feature_names": List of feature names corresponding to the flattened input
    """
    print(f"x shape: {x.shape}")
    print(f"output shape: {output.shape}")
    print("Creating KernelExplainer...")

    # Reshape input to 2D (instances, flattened features) for SHAP KernelExplainer
    original_shape = x.shape
    shape_kernel_explainer = (original_shape[0], original_shape[1] * original_shape[2])
    input_kernel_explainer = x.detach().numpy().reshape(shape_kernel_explainer)

    # Initialize SHAP KernelExplainer with model prediction function
    explainer = shap.KernelExplainer(model_predict_fn(model), input_kernel_explainer)

    # Compute SHAP values for the selected instance
    shap_values = explainer.shap_values(input_kernel_explainer[instance_X_test])

    # Handle possible multi-output models (list of arrays)
    if isinstance(shap_values, list):
        print("[INFO] KernelExplainer returned list, using shap_values[0]")
        shap_values = shap_values[0]

    # Retrieve feature names for interpretability
    feature_names = get_feature_names()

    # Return all relevant objects
    return {
        "shap_values": shap_values,
        "x": x,
        "explainer": explainer,
        "feature_names": feature_names,
        "model_output": output[instance_X_test].detach().cpu().numpy().tolist()
    }

def main_calculation_shap_from_tensor(x, output, model, instance_X_test):
    """
    Main function to compute SHAP values for a given input tensor and model using SHAP's KernelExplainer.

    This function:
    - Prepares the input data by reshaping it to 2D for KernelExplainer
    - Initializes the SHAP KernelExplainer with the wrapped model prediction function
    - Computes SHAP values for the specified instance index
    - Handles multi-output SHAP value lists by selecting the first output
    - Loads the corresponding feature names

    Args:
        x (torch.Tensor): Input tensor with shape (n_instances, sequence_length, num_features)
        output (torch.Tensor): Model output (unused here, but may be used in future extensions)
        model (torch.nn.Module): Trained PyTorch model to explain
        instance_X_test (int): Index of the instance to explain

    Returns:
        dict: A dictionary with keys:
            - "shap_values": SHAP values for the selected instance
            - "x": The original input tensor
            - "explainer": The SHAP explainer instance
            - "feature_names": List of feature names corresponding to the flattened input
    """
    print(f"x shape: {x.shape}")
    print(f"output shape: {output.shape}")
    print("Creating KernelExplainer...")

    # Reshape input to 2D (instances, flattened features) for SHAP KernelExplainer
    original_shape = x.shape
    shape_kernel_explainer = (original_shape[0], original_shape[1] * original_shape[2])
    input_kernel_explainer = x.detach().numpy().reshape(shape_kernel_explainer)
    

    # Initialize SHAP KernelExplainer with model prediction function
    explainer = shap.KernelExplainer(model_predict_fn(model), input_kernel_explainer)

    original_shape_instance = instance_X_test.shape
    shape_kernel_explainer_instance = (original_shape_instance[0], original_shape_instance[1] * original_shape_instance[2])
    input_kernel_explainer_instance = instance_X_test.detach().numpy().reshape(shape_kernel_explainer_instance)
    # Compute SHAP values for the selected instance
    shap_values = explainer.shap_values(input_kernel_explainer_instance)

    # Handle possible multi-output models (list of arrays)
    if isinstance(shap_values, list):
        print("[INFO] KernelExplainer returned list, using shap_values[0]")
        shap_values = shap_values[0]

    # Retrieve feature names for interpretability
    feature_names = get_feature_names()

    # Return all relevant objects
    return {
        "shap_values": shap_values,
        "x": x,
        "explainer": explainer,
        "feature_names": feature_names,
        "model_output": output.detach().cpu().numpy().tolist()
    }
