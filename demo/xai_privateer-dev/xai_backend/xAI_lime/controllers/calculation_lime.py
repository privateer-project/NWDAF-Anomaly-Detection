import numpy as np
import torch
from lime import lime_tabular


def model_predict_fn(model):
    """
    Returns a function that reshapes the input and performs prediction on tabular input.
    This function is used by LIME to simulate model predictions.

    Parameters:
        model (torch.nn.Module): The trained PyTorch model.

    Returns:
        function: A prediction function compatible with LIME.
    """

    def predict(x):
        # Convert to tensor and reshape to original format: (batch_size, sequence_length, features)
        x_tensor = torch.tensor(x, dtype=torch.float32)
        x_tensor = x_tensor.reshape((-1, 12, 8))  # expected shape: (N, 12, 8)

        # Run the model in inference mode
        with torch.no_grad():
            output = model(x_tensor)
            # Compute the mean squared error per sample
            mse_output = ((x_tensor - output) ** 2).mean(dim=1).mean(dim=1)
            return mse_output.detach().numpy()

    return predict


def main_calculation_lime(x, model, output, mode='regression', num_features=10, sample_index=0):
    """
    Main function to calculate LIME explanations for one sample in a batch.

    Parameters:
        x (Tensor): Input tensor with shape (batch_size, 12, 8).
        model (torch.nn.Module): The model to be explained.
        output (Tensor): Output from the model.
        mode (str): Either 'regression' or 'classification'.
        num_features (int): Number of top features to include in the explanation.
        sample_index (int): Index of the sample to explain from batch.

    Returns:
        dict: Explanation result including explanation object, sample values, and more.
    """
    print('Running LIME explanation...')

    print(f"Input tensor shape (x): {x.shape}")
    print(f"Model output shape: {output.shape}")

    # Define base feature names
    feature_columns = [
        'dl_bitrate', 'ul_bitrate',
        'cell_x_dl_retx', 'cell_x_dl_tx',
        'cell_x_ul_retx', 'cell_x_ul_tx',
        'ul_total_bytes_non_incr', 'dl_total_bytes_non_incr'
    ]

    # Generate time-based feature names, e.g., dl_bitrate_0, dl_bitrate_1, ..., dl_bitrate_11
    feature_names = []
    for i in range(12):
        feature_names += [f"{item}_{i}" for item in feature_columns]

    print("Generated feature names:")
    print(feature_names)

    # Flatten the input to transform sequential data into tabular format
    original_shape = x.shape
    flattened_shape = (original_shape[0], original_shape[1] * original_shape[2])
    input_kernel_explainer = x.detach().numpy().reshape(flattened_shape)

    # Create LIME explainer for tabular data
    explainer = lime_tabular.LimeTabularExplainer(
        training_data=input_kernel_explainer,
        feature_names=feature_names,
        mode=mode,
        discretize_continuous=False
    )

    # Select the sample to explain
    instance = input_kernel_explainer[sample_index]

    # Generate explanation for the selected instance
    explanation = explainer.explain_instance(
        data_row=instance,
        predict_fn=model_predict_fn(model),
        num_features=num_features
    )

    print("LIME Explanation Results:")
    print("Feature contributions (format: feature => contribution):")
    for feature, contribution in explanation.as_list():
        print(f"{feature}: {contribution:+.4f}")
    print()

    print("Feature values for the explained instance:")
    for name, value in zip(feature_names, instance):
        print(f"{name}: {value:.4f}")
    print()

    # Return structured data to be used for plotting, reports, or API output
    return {
        "explanation": explanation,  # LIME explanation object
        "x_tab": input_kernel_explainer,  # Flattened input used for the explainer
        "sample": instance.tolist(),  # Values of the explained instance
        "feature_names": feature_names,  # Feature name list
        "predict_fn": model_predict_fn(model),  # Prediction function used by LIME
        "model_output": output[sample_index].detach().cpu().numpy().tolist()
    }

def main_calculation_lime_from_tensor(x, model, output, instance_X_test, mode='regression', num_features=8 ):
    """
    Main function to calculate LIME explanations for one sample in a batch.

    Parameters:
        x (Tensor): Input tensor with shape (batch_size, 12, 8).
        model (torch.nn.Module): The model to be explained.
        output (Tensor): Output from the model.
        mode (str): Either 'regression' or 'classification'.
        num_features (int): Number of top features to include in the explanation.
        sample_index (int): Index of the sample to explain from batch.

    Returns:
        dict: Explanation result including explanation object, sample values, and more.
    """
    print('Running LIME explanation...')

    print(f"Input tensor shape (x): {x.shape}")
    print(f"Model output shape: {output.shape}")

    # Define base feature names
    feature_columns = [
        'dl_bitrate', 'ul_bitrate',
        'cell_x_dl_retx', 'cell_x_dl_tx',
        'cell_x_ul_retx', 'cell_x_ul_tx',
        'ul_total_bytes_non_incr', 'dl_total_bytes_non_incr'
    ]

    # Generate time-based feature names, e.g., dl_bitrate_0, dl_bitrate_1, ..., dl_bitrate_11
    feature_names = []
    for i in range(12):
        feature_names += [f"{item}_{i}" for item in feature_columns]

    print("Generated feature names:")
    print(feature_names)

    # Flatten the input to transform sequential data into tabular format
    original_shape = x.shape
    flattened_shape = (original_shape[0], original_shape[1] * original_shape[2])
    input_kernel_explainer = x.detach().numpy().reshape(flattened_shape)

    # Create LIME explainer for tabular data
    explainer = lime_tabular.LimeTabularExplainer(
        training_data=input_kernel_explainer,
        feature_names=feature_names,
        mode=mode,
        discretize_continuous=False
    )

    # Select the sample to explain
    # instance = tensor
    original_shape_instance = instance_X_test.shape
    shape_kernel_explainer_instance = (original_shape_instance[0], original_shape_instance[1] * original_shape_instance[2])
    input_kernel_explainer_instance = instance_X_test.detach().numpy().reshape(shape_kernel_explainer_instance)

    # Generate explanation for the selected instance
    explanation = explainer.explain_instance(
        data_row=input_kernel_explainer_instance,
        predict_fn=model_predict_fn(model),
        num_features=num_features
    )

    print("LIME Explanation Results:")
    print("Feature contributions (format: feature => contribution):")
    for feature, contribution in explanation.as_list():
        print(f"{feature}: {contribution:+.4f}")
    print()

    print("Feature values for the explained instance:")
    for name, value in zip(feature_names, input_kernel_explainer_instance):
        print(f"{name}: {value:.4f}")
    print()

    # Return structured data to be used for plotting, reports, or API output
    return {
        "explanation": explanation,  # LIME explanation object
        "x_tab": input_kernel_explainer,  # Flattened input used for the explainer
        "sample": input_kernel_explainer_instance.tolist(),  # Values of the explained instance
        "feature_names": feature_names,  # Feature name list
        "predict_fn": model_predict_fn(model),  # Prediction function used by LIME
        "model_output": output.detach().cpu().numpy().tolist()
    }
