import torch

def make_prediction(x, model):
    """
    Perform a forward pass through the model using the given input tensor `x`.

    This function is used during SHAP calculations or standard inference,
    and ensures that the model runs in evaluation mode without tracking gradients.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, num_features).
        model (torch.nn.Module): The PyTorch model to evaluate.

    Returns:
        torch.Tensor: The model's output tensor, same shape as `x` if autoencoder.
    """
    model.eval()  # Set the model to evaluation mode (important for layers like dropout or batchnorm)
    with torch.no_grad():  # Disable gradient computation (for memory and performance optimization)
        output = model(x)  # Forward pass
    return output

def get_feature_names():
    """
    Generate the list of feature names for each time step.

    This function returns a list of descriptive feature names in the format
    "<feature>_<time_step>" for 12 time steps. It avoids relying on any
    external configuration or metadata classes.

    Returns:
        list: A list of strings, each representing a feature at a specific time step.
              Example: ['dl_bitrate_0', 'ul_bitrate_0', ..., 'dl_total_bytes_non_incr_11']
    """
    # Base feature names, assumed to repeat across each time step
    feature_columns = [
        'dl_bitrate', 'ul_bitrate',
        'cell_x_dl_retx', 'cell_x_dl_tx',
        'cell_x_ul_retx', 'cell_x_ul_tx',
        'ul_total_bytes_non_incr', 'dl_total_bytes_non_incr'
    ]

    # Construct feature names with time step suffixes (e.g., 'dl_bitrate_0', ..., 'dl_bitrate_11')
    feature_names = []
    for i in range(12):  # Loop through 12 time steps
        feature_names += [f"{col}_{i}" for col in feature_columns]

    return feature_names
