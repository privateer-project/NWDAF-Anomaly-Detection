import torch

def make_prediction(x, model):
    """
    Perform a forward pass through the model using the given input tensor `x`.

    This function is typically used for making predictions with an autoencoder model,
    including during SHAP or LIME explanation processes. It ensures that the model
    is in evaluation mode and does not compute gradients (which saves memory and speeds up execution).

    Args:
        x (torch.Tensor): Input tensor with shape (batch_size, sequence_length, num_features),
                          typically (N, 12, 8) for this project.
        model (torch.nn.Module): The PyTorch model to evaluate.

    Returns:
        torch.Tensor: The output tensor from the model, expected to have the same shape as `x`
                      in case of a reconstruction model (e.g., autoencoder).
    """
    model.eval()  # Disable dropout, batchnorm, etc.
    #with torch.no_grad():  # Disable gradient tracking
    output = model(x)
    return output


def get_feature_names():
    """
    Generate the list of feature names used in the LIME and SHAP explanations.

    This function defines the base features and dynamically generates feature names
    for each time step (e.g., from timestep 0 to 11) by appending the timestep index.

    Example:
        For base feature "dl_bitrate", this function generates:
        ['dl_bitrate_0', 'dl_bitrate_1', ..., 'dl_bitrate_11']

    This is essential for making the LIME/SHAP explanations interpretable
    by associating weights with human-readable feature labels.

    Returns:
        list[str]: A list of strings representing feature names over time,
                   with the format "<feature_name>_<time_step>".
    """
    # Define the base feature columns expected at each timestep
    feature_columns = [
        'dl_bitrate', 'ul_bitrate',
        'cell_x_dl_retx', 'cell_x_dl_tx',
        'cell_x_ul_retx', 'cell_x_ul_tx',
        'ul_total_bytes_non_incr', 'dl_total_bytes_non_incr'
    ]

    # Append timestep suffix (_0 to _11) to each base feature
    feature_names = []
    for i in range(12):  # Assumes 12 time steps
        feature_names += [f"{col}_{i}" for col in feature_columns]

    return feature_names
