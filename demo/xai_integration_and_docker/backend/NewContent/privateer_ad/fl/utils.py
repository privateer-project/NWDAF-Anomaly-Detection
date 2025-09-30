from collections import OrderedDict

import torch

from flwr.common.typing import NDArrays

def set_weights(net, parameters: NDArrays):
    """
        Load model parameters from federated learning parameter arrays into a PyTorch model.

        This function bridges the gap between Flower's federated learning parameter
        representation and PyTorch's native model state management. It reconstructs
        the model's state dictionary from the flattened parameter arrays received
        from federated aggregation, ensuring that the local model reflects the most
        recent collaborative learning outcomes.

        Args:
            net: PyTorch model instance that will receive the updated parameters.
                 The model's architecture must match the parameter structure
                 provided in the parameters array.
            parameters (NDArrays): Collection of NumPy arrays containing model
                                 parameters in the same order as the model's
                                 state dictionary keys. These typically come
                                 from federated aggregation processes.

        Note:
            The function uses strict loading to ensure parameter integrity, which
            means all expected parameters must be present and correctly shaped.
            This prevents silent failures that could compromise federated learning
            effectiveness.
        """
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)

def get_weights(net):
    """
    Extract model parameters as NumPy arrays for federated learning transmission.

    This function performs the inverse operation of set_weights, converting a
    PyTorch model's parameters into the flattened NumPy array format required
    by Flower's federated learning infrastructure. The extraction process
    maintains parameter ordering consistency, which is essential for proper
    aggregation and redistribution across federation participants.

    The conversion automatically handles device placement by moving parameters
    to CPU before NumPy conversion, ensuring compatibility with federated
    learning protocols regardless of whether the local model was trained on
    GPU or CPU. This device-agnostic approach simplifies federated learning
    deployment across heterogeneous computational environments.
    Args:
        net: PyTorch model instance from which parameters will be extracted.
             The model should be in a consistent state for meaningful
             federated aggregation.

    Returns:
        list: Collection of NumPy arrays containing all model parameters
              in state dictionary order. These arrays are ready for
              transmission through federated learning protocols and
              subsequent aggregation processes.

    Example:
        Extracting parameters for federated submission:

        >>> # Train local model
        >>> train_local_model(model, local_data)
        >>> # Extract parameters for federation
        >>> local_params = get_weights(model)
        >>> # Send to federated server
        >>> server.submit_update(local_params)
    """
    return [val.cpu().numpy() for _, val in net.state_dict().items()]
