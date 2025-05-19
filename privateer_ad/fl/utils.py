from collections import OrderedDict

import torch

from flwr.common.typing import NDArrays

def set_weights(net, parameters: NDArrays):
    """Set model parameters from a list of NumPy arrays."""
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)

def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]
