from typing import List, Type, TypeVar

import pandas as pd
import numpy as np
from numpy.typing import NDArray
import torch
from artifact_torch.nn import Dataset

from hitl.models.ae_artifact_ml import ForwardPassModelInput
# TabularVAEDatasetT = TypeVar("TabularVAEDatasetT", bound="FilterModelClassifierDataset")


class FilterModelClassifierDataset(Dataset[ForwardPassModelInput]):
    def __init__(self, input_array: NDArray):        
        self._t_features = torch.tensor(input_array, dtype=torch.float32)
        
    def __len__(self) -> int:
        return self._t_features.shape[0]

    def __getitem__(self, idx: int) -> ForwardPassModelInput:
        t_features = self._t_features[idx]
        model_input = ForwardPassModelInput(initial_tensor=t_features)
                
        return model_input