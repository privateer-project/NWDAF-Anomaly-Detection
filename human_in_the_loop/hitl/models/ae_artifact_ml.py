from typing import Mapping
import numpy as np
import torch

from artifact_torch.binary_classification import (
    BinaryClassifier, ClassificationParams, BinaryClassificationResults
)
from artifact_core.typing import Array
from artifact_torch.nn import ModelInput, ModelOutput

from hitl.models.ae import (
    Conv1dAE,
    DenseAE,
)

from hitl.utils.utils import calculate_percentiles

class ForwardPassModelInput(ModelInput):
    initial_tensor: torch.Tensor
    
class ForwardPassModelOutput(ModelOutput):
    reconstructed_tensor: torch.Tensor
    t_loss: float = None

class ClassificationParameters(ClassificationParams):
    threshold: float = 0.5

class FilterModel(BinaryClassifier[
    ForwardPassModelInput,
    ForwardPassModelOutput,
    ClassificationParameters,
    Mapping[str, Array]
]):
    def __init__(
        self,
        model: Conv1dAE
    ):
        super().__init__()
        self.model = model

    def forward(self, model_input: ForwardPassModelInput) -> ForwardPassModelOutput:
        reconstructed = self.model(model_input["initial_tensor"])
        return ForwardPassModelOutput(
            reconstructed_tensor=reconstructed,
            t_loss=torch.mean((model_input["initial_tensor"] - reconstructed) ** 2)
        )
        
    def classify(self, data: Mapping[str, Array], params: ClassificationParameters) -> BinaryClassificationResults:

        array = np.stack([value for value in data.values()])
        # Preprocess input array and support both dense and conv1d models
        if array.ndim == 3:
            # conv1d path: expected per-sample storage (T, F) stacked -> (N, T, F)
            # The model expects (N, F, T) for Conv1dAE, so try to transpose if needed.
            # Detect whether the stacked array is (N, T, F) by comparing dimensions to model.seq_len/num_features
            tensor = torch.from_numpy(array).float().to(self._device)
            # Try conv1d forward; if model is Conv1dAE we expect (N, F, T)
            if isinstance(self.model, Conv1dAE):
                # If input seems (N, T, F) (time first), transpose to (N, F, T)
                if tensor.shape[1] != self.model.num_features and tensor.shape[2] == self.model.num_features:
                    tensor = tensor.transpose(1, 2).contiguous()

                with torch.no_grad():
                    recon = self.model(tensor)
                    mse = ((tensor - recon) ** 2).mean(dim=(1, 2))
            else:
                # If the wrapped model is dense but received 3D input, flatten per-sample
                B = tensor.shape[0]
                flat = tensor.view(B, -1)
                flat = flat.to(self._device)
                with torch.no_grad():
                    recon = self.model(flat)
                    mse = ((flat - recon) ** 2).mean(dim=1)

        elif array.ndim == 2:
            # dense path: stacked 2D array (N, D)
            tensor = torch.from_numpy(array).float().to(self._device)
            with torch.no_grad():
                recon = self.model(tensor)
                # recon shape (N, D) for DenseAE
                mse = ((tensor - recon) ** 2).mean(dim=1)

        else:
            raise ValueError("Unsupported input array ndim for classification: expected 2 or 3")

        # Percentile based probability estimation
        anom_probs = calculate_percentiles(mse.cpu().numpy())

        POSITIVE_CLASS = "anomaly"
        NEGATIVE_CLASS = "normal"
        CLASSNAMES = [NEGATIVE_CLASS, POSITIVE_CLASS]

        id_to_predicted_class = {}
        id_to_prob_positive_class = {}
        
        
        keys = list(data.keys())
        anom_probs = anom_probs.tolist()
        
        for i, anom_prob in enumerate(anom_probs):
            # Get label based on threshold
            id_to_prob_positive_class[keys[i]] = anom_prob
            
            id_to_predicted_class[keys[i]] = POSITIVE_CLASS if anom_prob >= params["threshold"] else NEGATIVE_CLASS

        return BinaryClassificationResults.build(
            class_names=CLASSNAMES,
            positive_class=POSITIVE_CLASS,
            id_to_class=id_to_predicted_class,
            id_to_prob_pos=id_to_prob_positive_class,
        )


		# threshold_value = params["threshold"]

		# THIS IS THE WHOLE ESSENCE OF THE PROJECT
		# Normally, an anomaly is when the reconstruction error is HIGHER than the threshold.
		# However, to flip the logic to detect NORMAL samples instead, we invert the condition
		# label = 1 if mse < threshold_value else 0
        
        
        
        
        # Classify based on threshold
        # return label