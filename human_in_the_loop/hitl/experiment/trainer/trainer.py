from typing import Any, Optional

import torch
from artifact_torch.binary_classification import BinaryClassifier
from artifact_torch.nn import Trainer
from artifact_torch.nn.early_stopping import EarlyStopper, EpochBoundStopper, StopperUpdateData
from artifact_torch.nn.model_tracking import ModelTracker, ModelTrackingCriterion
from torch import optim

from hitl.models.ae_artifact_ml import ForwardPassModelInput, ForwardPassModelOutput
from hitl.config import CHECKPOINT_PERIOD, LEARNING_RATE, MAX_N_EPOCHS

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DEVICE = torch.device("cpu")


class DemoTrainer(
    Trainer[
        BinaryClassifier[Any, Any, Any, Any],
        ForwardPassModelInput,
        ForwardPassModelOutput,
        StopperUpdateData,
        ModelTrackingCriterion,
    ]
):
    @staticmethod
    def _get_optimizer(
        model: BinaryClassifier[Any, Any, Any, Any],
    ) -> optim.Optimizer:
        return optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

    @staticmethod
    def _get_scheduler(
        optimizer: optim.Optimizer,
    ) -> Optional[optim.lr_scheduler._LRScheduler]:
        _ = optimizer

    @staticmethod
    def _get_device() -> torch.device:
        return DEVICE

    @staticmethod
    def _get_checkpoint_period() -> Optional[int]:
        return CHECKPOINT_PERIOD

    @staticmethod
    def _get_model_tracker() -> Optional[ModelTracker[ModelTrackingCriterion]]:
        pass

    def _get_model_tracking_criterion(self) -> Optional[ModelTrackingCriterion]:
        pass

    @staticmethod
    def _get_early_stopper() -> EarlyStopper[StopperUpdateData]:
        return EpochBoundStopper(max_n_epochs=MAX_N_EPOCHS)

    def _get_stopper_update_data(self) -> StopperUpdateData:
        return StopperUpdateData(n_epochs_elapsed=self.n_epochs_elapsed)