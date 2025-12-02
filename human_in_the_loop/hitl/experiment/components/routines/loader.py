from typing import Any, Optional, Type

from artifact_experiment.tracking import DataSplit
from artifact_torch.nn import Model
from artifact_torch.nn.plans import ForwardHookPlan, ModelIOPlan
from artifact_torch.nn.routines import DataLoaderRoutine

from hitl.experiment.components.plans.forward_hook import DataLoaderForwardHookPlan
from hitl.experiment.components.plans.model_io import DataLoaderModelIOPlan
from hitl.models.ae_artifact_ml import ForwardPassModelInput, ForwardPassModelOutput


class DemoLoaderRoutine(DataLoaderRoutine[Model[Any, Any], ForwardPassModelInput, ForwardPassModelOutput]):
    @classmethod
    def _get_model_io_plan(
        cls, data_split: DataSplit
    ) -> Optional[Type[ModelIOPlan[ForwardPassModelInput, ForwardPassModelOutput]]]:
        if data_split is DataSplit.TRAIN:
            return DataLoaderModelIOPlan
        elif data_split is DataSplit.VALIDATION:
            return DataLoaderModelIOPlan

    @classmethod
    def _get_forward_hook_plan(
        cls, data_split: DataSplit
    ) -> Optional[Type[ForwardHookPlan[Model[Any, Any]]]]:
        if data_split is DataSplit.TRAIN:
            return DataLoaderForwardHookPlan