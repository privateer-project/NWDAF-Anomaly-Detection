from typing import Any, Optional, Type

from artifact_torch.nn import Model
from artifact_torch.nn.plans import BackwardHookPlan, ForwardHookPlan, ModelIOPlan
from artifact_torch.nn.routines import TrainDiagnosticsRoutine

from hitl.experiment.components.plans.model_io import TrainDiagnosticsModelIOPlan
from hitl.models.ae_artifact_ml import ForwardPassModelInput, ForwardPassModelOutput


class DemoTrainDiagnosticsRoutine(
    TrainDiagnosticsRoutine[Model[Any, Any], ForwardPassModelInput, ForwardPassModelOutput]
):
    @classmethod
    def _get_model_io_plan(cls) -> Optional[Type[ModelIOPlan[ForwardPassModelInput, ForwardPassModelOutput]]]:
        return TrainDiagnosticsModelIOPlan

    @classmethod
    def _get_forward_hook_plan(cls) -> Optional[Type[ForwardHookPlan[Model[Any, Any]]]]:
        pass

    @classmethod
    def _get_backward_hook_plan(cls) -> Optional[Type[BackwardHookPlan[Model[Any, Any]]]]:
        pass