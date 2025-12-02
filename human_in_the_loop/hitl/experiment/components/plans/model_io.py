from typing import List

from artifact_torch.nn.callbacks import (
    LossCallback,
    ModelIOArrayCallback,
    ModelIOArrayCollectionCallback,
    ModelIOPlotCallback,
    ModelIOPlotCollectionCallback,
    ModelIOScoreCallback,
    ModelIOScoreCollectionCallback,
)
from artifact_torch.nn.plans import ModelIOPlan, ModelIOPlanBuildContext

from hitl.models.ae_artifact_ml import ForwardPassModelInput, ForwardPassModelOutput
from hitl.config import LOADER_VALIDATION_PERIOD, TRAIN_DIAGNOSTICS_PERIOD

class DataLoaderModelIOPlan(ModelIOPlan[ForwardPassModelInput, ForwardPassModelOutput]):
    @classmethod
    def _get_score_callbacks(
        cls, context: ModelIOPlanBuildContext
    ) -> List[ModelIOScoreCallback[ForwardPassModelInput, ForwardPassModelOutput]]:
        return [LossCallback(period=LOADER_VALIDATION_PERIOD, writer=context.score_writer)]

    @classmethod
    def _get_array_callbacks(
        cls, context: ModelIOPlanBuildContext
    ) -> List[ModelIOArrayCallback[ForwardPassModelInput, ForwardPassModelOutput]]:
        _ = context
        return []

    @classmethod
    def _get_plot_callbacks(
        cls, context: ModelIOPlanBuildContext
    ) -> List[ModelIOPlotCallback[ForwardPassModelInput, ForwardPassModelOutput]]:
        _ = context
        return []

    @classmethod
    def _get_score_collection_callbacks(
        cls, context: ModelIOPlanBuildContext
    ) -> List[ModelIOScoreCollectionCallback[ForwardPassModelInput, ForwardPassModelOutput]]:
        _ = context
        return []

    @classmethod
    def _get_array_collection_callbacks(
        cls, context: ModelIOPlanBuildContext
    ) -> List[ModelIOArrayCollectionCallback[ForwardPassModelInput, ForwardPassModelOutput]]:
        _ = context
        return []

    @classmethod
    def _get_plot_collection_callbacks(
        cls, context: ModelIOPlanBuildContext
    ) -> List[ModelIOPlotCollectionCallback[ForwardPassModelInput, ForwardPassModelOutput]]:
        _ = context
        return []


class TrainDiagnosticsModelIOPlan(ModelIOPlan[ForwardPassModelInput, ForwardPassModelOutput]):
    @classmethod
    def _get_score_callbacks(
        cls, context: ModelIOPlanBuildContext
    ) -> List[ModelIOScoreCallback[ForwardPassModelInput, ForwardPassModelOutput]]:
        return [LossCallback(period=TRAIN_DIAGNOSTICS_PERIOD, writer=context.score_writer)]

    @classmethod
    def _get_array_callbacks(
        cls, context: ModelIOPlanBuildContext
    ) -> List[ModelIOArrayCallback[ForwardPassModelInput, ForwardPassModelOutput]]:
        _ = context
        return []

    @classmethod
    def _get_plot_callbacks(
        cls, context: ModelIOPlanBuildContext
    ) -> List[ModelIOPlotCallback[ForwardPassModelInput, ForwardPassModelOutput]]:
        _ = context
        return []

    @classmethod
    def _get_score_collection_callbacks(
        cls, context: ModelIOPlanBuildContext
    ) -> List[ModelIOScoreCollectionCallback[ForwardPassModelInput, ForwardPassModelOutput]]:
        _ = context
        return []

    @classmethod
    def _get_array_collection_callbacks(
        cls, context: ModelIOPlanBuildContext
    ) -> List[ModelIOArrayCollectionCallback[ForwardPassModelInput, ForwardPassModelOutput]]:
        _ = context
        return []

    @classmethod
    def _get_plot_collection_callbacks(
        cls, context: ModelIOPlanBuildContext
    ) -> List[ModelIOPlotCollectionCallback[ForwardPassModelInput, ForwardPassModelOutput]]:
        _ = context
        return []