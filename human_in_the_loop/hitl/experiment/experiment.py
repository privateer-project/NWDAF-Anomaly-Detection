from typing import Any, Optional, Type

import pandas as pd
from artifact_torch.binary_classification import (
    BinaryClassificationExperiment,
    BinaryClassificationRoutine,
    BinaryClassifier,
)
from artifact_torch.nn import Trainer
from artifact_torch.nn.routines import DataLoaderRoutine, TrainDiagnosticsRoutine

from hitl.experiment.components.routines.artifact import DemoBinaryClassificationRoutine
from hitl.experiment.components.routines.loader import DemoLoaderRoutine
from hitl.experiment.components.routines.train_diagnostics import (
    DemoTrainDiagnosticsRoutine,
)
from hitl.models.ae_artifact_ml import (
    ClassificationParameters,
    ForwardPassModelInput,
    ForwardPassModelOutput,
)

from hitl.experiment.trainer.trainer import DemoTrainer

class DemoBinaryClassificationExperiment(
    BinaryClassificationExperiment[
        BinaryClassifier[
            Any,
            ForwardPassModelOutput,
            ClassificationParameters,
            pd.DataFrame,
        ],
        ForwardPassModelInput,
        ForwardPassModelOutput,
        ClassificationParameters,
        pd.DataFrame,
    ]
):
    @classmethod
    def _get_trainer(
        cls,
    ) -> Type[
        Trainer[
            BinaryClassifier[
                Any,
                ForwardPassModelOutput,
                ClassificationParameters,
                pd.DataFrame,
            ],
            ForwardPassModelInput,
            ForwardPassModelOutput,
            Any,
            Any,
        ]
    ]:
        return DemoTrainer

    @classmethod
    def _get_train_diagnostics_routine(
        cls,
    ) -> Optional[
        Type[
            TrainDiagnosticsRoutine[
                BinaryClassifier[
                    Any,
                    ForwardPassModelOutput,
                    ClassificationParameters,
                    pd.DataFrame,
                ],
                ForwardPassModelInput,
                ForwardPassModelOutput,
            ]
        ]
    ]:
        return DemoTrainDiagnosticsRoutine

    @classmethod
    def _get_loader_routine(
        cls,
    ) -> Optional[
        Type[
            DataLoaderRoutine[
                BinaryClassifier[
                    Any,
                    ForwardPassModelOutput,
                    ClassificationParameters,
                    pd.DataFrame,
                ],
                ForwardPassModelInput,
                ForwardPassModelOutput,
            ]
        ]
    ]:
        return DemoLoaderRoutine

    @classmethod
    def _get_artifact_routine(
        cls,
    ) -> Optional[Type[BinaryClassificationRoutine[ClassificationParameters, pd.DataFrame]]]:
        return DemoBinaryClassificationRoutine