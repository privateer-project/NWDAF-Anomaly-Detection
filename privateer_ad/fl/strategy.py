from typing import List, Tuple, Union, Optional

import numpy as np
import torch
import mlflow

from mlflow.entities import RunStatus
from flwr.common import (ndarrays_to_parameters, parameters_to_ndarrays, Scalar, Parameters,
                         Metrics, EvaluateRes, FitRes)
from flwr.server.strategy import FedAvg
from flwr.server.client_proxy import ClientProxy

from privateer_ad import logger
from privateer_ad.config import MLFlowConfig, HParams, PathsConf, DPConfig
from privateer_ad.fl.utils import set_weights
from privateer_ad.etl.transform import DataProcessor
from privateer_ad.models import TransformerAD, TransformerADConfig
from privateer_ad.evaluate.evaluator import ModelEvaluator

def metrics_aggregation_fn(results: List[Tuple[int, Metrics]]):
    weighted_sums = {}
    total_num_examples = 0
    for num_examples, _metrics in results:
        total_num_examples += num_examples
        for name, value in _metrics.items():
            if name not in weighted_sums:
                weighted_sums[name] = 0
            weighted_sums[name] += (num_examples * value)
    weighted_metrics = {name: (value / total_num_examples) for name, value in weighted_sums.items()}
    return weighted_metrics

def config_fn(server_round: int):
    """Generate evaluation configuration for each round."""
    # Create the configuration dictionary
    return {'server_round': server_round}

class CustomStrategy(FedAvg):
    def __init__(self, num_rounds):
        model_config = TransformerADConfig()
        self.model = TransformerAD(config=model_config)
        initial_parameters = ndarrays_to_parameters([val.cpu().numpy() for _, val in self.model.state_dict().items()])
        super().__init__(on_fit_config_fn=config_fn,
                         on_evaluate_config_fn=config_fn,
                         fit_metrics_aggregation_fn=metrics_aggregation_fn,
                         evaluate_metrics_aggregation_fn=metrics_aggregation_fn,
                         initial_parameters=initial_parameters)
        self.num_rounds =num_rounds
        self.paths = PathsConf()
        self.mlflow_config = MLFlowConfig()
        hparams = HParams()
        dp_config = DPConfig()
        self.data_processor = DataProcessor(partition=False)
        self.test_dl = self.data_processor.get_dataloader('test',
                                                 batch_size=hparams.batch_size,
                                                 seq_len=hparams.seq_len,
                                                 only_benign=False)
        self.sample = next(iter(self.test_dl))[0]['encoder_cont'][:1].to('cpu')

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.evaluator = ModelEvaluator(criterion=hparams.loss, device=self.device)

        self.best_loss = np.Inf
        self.is_best = False
        server_run_name = self.mlflow_config.server_run_name
        server_run_id = None

        if self.mlflow_config.track:
            mlflow.set_tracking_uri(self.mlflow_config.server_address)
            mlflow.set_experiment(self.mlflow_config.experiment_name)

            if not server_run_name:
                server_run_name = 'federated_learning'
            if dp_config.enable:
                server_run_name += '-dp'

            if mlflow.active_run():
                logger.info(f"Found active run: {mlflow.active_run().info.run_name} - "
                            f"{mlflow.active_run().info.run_id}, ending it")
                mlflow.end_run()

            runs = mlflow.search_runs(experiment_names=[self.mlflow_config.experiment_name],
                                             filter_string=f'tags.mlflow.runName = \'{server_run_name}\'',
                                             max_results=1)
            if len(runs) > 0:
                # Use existing run
                server_run_id = runs.iloc[0].run_id
                logger.info(f'Found existing run: {server_run_name} - {server_run_id}')
                if not RunStatus.is_terminated(mlflow.get_run(server_run_id).info.status):
                    mlflow.MlflowClient().set_terminated(run_id=server_run_id)
            mlflow.start_run(run_id=server_run_id, run_name=server_run_name)
            server_run_name= mlflow.active_run().info.run_name

        self.trial_path = PathsConf().experiments_dir.joinpath(server_run_name)
        self.trial_path.mkdir(exist_ok=True)

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[Union[tuple[ClientProxy, FitRes], BaseException]],
    ) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        parameters_aggregated, metrics_aggregated = super().aggregate_fit(server_round, results, failures)
        if metrics_aggregated['val_loss'] < self.best_loss:
            set_weights(self.model, parameters_to_ndarrays(parameters_aggregated))
        if mlflow.active_run():
            mlflow.log_metrics(metrics_aggregated, step=server_round)
        return parameters_aggregated, metrics_aggregated

    def aggregate_evaluate(self,
        server_round: int,
        results: list[tuple[ClientProxy, EvaluateRes]],
        failures: list[Union[tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> tuple[Optional[float], dict[str, Scalar]]:
        loss_aggregated, metrics_aggregated = super().aggregate_evaluate(server_round=server_round,
                                                                         results=results,
                                                                         failures=failures)
        if mlflow.active_run():
            mlflow.log_metrics(metrics_aggregated, step=server_round)

        if server_round == self.num_rounds:
            self.model.to(self.device)
            metrics, figures = self.evaluator.evaluate(self.model, self.test_dl, prefix='best')
            if mlflow.active_run():
                mlflow.log_metrics(metrics, step=server_round)
                for name, fig in figures.items():
                    mlflow.log_figure(fig, f'{name}.png')

                self.model.to('cpu')
                _input = self.sample.to('cpu')
                _output = self.model(_input)
                if isinstance(_output, dict):
                    _output = {key: val.detach().numpy() for key, val in _output.items()}
                else:
                    _output = _output.detach().numpy()

                    mlflow.pytorch.log_model(pytorch_model=self.model,
                                             artifact_path=f'best_model',
                                             registered_model_name=f'best_model',
                                             signature=mlflow.models.infer_signature(model_input=_input.detach().numpy(),
                                                                                     model_output=_output),
                                             pip_requirements=self.paths.root.joinpath('requirements.txt').as_posix())
        return loss_aggregated, metrics_aggregated
