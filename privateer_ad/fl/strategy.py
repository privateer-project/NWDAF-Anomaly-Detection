from typing import List, Tuple, Union, Optional, OrderedDict

import numpy as np
import torch
import mlflow
from mlflow.models import infer_signature

from flwr.common import Scalar, Parameters, Metrics, parameters_to_ndarrays, ndarrays_to_parameters, FitIns, FitRes
from flwr.common.logger import FLOWER_LOGGER
from flwr.server.strategy import FedAvg, DifferentialPrivacyClientSideFixedClipping
from flwr.server.client_proxy import ClientProxy
from flwr.server.client_manager import ClientManager

from privateer_ad.data_utils.transform import DataProcessor
from privateer_ad.evaluate.evaluator import ModelEvaluator
from privateer_ad.models import AttentionAutoencoder, AttentionAutoencoderConfig
from privateer_ad.config import PathsConf, MLFlowConfig, HParams

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

class CustomStrategy(DifferentialPrivacyClientSideFixedClipping):
    def __init__(self, noise_multiplier: float, clipping_norm: float, num_sampled_clients: int, run_name:str):
        self.paths = PathsConf()
        self.mlflow_config = MLFlowConfig()
        hparams = HParams()
        model_config = AttentionAutoencoderConfig()
        self.model = AttentionAutoencoder(config=model_config)
        data_processor = DataProcessor()
        self.logger = FLOWER_LOGGER
        self.test_dl = data_processor.get_dataloader('test',
                                                 use_pca=hparams.use_pca,
                                                 batch_size=hparams.batch_size,
                                                 seq_len=hparams.seq_len,
                                                 only_benign=False)
        self.sample = next(iter(self.test_dl))[0]['encoder_cont'][:1].to('cpu')

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.evaluator = ModelEvaluator(criterion=hparams.loss, device=self.device)

        initial_parameters = ndarrays_to_parameters([val.cpu().numpy() for _, val in self.model.state_dict().items()])
        base_strategy = FedAvg(on_fit_config_fn=config_fn,
                               on_evaluate_config_fn=config_fn,
                               fit_metrics_aggregation_fn=metrics_aggregation_fn,
                               evaluate_metrics_aggregation_fn=metrics_aggregation_fn,
                               initial_parameters=initial_parameters)
        super().__init__(base_strategy, noise_multiplier, clipping_norm, num_sampled_clients)

        self.best_loss = np.Inf
        self.parent_run_id = None

        if self.mlflow_config.track:
            mlflow.set_tracking_uri(self.mlflow_config.server_address)
            mlflow.set_experiment(self.mlflow_config.experiment_name)

            # Create a new run
            parent_runs = mlflow.search_runs(experiment_names=[self.mlflow_config.experiment_name],
                                             filter_string=f'tags.mlflow.runName = \'{run_name}\'',
                                             max_results=1)
            if len(parent_runs) > 0:
                # Use existing run
                self.parent_run_id = parent_runs.iloc[0].run_id
                self.logger.info(f'Found existing parent run: {run_name} with id: {self.parent_run_id}')
            else:
                self.logger.info(f'No Parent run with name {run_name} found. Creating parent run.')
                with mlflow.start_run(run_name=run_name):
                    self.parent_run_id = mlflow.active_run().info.run_id

        self.trial_path = PathsConf().experiments_dir.joinpath('federated_learning')
        self.trial_path.mkdir(exist_ok=True)

    def aggregate_fit(self,
                      server_round: int,
                      results: list[tuple[ClientProxy, FitRes]],
                      failures: list[Union[tuple[ClientProxy, FitRes], BaseException]],
                      ) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        """Aggregate model weights using weighted average and store checkpoint"""
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        self.best_loss = aggregated_metrics['val_loss']
        print(f'Saving round {server_round} aggregated_parameters...')

        # Convert `Parameters` to `list[np.ndarray]`
        aggregated_ndarrays: list[np.ndarray] = parameters_to_ndarrays(aggregated_parameters)
        # Convert to PyTorch `state_dict`
        aggregated_params_dict = zip(self.model.state_dict().keys(), aggregated_ndarrays)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in aggregated_params_dict})
        self.model.load_state_dict(state_dict, strict=True)

        # Save the model to disk
        model_path = self.trial_path.joinpath('model.pt')
        torch.save(self.model.state_dict(), model_path)
        # Log to MLFlow
        self.model.to('cpu')
        _input = self.sample.to('cpu')
        _output = self.model(_input)
        if isinstance(_output, dict):
            _output = {key: val.detach().numpy() for key, val in _output.items()}
        else:
            _output = _output.detach().numpy()

        if self.parent_run_id:
            mlflow.pytorch.log_model(pytorch_model=self.model,
                                     artifact_path=f'fl_model_{server_round}',
                                     registered_model_name=f'fl_model_{server_round}',
                                     signature=infer_signature(model_input=_input.detach().numpy(),
                                                               model_output=_output),
                                     pip_requirements=self.paths.root.joinpath('requirements.txt').as_posix(),
                                     run_id=self.parent_run_id)

            mlflow.log_metrics(aggregated_metrics,
                               step=server_round,
                               run_id=self.parent_run_id)

        return aggregated_parameters, aggregated_metrics

    def configure_fit(self, server_round: int, parameters: Parameters, client_manager: ClientManager) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        # Get FitIns from base class
        client_instructions = super().configure_fit(server_round, parameters, client_manager)
        # Add the parent run ID to the config for each client
        # if self.mlflow_config.track:
        #     for client_proxy, fit_ins in client_instructions:
        #         fit_ins.config.update({'parent_run_id': self.parent_run_id})
        return client_instructions

    def evaluate(self, server_round: int, parameters: Parameters) -> Optional[tuple[float, dict[str, Scalar]]]:
        """Evaluate model parameters using an evaluation function."""
        self.logger.info('Evaluating model on Server')
        parameters_ndarrays = parameters_to_ndarrays(parameters)
        params_dict = zip(self.model.state_dict().keys(), parameters_ndarrays)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)
        self.model.to(self.device)
        metrics, figures = self.evaluator.evaluate(self.model, self.test_dl, prefix='eval', step=server_round)
        if self.parent_run_id:
            mlflow.log_metrics(metrics, step=server_round, run_id=self.parent_run_id)
                # Log figures in the parent run
            with mlflow.start_run(run_id=self.parent_run_id):
                for name, fig in figures.items():
                    mlflow.log_figure(fig, f'server_{name}_round_{server_round}.png')
        loss = metrics.pop('eval_loss')
        return loss, metrics
