from typing import List, Tuple, Union, Optional

import numpy as np
import torch
import mlflow
from mlflow.models import infer_signature

from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays, Scalar, Parameters, Metrics, FitIns, FitRes
from flwr.common.logger import FLOWER_LOGGER
from flwr.server.strategy import FedAvg, DifferentialPrivacyClientSideFixedClipping
from flwr.server.client_proxy import ClientProxy
from flwr.server.client_manager import ClientManager

from privateer_ad.data_utils.transform import DataProcessor
from privateer_ad.evaluate.evaluator import ModelEvaluator
from privateer_ad.models import AttentionAutoencoder, AttentionAutoencoderConfig
from privateer_ad.config import PathsConf, MLFlowConfig, HParams
from privateer_ad.fl.utils import set_weights

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
    def __init__(self, noise_multiplier: float, clipping_norm: float, num_sampled_clients: int):
        model_config = AttentionAutoencoderConfig()
        self.model = AttentionAutoencoder(config=model_config)
        initial_parameters = ndarrays_to_parameters([val.cpu().numpy() for _, val in self.model.state_dict().items()])
        base_strategy = FedAvg(on_fit_config_fn=config_fn,
                               on_evaluate_config_fn=config_fn,
                               fit_metrics_aggregation_fn=metrics_aggregation_fn,
                               evaluate_metrics_aggregation_fn=metrics_aggregation_fn,
                               initial_parameters=initial_parameters)
        super().__init__(base_strategy, noise_multiplier, clipping_norm, num_sampled_clients)

        self.paths = PathsConf()
        self.mlflow_config = MLFlowConfig()
        hparams = HParams()
        self.logger = FLOWER_LOGGER
        self.data_processor = DataProcessor()
        self.test_dl = self.data_processor.get_dataloader('test',
                                                 use_pca=hparams.use_pca,
                                                 batch_size=hparams.batch_size,
                                                 seq_len=hparams.seq_len,
                                                 only_benign=False)
        self.sample = next(iter(self.test_dl))[0]['encoder_cont'][:1].to('cpu')

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.evaluator = ModelEvaluator(criterion=hparams.loss, device=self.device)

        self.best_loss = np.Inf

        if self.mlflow_config.track:
            mlflow.set_tracking_uri(self.mlflow_config.server_address)
            mlflow.set_experiment(self.mlflow_config.experiment_name)

            runs = mlflow.search_runs(experiment_names=[self.mlflow_config.experiment_name],
                                             filter_string=f'tags.mlflow.runName = \'{self.mlflow_config.server_run_name}\'',
                                             max_results=1)
            if len(runs) > 0:
                # Use existing run
                self.run_id = runs.iloc[0].run_id
                self.logger.info(f'Found existing run: {self.mlflow_config.server_run_name} with id: {self.run_id}')
            else:
                # Create a new run
                self.logger.info(f'No run with name {self.mlflow_config.server_run_name} found. Creating parent run.')
                with mlflow.start_run(run_name=self.mlflow_config.server_run_name):
                    self.run_id = mlflow.active_run().info.run_id

        self.trial_path = PathsConf().experiments_dir.joinpath(self.mlflow_config.server_run_name)
        self.model_path = self.trial_path.joinpath('best_model.pt')
        self.trial_path.mkdir(exist_ok=True)

    def aggregate_fit(self,
                      server_round: int,
                      results: list[tuple[ClientProxy, FitRes]],
                      failures: list[Union[tuple[ClientProxy, FitRes], BaseException]],
                      ) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        """Aggregate model weights using weighted average and store checkpoint"""
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        is_best = self.best_loss > aggregated_metrics['val_loss']
        if is_best:
            # Save if aggregated loss is improved
            self.best_loss = aggregated_metrics['val_loss']
            # Save the model to disk
            print(f'Saving best model of round {server_round}...')
            set_weights(self.model, parameters_to_ndarrays(aggregated_parameters))
            torch.save(self.model.state_dict(), self.model_path)

        if self.mlflow_config.track:
            with mlflow.start_run(run_id=self.run_id):
                mlflow.log_metrics(aggregated_metrics, step=server_round)
                if is_best:
                    self.model.to('cpu')
                    _input = self.sample.to('cpu')
                    _output = self.model(_input)
                    if isinstance(_output, dict):
                        _output = {key: val.detach().numpy() for key, val in _output.items()}
                    else:
                        _output = _output.detach().numpy()

                        mlflow.pytorch.log_model(pytorch_model=self.model,
                                                 artifact_path=f'best_model_{server_round}',
                                                 registered_model_name=f'best_model_{server_round}',
                                                 signature=infer_signature(model_input=_input.detach().numpy(),
                                                                           model_output=_output),
                                                 pip_requirements=self.paths.root.joinpath('requirements.txt').as_posix())
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
        set_weights(self.model, parameters_to_ndarrays(parameters))
        self.model.to(self.device)
        metrics, figures = self.evaluator.evaluate(self.model, self.test_dl, prefix='eval', step=server_round)
        if self.mlflow_config.track:
            with mlflow.start_run(run_id=self.run_id):
                mlflow.log_metrics(metrics, step=server_round)
                # Log figures in the parent run
                for name, fig in figures.items():
                    mlflow.log_figure(fig, f'server_{name}_round_{server_round}.png')
        loss = metrics.pop('eval_loss')
        return loss, metrics
