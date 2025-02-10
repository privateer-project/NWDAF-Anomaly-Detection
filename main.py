import os
import logging
from datetime import datetime

import mlflow

import config
from config import Paths, MLFlowConfig, FlowerConfig, PartitionConfig, HParams, MetaData
from data_handling.extract import DownloadConfig, Downloader
from training import ModelTrainer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main(mode : str=None, client_id: int = None,
         flwr_server: str = None,
         partition_id: int = None,
         num_partitions: int = None,
         num_classes_per_partition: int = None):
    paths = Paths()
    dlconf = DownloadConfig()
    dloader = Downloader(dlconf)
    mlflowconf = MLFlowConfig()
    flwrconf = FlowerConfig()
    partconf = PartitionConfig()
    paths
    metadata = MetaData()
    hparams = HParams()
    # model_config =
    # TransformerADConfig
    # LSTMAutoencoderConfig
    # optimizer_config
    # diff_privacy_config
    # mlflow_config
    # partition_config
    # Check if data needs to be downloaded
    if not os.path.exists(paths.raw_dataset):
        logger.info("Downloading dataset...")
        dloader.download_extract()

    mlflow.set_tracking_uri(mlflowconf.server_address)
    mlflow.set_experiment(mlflowconf.experiment_name)
    run_name = datetime.now().strftime("%Y%m%d-%H%M%S")

    if flwr_server:
        flwrconf.server_address = flwr_server
    if partition_id:
        partconf.partition_id = partition_id
    if num_partitions:
        partconf.num_partitions = num_partitions
    if num_partitions:
        partconf.num_classes_per_partition = num_classes_per_partition
    #todo create autotuneconf
    if config['autotune']['enabled']:
        mlflow.start_run(run_name='autotune')

    if mode == 'server':
        run_name = '_'.join(['mode', run_name])
        mlflow.start_run(run_name=run_name, nested=config['autotune']['enabled'])
    if mode == 'client':
        if client_id:
            flwrconf.client_id = client_id
        trainer = ModelTrainer(paths=paths,metadata=metadata)
        client = NWDAFClient(trainer=trainer).to_client()
        # Start client
        fl.client.start_client(server_address=config['flwr']['server_address'], client=client)
    else:
        mlflow.start_run(run_name=run_name, nested=config['autotune']['enabled'])
        trainer = ModelTrainer()
        best_checkpoint = trainer.training()
        trainer._evaluate_loop()
        model = trainer.model.load_state_dict(best_checkpoint['model_state_dict'])
        testing.test(model, test_dl=trainer.test_dl, criterion=trainer.loss_class(reduction='none'), device=trainer.device)


if __name__ == '__main__':
    Fire(main)