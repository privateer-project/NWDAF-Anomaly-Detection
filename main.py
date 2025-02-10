import os
import logging

from fire import Fire

from src.data_utils import Downloader
from datetime import datetime

import mlflow
import flwr as fl
from src.fed_learn import NWDAFClient
from src.training import ModelTrainer
from src.config.other_configs import config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main(mode : str=None, client_id: int = None,
         flwr_server: str = None,
         partition_id: int = None,
         num_partitions: int = None,
         num_classes_per_partition: int = None):

    for _dir in ['data', 'processed', 'architectures', 'scalers', 'analysis']:
        os.makedirs(config['paths'][_dir], exist_ok=True)

    # Check if data needs to be downloaded
    if not os.path.exists(config['metadata']['raw_dataset']):
        logger.info("Downloading dataset...")
        Downloader(url=config['metadata']['url'], extract_path=config['paths']['raw']).download_extract()

    mlflow.set_tracking_uri(f'{config['mlflow']['server_address']}')
    mlflow.set_experiment(config['mlflow']['experiment_name'])
    run_name = datetime.now().strftime("%Y%m%d-%H%M%S")

    if flwr_server:
        config['flwr']['server'] = flwr_server
    if partition_id:
        config['train_hparams']['partition_id'] = partition_id
    if num_partitions:
        config['train_hparams']['num_partitions'] = num_partitions
    if num_partitions:
        config['train_hparams']['num_classes_per_partition'] = num_classes_per_partition

    if config['autotune']['enabled']:
        mlflow.start_run(run_name='autotune')

    if mode == 'server':
        run_name = '_'.join(['mode', run_name])
        mlflow.start_run(run_name=run_name, nested=config['autotune']['enabled'])
    if mode == 'client':
        if client_id:
            config['flwr']['client_id'] = client_id
        trainer = ModelTrainer(config=config)
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