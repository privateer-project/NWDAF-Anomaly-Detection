import argparse

import mlflow

from src import config
from src.config import *
from src.training import ModelTrainer


def main():
    parser = argparse.ArgumentParser(description='Train LSTM Autoencoder')
    # Training parameters
    parser.add_argument('--epochs', type=int, default=1,
                        help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--window_size', type=int, default=128,
                        help='Window size for time series')
    # Model parameters
    parser.add_argument('--hidden_size1', type=int, default=64,
                        help='First hidden layer size')
    parser.add_argument('--hidden_size2', type=int, default=64,
                        help='Second hidden layer size')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='Number of LSTM layers')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')

    # Differential Privacy parameters
    parser.add_argument('--enable_dp', action='store_true',
                        help='Enable differential privacy')
    parser.add_argument('--target_epsilon', type=float, default=2.0,
                        help='Target epsilon for DP')
    parser.add_argument('--target_delta', type=float, default=1e-7,
                        help='Target delta for DP')

    args = parser.parse_args()
    paths = Paths()
    metadata = MetaData()
    optimizer_config = OptimizerConfig()
    mlflow_config = MLFlowConfig()
    partition_config = PartitionConfig()

    # Initialize configs with command line arguments
    hparams = HParams(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        window_size=args.window_size
    )

    model_config = getattr(config, hparams.model + 'Config')(hidden_size1=args.hidden_size1,
                                                             hidden_size2=args.hidden_size2,
                                                             num_layers=args.num_layers,
                                                             dropout=args.dropout)

    dp_conf = DifferentialPrivacyConfig(enable=args.enable_dp,
                                        target_epsilon=args.target_epsilon,
                                        target_delta=args.target_delta)


    if hparams.model == 'LSTMAutoencoder':
        model_config.hidden_size1=args.hidden_size1,
        model_config.hidden_size2=args.hidden_size2,
        model_config.num_layers=args.num_layers,
        model_config.dropout=args.dropout
    if hparams.model == 'TransformerAD':
        model_config.seq_len = hparams.window_size
        model_config.n_layers = args.num_layers
        model_config.dropout = args.dropout

    if mlflow_config.track:
        mlflow.set_tracking_uri(mlflow_config.server_address)
        mlflow.set_experiment(mlflow_config.experiment_name)
        mlflow.start_run()

    trainer = ModelTrainer(
        paths=paths,
        metadata=metadata,
        hparams=hparams,
        model_config=model_config,
        optimizer_config=optimizer_config,
        diff_privacy_config=dp_conf,
        mlflow_config=mlflow_config,
        partition_config=partition_config,
    )

    trainer.training()
    metrics = trainer.evaluate_model(save_path=args.save_path)
    print("Test metrics:", metrics)
    mlflow.end_run()

if __name__ == '__main__':
    main()