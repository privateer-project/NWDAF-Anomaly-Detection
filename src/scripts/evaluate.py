import argparse
from pathlib import Path

import torch

from src.config import *
from src.data_utils.data_loading.dataloader import DataLoaderFactory
from src.training import ModelEvaluator
from src.architectures import LSTMAutoencoder


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained LSTM Autoencoder')

    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to saved model checkpoint')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for evaluation')
    parser.add_argument('--window_size', type=int, default=12,
                        help='Window size for time series')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'val', 'test'],
                        help='Dataset split to evaluate on')
    parser.add_argument('--save_path', type=str, default=Path.cwd(),
                        help='Path to save results')

    args = parser.parse_args()

    # Initialize configs
    paths = Paths()
    metadata = MetaData()
    mlflow_config = MLFlowConfig()
    hparams = HParams(batch_size=args.batch_size, window_size=args.window_size)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model checkpoint
    checkpoint = torch.load(args.model_path, map_location=device)

    # Initialize model with checkpoint configuration
    model = LSTMAutoencoder(**checkpoint['model_config'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # Create dataloader
    loader_factory = DataLoaderFactory(metadata, paths, hparams)
    dataloader = loader_factory.get_single_dataloader(
        args.split,
        window_size=args.window_size,
        train=False
    )

    # Initialize evaluator
    evaluator = ModelEvaluator(
        model=model,
        device=device,
        paths=paths,
        hparams=hparams,
        mlflow_config=mlflow_config
    )

    # Run evaluation
    metrics = evaluator.evaluate(dataloader, save_path=args.save_path)

    # Print results
    print(f"\nEvaluation metrics on {args.split} set:")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")

    print(f"\nPlots saved in: {paths.analysis}/plots/")


if __name__ == '__main__':
    main()