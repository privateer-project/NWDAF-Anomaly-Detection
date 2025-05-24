from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import mlflow
import torch

from privateer_ad import logger
from privateer_ad.config import (get_model_config,
                                 get_training_config,
                                 get_mlflow_config,
                                 get_paths
                                 )
from privateer_ad.architectures import TransformerAD, TransformerADConfig
from privateer_ad.etl import DataProcessor
from .evaluator import ModelEvaluator

def evaluate(
        model_path: str,
        data_path: str = 'test',
        threshold: Optional[float] = None,
        batch_size: Optional[int] = None,
        seq_len: Optional[int] = None,
        device: Optional[str] = None,
        save_figures: bool = True,
        log_to_mlflow: bool = True,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """
    Evaluate a trained model with configuration injection.

    Args:
        model_path: Path to model file or experiment directory
        data_path: Dataset to evaluate on ('train', 'val', 'test', or custom path)
        threshold: Custom threshold for anomaly detection (None for auto-threshold)
        batch_size: Override batch size for evaluation
        seq_len: Override sequence length
        device: Override device selection ('auto', 'cpu', 'cuda')
        save_figures: Whether to save evaluation figures locally
        log_to_mlflow: Whether to log results to MLFlow (if enabled)

    Returns:
        Tuple of (metrics_dict, figures_dict)
    """
    # Inject configurations
    model_config = get_model_config()
    training_config = get_training_config()
    mlflow_config = get_mlflow_config()
    paths_config = get_paths()

    logger.info(f'Starting evaluation of model: {model_path}')
    logger.info(f'Evaluating on dataset: {data_path}')

    # Setup device
    device = device or 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    logger.info(f'Using device: {device}')

    # Setup data processing
    logger.info('Setting up data processing...')
    dp = DataProcessor()

    # Use provided parameters or fall back to config
    eval_batch_size = batch_size or training_config.batch_size
    eval_seq_len = seq_len or model_config.seq_len

    # Get dataloader
    try:
        dl = dp.get_dataloader(
            data_path,
            batch_size=eval_batch_size,
            seq_len=eval_seq_len,
            only_benign=False  # Always include both classes for evaluation
        )
        logger.info(f'Data loader created successfully for {data_path}')
    except Exception as e:
        logger.error(f'Failed to create data loader: {e}')
        raise

    # Get sample for model configuration
    sample_batch = next(iter(dl))
    sample_input = sample_batch[0]['encoder_cont'][:1]
    input_size = sample_input.shape[-1]

    logger.info(f'Detected input size: {input_size}')
    logger.info(f'Using sequence length: {eval_seq_len}')

    # Setup model configuration
    transformer_config = TransformerADConfig(
        seq_len=eval_seq_len,
        input_size=input_size,
        num_layers=model_config.num_layers,
        hidden_dim=model_config.hidden_dim,
        latent_dim=model_config.latent_dim,
        num_heads=model_config.num_heads,
        dropout=model_config.dropout
    )

    # Create model
    logger.info('Creating model...')
    model = TransformerAD(transformer_config)

    # Load model weights
    logger.info(f'Loading model from: {model_path}')
    model_state_dict = _load_model_weights(model_path, paths_config)

    try:
        model.load_state_dict(model_state_dict)
        logger.info('Model weights loaded successfully')
    except Exception as e:
        logger.error(f'Failed to load model weights: {e}')
        raise

    # Move model to device and set to evaluation mode
    model = model.to(device)
    model.eval()

    # Setup evaluator
    logger.info('Initializing evaluator...')
    evaluator = ModelEvaluator(
        criterion=training_config.loss_function,
        device=device
    )

    # Setup MLFlow if enabled and requested
    if log_to_mlflow and mlflow_config.enabled:
        _setup_evaluation_mlflow(mlflow_config, model_path)

    # Perform evaluation
    logger.info(f'Evaluating model on {data_path} dataset...')
    try:
        metrics, figures = evaluator.evaluate(
            model=model,
            dataloader=dl,
            threshold=threshold,
            prefix=f'eval_{data_path}',
            step=0
        )
        logger.info('Evaluation completed successfully')
    except Exception as e:
        logger.error(f'Evaluation failed: {e}')
        raise

    # Log results
    _log_evaluation_results(metrics, figures, mlflow_config, log_to_mlflow, save_figures, model_path)

    return metrics, figures


def _load_model_weights(model_path: str, paths_config) -> Dict[str, torch.Tensor]:
    """
    Load model weights from various path formats.

    Args:
        model_path: Path to model file or experiment directory
        paths_config: Paths configuration

    Returns:
        Model state dictionary
    """
    model_path = Path(model_path)

    # Try different path resolutions
    possible_paths = [
        model_path,  # Direct path
        paths_config.experiments_dir / model_path / 'model.pt',  # Experiment directory
        paths_config.models_dir / model_path,  # Models directory
        Path(model_path).with_suffix('.pt'),  # Add .pt extension
    ]

    for path in possible_paths:
        if path.exists():
            logger.info(f'Loading model from: {path}')
            try:
                state_dict = torch.load(path, map_location='cpu')

                # Handle different state dict formats
                if isinstance(state_dict, dict):
                    # Remove common prefixes from distributed training
                    cleaned_state_dict = {}
                    for key, value in state_dict.items():
                        # Remove common prefixes
                        clean_key = key
                        for prefix in ['_module.', 'module.']:
                            if clean_key.startswith(prefix):
                                clean_key = clean_key[len(prefix):]
                        cleaned_state_dict[clean_key] = value

                    return cleaned_state_dict
                else:
                    raise ValueError(f'Unexpected state dict format: {type(state_dict)}')

            except Exception as e:
                logger.warning(f'Failed to load from {path}: {e}')
                continue

    raise FileNotFoundError(f'Could not find model file at any of: {[str(p) for p in possible_paths]}')


def _setup_evaluation_mlflow(mlflow_config, model_path: str):
    """Setup MLFlow for evaluation tracking."""
    try:
        mlflow.set_tracking_uri(mlflow_config.server_address)
        mlflow.set_experiment(mlflow_config.experiment_name)

        # Create evaluation run
        run_name = f'eval_{Path(model_path).stem}'
        mlflow.start_run(run_name=run_name)

        # Log evaluation parameters
        mlflow.log_params({
            'model_path': str(model_path),
            'evaluation_type': 'standalone',
            'timestamp': torch.utils.data.get_worker_info() or 'main'
        })

        logger.info(f'MLFlow evaluation run started: {run_name}')

    except Exception as e:
        logger.warning(f'Failed to setup MLFlow: {e}')


def _log_evaluation_results(
        metrics: Dict[str, float],
        figures: Dict[str, Any],
        mlflow_config,
        log_to_mlflow: bool,
        save_figures: bool,
        model_path: str
):
    """Log evaluation results to various outputs."""

    # Format and log metrics to console
    metrics_log = '\n'.join([f'{key}: {value:.5f}' for key, value in metrics.items()])
    logger.info(f'Evaluation metrics:\n{metrics_log}')

    # Log to MLFlow if enabled and requested
    if log_to_mlflow and mlflow_config.enabled and mlflow.active_run():
        try:
            logger.info('Logging metrics to MLFlow...')
            mlflow.log_metrics(metrics)

            # Log figures
            for name, fig in figures.items():
                mlflow.log_figure(fig, f'evaluation_figures/{name}.png')

            logger.info('MLFlow logging completed')

        except Exception as e:
            logger.warning(f'Failed to log to MLFlow: {e}')

    # Save figures locally if requested
    if save_figures:
        try:
            # Create output directory
            output_dir = Path('evaluation_results') / Path(model_path).stem
            output_dir.mkdir(parents=True, exist_ok=True)

            # Save figures
            for name, fig in figures.items():
                fig_path = output_dir / f'{name}.png'
                fig.savefig(fig_path, dpi=300, bbox_inches='tight')
                logger.info(f'Figure saved: {fig_path}')

            # Save metrics
            metrics_path = output_dir / 'metrics.txt'
            with open(metrics_path, 'w') as f:
                for key, value in metrics.items():
                    f.write(f'{key}: {value:.5f}\n')
            logger.info(f'Metrics saved: {metrics_path}')

        except Exception as e:
            logger.warning(f'Failed to save figures locally: {e}')


def evaluate_multiple_models(
        model_dir: str,
        data_path: str = 'test',
        pattern: str = '*.pt',
        **kwargs
) -> Dict[str, Tuple[Dict[str, float], Dict[str, Any]]]:
    """
    Evaluate multiple models in a directory.

    Args:
        model_dir: Directory containing model files
        data_path: Dataset to evaluate on
        pattern: File pattern to match (e.g., '*.pt', 'model_*.pt')
        **kwargs: Additional arguments passed to evaluate()

    Returns:
        Dictionary mapping model names to (metrics, figures) tuples
    """
    model_dir = Path(model_dir)

    if not model_dir.exists():
        raise FileNotFoundError(f'Model directory not found: {model_dir}')

    # Find model files
    model_files = list(model_dir.glob(pattern))
    if not model_files:
        raise FileNotFoundError(f'No model files found matching pattern: {pattern}')

    logger.info(f'Found {len(model_files)} model files to evaluate')

    results = {}
    for model_file in model_files:
        logger.info(f'\n{"=" * 50}')
        logger.info(f'Evaluating: {model_file.name}')
        logger.info(f'{"=" * 50}')

        try:
            metrics, figures = evaluate(
                model_path=str(model_file),
                data_path=data_path,
                **kwargs
            )
            results[model_file.stem] = (metrics, figures)

        except Exception as e:
            logger.error(f'Failed to evaluate {model_file.name}: {e}')
            results[model_file.stem] = (None, None)

    return results


def compare_models(
        models_dict: Dict[str, str],
        data_path: str = 'test',
        metrics_to_compare: Optional[list] = None,
        **kwargs
) -> Dict[str, Any]:
    """
    Compare multiple models and generate comparison report.

    Args:
        models_dict: Dictionary mapping model names to model paths
        data_path: Dataset to evaluate on
        metrics_to_compare: List of metrics to include in comparison
        **kwargs: Additional arguments passed to evaluate()

    Returns:
        Comparison results dictionary
    """
    if metrics_to_compare is None:
        metrics_to_compare = ['eval_test_roc_auc', 'eval_test_f1-score', 'eval_test_precision', 'eval_test_recall']

    logger.info(f'Comparing {len(models_dict)} models')

    results = {}
    all_metrics = {}

    for model_name, model_path in models_dict.items():
        logger.info(f'\nEvaluating model: {model_name}')
        try:
            metrics, figures = evaluate(
                model_path=model_path,
                data_path=data_path,
                log_to_mlflow=False,  # Don't log individual runs
                **kwargs
            )
            results[model_name] = (metrics, figures)
            all_metrics[model_name] = metrics

        except Exception as e:
            logger.error(f'Failed to evaluate {model_name}: {e}')
            results[model_name] = (None, None)

    # Generate comparison
    comparison = {
        'individual_results': results,
        'metric_comparison': {},
        'best_models': {}
    }

    # Compare metrics
    for metric in metrics_to_compare:
        metric_values = {}
        for model_name, metrics in all_metrics.items():
            if metrics and metric in metrics:
                metric_values[model_name] = metrics[metric]

        if metric_values:
            comparison['metric_comparison'][metric] = metric_values
            # Find best model for this metric
            best_model = max(metric_values.items(), key=lambda x: x[1])
            comparison['best_models'][metric] = best_model

    # Log comparison summary
    logger.info('\n' + '=' * 50)
    logger.info('MODEL COMPARISON SUMMARY')
    logger.info('=' * 50)

    for metric, best_model in comparison['best_models'].items():
        logger.info(f'{metric}: {best_model[0]} ({best_model[1]:.5f})')

    return comparison


def main():
    """Main function with Fire integration for CLI usage."""
    from fire import Fire

    # Create a command interface that supports multiple functions
    commands = {
        'evaluate': evaluate,
        'evaluate_multiple': evaluate_multiple_models,
        'compare': compare_models
    }

    Fire(commands)


if __name__ == '__main__':
    main()