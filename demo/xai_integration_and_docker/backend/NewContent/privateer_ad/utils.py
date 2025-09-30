import logging
from pathlib import Path
import mlflow
import numpy as np
import torch
from privateer_ad.config import TrainingConfig


def load_model_weights(model_path: str, paths_config) -> dict:
    """
    Load PyTorch model weights with flexible path resolution and format handling.

    Attempts to locate and load model weights from various common path patterns,
    handling different state dictionary formats and cleaning distributed training
    artifacts for compatibility with single-device loading.

    Args:
        model_path (str): Model file path or experiment identifier
        paths_config: Path configuration object with directory specifications

    Returns:
        dict: Cleaned model state dictionary ready for loading

    Raises:
        FileNotFoundError: When model file cannot be located in any expected location
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
            logging.info(f'Loading model from: {path}')
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
                logging.warning(f'Failed to load from {path}: {e}')
                continue

    raise FileNotFoundError(f'Could not find model file at any of: {[str(p) for p in possible_paths]}')


def log_model(model, model_name, sample, direction, target_metric, current_metrics, experiment_id, pip_requirements):
    """
    Log trained model to MLflow with champion tracking and version management.

    Registers the model in MLflow with champion tagging based on
    performance comparison with previous versions. Manages version lifecycle
    and metadata tagging for model selection and deployment decisions.

    Args:
        model: Trained PyTorch model to register
        model_name (str): Registry name for the model
        sample: Representative input sample for signature inference
        direction (str): Optimization direction ('maximize' or 'minimize')
        target_metric (str): Primary metric for champion selection
        current_metrics (dict): Performance metrics from current training
        experiment_id (str): MLflow experiment identifier
        pip_requirements (str): Path to requirements file for reproducibility

    Note:
        Automatically manages champion tag transfer between model versions
        based on performance improvement detection.
    """
    model.to('cpu')
    signature = get_signature(model=model, sample=sample)
    client = mlflow.tracking.MlflowClient()

    # Determine sort direction for finding best run
    current_target_metric = current_metrics[target_metric]
    sorting = 'DESC' if direction == 'maximize' else 'ASC'

    # Check if this is a new champion
    is_champion = True
    best_target_metric = -np.inf if direction == 'maximize' else np.inf
    champions = []

    try:
        # Find the best run across all experiments
        best_run = client.search_runs([experiment_id],
                                      order_by=[f'metrics.`{target_metric}` {sorting}'],
                                      max_results=1)[0]

        if target_metric in best_run.data.metrics:
            best_target_metric = best_run.data.metrics[target_metric]

            # Compare with previous best
            if direction == 'maximize':
                is_champion = current_target_metric >= best_target_metric
            else:
                is_champion = current_target_metric <= best_target_metric

            logging.info(f'Previous best {target_metric}: {best_target_metric}')
            logging.info(f'Current {target_metric}: {current_target_metric}')
            logging.info(f'Is new champion: {is_champion}')

        # Find previous champion versions to remove tag
        try:
            champions = client.search_model_versions(filter_string=f"name = '{model_name}' and tag.champion = 'true'")
        except Exception as e:
            logging.warning(f"No previous champion found or error accessing registry: {e}")

    except (IndexError, KeyError) as e:
        logging.warning(f"No previous runs found or error accessing metrics: {e}")
        # This is likely the first model, so it's automatically the champion
        is_champion = True

    # Log the model
    model_info = mlflow.pytorch.log_model(
        pytorch_model=model,
        artifact_path=model_name,
        registered_model_name=model_name,
        signature=signature,
        pip_requirements=pip_requirements
    )

    current_version = model_info.registered_model_version

    # Handle champion tagging
    if is_champion:
        logging.info(f"🏆 New champion model! Logging with champion tag.")

        # Remove champion tag from previous versions
        for champion in champions:
            try:
                client.delete_model_version_tag(model_name, champion.version, 'champion')
                logging.info(f"Removed champion tag from version {champion.version}")
            except Exception as e:
                logging.warning(f"Failed to remove champion tag from version {champion.version}: {e}")

        try:
            client.set_model_version_tag(model_name, current_version, 'champion', 'true')

            # Add metadata about when it became champion
            import datetime
            client.set_model_version_tag(
                model_name,
                current_version,
                'champion_since',
                datetime.datetime.now().isoformat()
            )

            logging.info(f"Added champion tag to version {current_version}")
        except Exception as e:
            logging.error(f"Error adding champion tag: {e}")
    else:
        logging.info(f"Model performance ({current_target_metric}) did not exceed "
                     f"previous best ({best_target_metric}). No champion tag added.")

    client.set_model_version_tag(model_name, current_version, target_metric, str(current_target_metric))


def get_signature(model, sample):
    """Generate MLflow model signature from sample input and model output."""
    _output = model(sample)

    if isinstance(_output, dict):
        _output = {key: val.detach().numpy() for key, val in _output.items()}
    else:
        _output = _output.detach().numpy()
    signature = mlflow.models.infer_signature(model_input=sample.detach().numpy(), model_output=_output)
    return signature


def load_champion_model(tracking_uri, model_name: str = "TransformerAD"):
    """
    Load the best-performing model version with associated metadata.

    Retrieves the champion model from MLflow registry along with its optimal
    threshold and loss function configuration. Falls back to latest version
    if no champion is tagged.

    Args:
        tracking_uri (str): MLflow tracking server address
        model_name (str): Registered model name to load

    Returns:
        tuple: (model, threshold, loss_fn) for complete inference setup

    Raises:
        Exception: When model loading or metadata extraction fails
    """
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()

    try:
        # Get registered model
        version = None
        run_id = None
        source = None
        # Get all versions of the model
        all_versions = client.search_model_versions(f"name='{model_name}'")
        logging.info(f'Found {len(all_versions)} total versions for model {model_name}')

        # Find champion version
        for model_version in all_versions:
            if model_version.tags.get('champion') == 'true':
                version = model_version.version
                run_id = model_version.run_id
                source = model_version.source
                logging.info(f"🏆 Found champion model v{version}")
                break

        if version is None:
            # Sort by version number and get latest
            latest_version = max(all_versions, key=lambda v: int(v.version))
            version = latest_version.version
            run_id = latest_version.run_id
            source = latest_version.source
            logging.warning(f"⚠️ No champion found, using latest version v{version}")

        # Load the model
        logging.info(f"Loading model from: {source}")
        load_conf = {'map_location': torch.device('cuda' if torch.cuda.is_available() else 'cpu')}
        model = mlflow.pytorch.load_model(model_uri=source, **load_conf)
        logging.info(f"Model loaded!")

        # Get run details to extract threshold and loss_fn
        run = client.get_run(run_id)
        # Extract threshold from metrics
        threshold = None
        threshold_keys = ['global_test_threshold', 'val_threshold', 'test_threshold', 'threshold']

        for key in threshold_keys:
            if key in run.data.metrics:
                threshold = run.data.metrics.get(key)
                logging.info(f"📏 Found threshold: {threshold:.6f} (from metric: {key})")
                break

        if threshold is None:
            logging.warning("⚠️ No threshold found in run metrics, using default: 0.5")
            threshold = 0.5

        # Extract loss_fn from parameters
        loss_fn_name = run.data.params.get('loss_fn_name', TrainingConfig().loss_fn_name)
        loss_fn = getattr(torch.nn, loss_fn_name)(reduction='none')

        logging.info(f"✅ Successfully loaded:")
        logging.info(f"🏆 Model: {model_name} v{version}")
        logging.info(f"📏 Threshold: {threshold}")
        logging.info(f"📊 Loss function: {loss_fn_name}")

        return model, threshold, loss_fn

    except Exception as e:
        logging.error(f"❌ Error loading champion model: {e}")
        raise