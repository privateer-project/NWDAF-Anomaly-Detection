import logging

from pathlib import Path

import mlflow
import numpy as np
import torch

from privateer_ad.config import TrainingConfig

def load_model_weights(model_path: str, paths_config) -> dict:
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
    """Log trained model to MLFlow with proper signature and champion tagging."""
    model.to('cpu')
    signature = get_signature(model=model, sample=sample)
    client = mlflow.tracking.MlflowClient()

    # Determine sort direction for finding best run
    if direction == 'maximize':
        sorting = 'DESC'
    else:
        sorting = 'ASC'

    previous_champion_version = None
    current_target_metric = current_metrics[target_metric]

    try:
        # Find the best run across all experiments
        best_run = client.search_runs(experiment_id, order_by=[f'metrics.`{target_metric}` {sorting}'], max_results=1)[0]
        best_target_metric = best_run.data.metrics[target_metric]

        # Compare with previous best
        if direction == 'maximize':
            is_champion = current_target_metric >= best_target_metric
        else:
            is_champion = current_target_metric <= best_target_metric

        logging.info(f'Previous best {target_metric}: {best_target_metric}')
        logging.info(f'Current {target_metric}: {current_target_metric}')
        logging.info(f'Is new champion: {is_champion}')

        # Find previous champion version to remove tag
        if is_champion:
            try:
                registered_model = client.get_registered_model(model_name)
                for version in registered_model.latest_versions:
                    version_details = client.get_model_version(model_name, version.version)
                    if version_details.tags.get('champion') == 'true':
                        previous_champion_version = version.version
                        break
            except Exception as e:
                logging.error(f"No previous champion found or error accessing registry: {e}")

    except (IndexError, KeyError) as e:
        logging.error(f"No previous runs found or error accessing metrics: {e}")
        # This is likely the first model, so it's automatically the champion
        is_champion = True
        if direction == 'maximize':
            best_target_metric = - np.inf
        else:
            best_target_metric = np.inf

    # Log the model
    model_info = mlflow.pytorch.log_model(
        pytorch_model=model,
        artifact_path=model_name,
        registered_model_name=model_name,
        signature=signature,
        pip_requirements=pip_requirements)
    # Get the version number of the newly logged model
    current_version = model_info.registered_model_version

    # Handle champion tagging
    if is_champion:
        logging.info(f"🏆 New champion model! Logging with champion tag.")

        # Remove champion tag from previous version
        if previous_champion_version:
            try:
                client.delete_model_version_tag(model_name, previous_champion_version, 'champion')
                logging.warning(f"Removed champion tag from version {previous_champion_version}")
            except Exception as e:
                logging.error(f"Error removing champion tag from previous version: {e}")

        try:
            client.set_model_version_tag(model_name, current_version, 'champion', 'true')

            # Also add metadata about when it became champion
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
        logging.info(f"Model performance ({current_target_metric}) did not exceed"
              f" previous best ({best_target_metric}). No champion tag added.")

    client.set_model_version_tag(model_name, current_version, target_metric, str(current_metrics))

def get_signature(model, sample):
    _output = model(sample)

    if isinstance(_output, dict):
        _output = {key: val.detach().numpy() for key, val in _output.items()}
    else:
        _output = _output.detach().numpy()
    signature = mlflow.models.infer_signature(model_input=sample.detach().numpy(), model_output=_output)
    return signature


def load_champion_model(tracking_uri, model_name: str = "TransformerAD"):
    """
    Load champion model with associated threshold and loss function.

    Returns:
        tuple: (model, threshold, loss_fn)
    """
    mlflow.tracking.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()

    try:
        # Get registered model
        registered_model = client.get_registered_model(model_name)

        champion_version = None
        champion_run_id = None

        # Find champion version
        for version in registered_model.latest_versions:
            version_details = client.get_model_version(model_name, version.version)
            if version_details.tags.get('champion') == 'true':
                champion_version = version.version
                champion_run_id = version_details.run_id
                break

        # If no champion found, use latest
        if champion_version is None:
            latest_version_details = client.get_model_version(model_name, registered_model.latest_versions[0].version)
            champion_version = latest_version_details.version
            champion_run_id = latest_version_details.run_id
            logging.warning(f"⚠️ No champion found, using latest v{champion_version}")
        else:
            logging.info(f"🏆 Found champion model v{champion_version}")

        # Load the model
        model_uri = f"models:/{model_name}/{champion_version}"
        load_conf = {'map_location': torch.device('cpu')}
        model = mlflow.pytorch.load_model(model_uri=model_uri)

        # Get run details to extract threshold and loss_fn
        run = client.get_run(champion_run_id)

        # Extract threshold from metrics (try different possible names)
        threshold = None
        threshold_keys = [
            'test_threshold', 'val_threshold', 'global_test_threshold',
            'threshold', 'eval_threshold', 'validation_threshold'
        ]

        for key in threshold_keys:
            if key in run.data.metrics:
                threshold = run.data.metrics[key]
                logging.info(f"📏 Found threshold: {threshold:.6f} (from metric: {key})")
                break

        if threshold is None:
            logging.warning("⚠️ No threshold found in run metrics, using default: 0.5")
            threshold = 0.5

        # Extract loss_fn from parameters
        loss_fn_name = run.data.params.get('loss_fn_name', TrainingConfig().loss_fn_name)
        loss_fn = getattr(torch.nn, loss_fn_name)(reduction='none')
        logging.info(f"✅ Successfully loaded:")
        logging.info(f"🏆 Model: {model_name} v{champion_version}")
        logging.info(f"📏 Threshold: {threshold}")
        logging.info(f"📊 Loss function: {loss_fn_name}")

        return model, threshold, loss_fn

    except Exception as e:
        logging.error(f"❌ Error loading champion model: {e}")
        raise


# Usage example:
if __name__ == '__main__':
    model, threshold, loss_fn = load_champion_model("http://localhost:5001", "TransformerAD")
