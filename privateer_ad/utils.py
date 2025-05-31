import logging

from pathlib import Path

import mlflow
import torch

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


def log_model(model, dataloader, direction, target_metric, current_target_metric, experiment_name, pip_requirements):
    """Log trained model to MLFlow with proper signature and champion tagging."""
    model.to('cpu')
    signature = get_signature(model=model, dataloader=dataloader)

    client = mlflow.tracking.MlflowClient()
    model_name = 'TransformerAD'

    # Determine sort direction for finding best run
    if direction == 'maximize':
        sorting = 'DESC'
    else:
        sorting = 'ASC'

    # Get current target metric value

    # Check if this is a new champion
    is_champion = True
    previous_champion_version = None

    try:
        # Find the best run across all experiments
        best_run = client.search_runs(
            mlflow.get_experiment_by_name(experiment_name).experiment_id,
            order_by=[f'metrics.{target_metric} {sorting}'],
            max_results=1
        )[0]

        if target_metric in best_run.data.metrics:
            best_target_metric = best_run.data.metrics[target_metric]

            # Compare with previous best
            if direction == 'maximize':
                is_champion = current_target_metric > best_target_metric
            else:
                is_champion = current_target_metric < best_target_metric

            print(f'Previous best {target_metric}: {best_target_metric}')
            print(f'Current {target_metric}: {current_target_metric}')
            print(f'Is new champion: {is_champion}')

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
                print(f"No previous champion found or error accessing registry: {e}")

    except (IndexError, KeyError) as e:
        print(f"No previous runs found or error accessing metrics: {e}")
        # This is likely the first model, so it's automatically the champion
        is_champion = True

    # Log the model
    model_info = mlflow.pytorch.log_model(
        pytorch_model=model,
        artifact_path='model',
        registered_model_name=model_name,
        signature=signature,
        pip_requirements=pip_requirements)

    # Handle champion tagging
    if is_champion:
        print(f"🏆 New champion model! Logging with champion tag.")

        # Remove champion tag from previous version
        if previous_champion_version:
            try:
                client.delete_model_version_tag(model_name, previous_champion_version, '🏆champion')
                print(f"Removed champion tag from version {previous_champion_version}")
            except Exception as e:
                print(f"Error removing champion tag from previous version: {e}")

        try:
            # Get the version number of the newly logged model
            current_version = model_info.registered_model_version
            client.set_model_version_tag(model_name, current_version, '🏆champion', 'true')

            # Also add metadata about when it became champion
            import datetime
            client.set_model_version_tag(
                model_name,
                current_version,
                'champion_since',
                datetime.datetime.now().isoformat()
            )
            client.set_model_version_tag(
                model_name,
                current_version,
                target_metric,
                str(current_target_metric)
            )

            print(f"Added champion tag to version {current_version}")
        except Exception as e:
            print(f"Error adding champion tag: {e}")
    else:
        print(f"Model performance ({current_target_metric}) did not exceed previous best. No champion tag added.")

def get_signature(model, dataloader):
    _input = next(iter(dataloader))[0]['encoder_cont'][:1].to('cpu')
    _output = model(_input)

    if isinstance(_output, dict):
        _output = {key: val.detach().numpy() for key, val in _output.items()}
    else:
        _output = _output.detach().numpy()
    signature = mlflow.models.infer_signature(model_input=_input.detach().numpy(), model_output=_output)
    return signature