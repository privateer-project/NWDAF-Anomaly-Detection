from typing import List, Tuple, Dict, Optional, Union

import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from fire import Fire
from sklearn.metrics import classification_report
from privateer_ad.config import AttentionAutoencoderConfig, AlertFilterConfig, PathsConf, HParams, logger
from privateer_ad.models import AttentionAutoencoder, AlertFilterModel
from privateer_ad.data_utils.transform import DataProcessor
from privateer_ad.train_alert_filter.feedback_collector import FeedbackCollector
from pathlib import Path

def make_predictions(
        model_path,
        data_path,
        threshold: float = 0.026970019564032555
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load a trained model and make predictions on the specified dataset.
    Parameters:
    -----------
    model_path : str
        Path to the saved model state dictionary or the experiment directory (eg. '20250317-100421').
    data_path : str
        Path to the CSV file containing the data to evaluate or 'train', 'val', 'test' for prepared datasets.
    threshold : float
        Precalculated threshold.
    Returns:
    --------
    Tuple[np.ndarray, np.ndarray, np.ndarray]:
        - inputs: Array of shape (n_samples, seq_len, n_features) containing input data
        - losses: Array of shape (n_samples,) containing reconstruction errors
        - predictions: Array of shape (n_samples,) containing anomaly decisions (1 = anomaly, 0 = normal)
        - labels: Array of shape (n_samples,) containing integer labels
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    paths = PathsConf()
    hparams = HParams()

    model_path = Path(model_path)
    if paths.experiments_dir.joinpath(model_path).exists():
        model_path = paths.experiments_dir.joinpath(model_path)

    # Load Data
    dp = DataProcessor()
    dl = dp.get_dataloader(data_path,
                           use_pca=hparams.use_pca,
                           batch_size=hparams.batch_size,
                           seq_len=hparams.seq_len,
                           only_benign=False)

    # Load model
    model = AttentionAutoencoder(config=AttentionAutoencoderConfig())
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    state_dict = {key.removeprefix('_module.'): value for key, value in state_dict.items()}
    model.load_state_dict(state_dict)
    model = model.to(device)

    model.to(device)
    inputs: List[float] = []
    losses: List[float] = []
    predictions: List[int] = []
    labels: List[int] = []
    criterion_fn = getattr(nn, hparams.loss)(reduction='none')
    model.eval()

    with torch.no_grad(): # Collect results
        for batch in tqdm(dl, desc="Computing reconstruction errors"):
            x = batch[0]['encoder_cont'].to(device)
            targets = np.squeeze(batch[1][0])
            labels.extend(targets.cpu().tolist())

            output = model(x)
            batch_rec_errors = criterion_fn(x, output)

            loss_per_sample = batch_rec_errors.mean(dim=(1, 2))
            losses.extend(loss_per_sample.cpu().tolist())

            inputs.extend(x.cpu().tolist())
            predictions.extend((loss_per_sample > threshold).cpu().tolist())

    inputs_np = np.asarray(inputs)
    losses_np = np.array(losses)
    predictions_np = np.asarray(predictions)
    labels_np = np.array(labels, dtype=int)

    # Keep same n samples for balanced metrics
    ben_labels = labels_np[labels_np == 0]
    ben_predictions = predictions_np[labels_np == 0]

    mal_labels = labels_np[labels_np == 1]
    mal_predictions = predictions_np[labels_np == 1]

    _len = len(mal_labels)
    ben_labels = ben_labels[:_len]
    ben_predictions = ben_predictions[:_len]
    mal_labels = mal_labels[:_len]
    mal_predictions = mal_predictions[:_len]
    labels = np.concatenate([ben_labels, mal_labels])
    predictions = np.concatenate([ben_predictions, mal_predictions])
    print(classification_report(labels, predictions))
    return inputs_np, losses_np, predictions_np, labels_np

def make_predictions_with_filter(
        model_path: Union[str, Path],
        data_path: Union[str, Path],
        filter_model_path: Optional[Union[str, Path]] = None,
        threshold: float = 0.026970019564032555,
        collect_feedback: bool = False,
        use_filter: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load a trained autoencoder model and alert filter model, and make predictions on the specified dataset.
    
    Parameters:
    -----------
    model_path : Union[str, Path]
        Path to the saved autoencoder model state dictionary or the experiment directory.
    data_path : Union[str, Path]
        Path to the CSV file containing the data to evaluate or 'train', 'val', 'test' for prepared datasets.
    filter_model_path : Optional[Union[str, Path]]
        Path to the saved alert filter model state dictionary. If None, no filtering is applied.
    threshold : float
        Precalculated threshold for anomaly detection.
    collect_feedback : bool
        Whether to collect feedback for the alert filter model. If True, the function will
        prompt the user for feedback on each alert.
    use_filter : bool
        Whether to use the alert filter model. If False, behaves like make_predictions but still returns latents.
        
    Returns:
    --------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        - inputs: Array containing input data
        - latents: Array containing latent representations
        - losses: Array containing reconstruction errors
        - predictions: Array containing anomaly decisions (1 = anomaly, 0 = normal)
        - filtered_decisions: Array containing filtered decisions (1 = allow alert, 0 = deny alert)
        - labels: Array containing true labels (1 = anomaly, 0 = normal)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    paths = PathsConf()
    hparams = HParams()

    # Convert paths to Path objects
    model_path = Path(model_path)
    if paths.experiments_dir.joinpath(model_path).exists():
        model_path = paths.experiments_dir.joinpath(model_path)

    # Load Data
    dp = DataProcessor()
    dl = dp.get_dataloader(data_path,
                           use_pca=hparams.use_pca,
                           batch_size=hparams.batch_size,
                           seq_len=hparams.seq_len,
                           only_benign=False)

    # Load autoencoder model
    autoencoder = AttentionAutoencoder(config=AttentionAutoencoderConfig())
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    state_dict = {key.removeprefix('_module.'): value for key, value in state_dict.items()}
    autoencoder.load_state_dict(state_dict)
    autoencoder = autoencoder.to(device)

    # Load alert filter model if provided and use_filter is True
    alert_filter = None
    if filter_model_path is not None and use_filter:
        filter_model_path = Path(filter_model_path)
        alert_filter = AlertFilterModel(config=AlertFilterConfig())
        alert_filter.load_state_dict(torch.load(filter_model_path, map_location=device))
        alert_filter = alert_filter.to(device)
        logger.info(f"Loaded alert filter model from {filter_model_path}")

    # Initialize feedback collector if needed
    feedback_collector = None
    if collect_feedback:
        feedback_collector = FeedbackCollector()
        logger.info("Feedback collection enabled")

    # Initialize lists to store results
    inputs: List[float] = []
    latents: List[float] = []
    losses: List[float] = []
    predictions: List[int] = []
    filtered_decisions: List[int] = []
    labels: List[int] = []
    
    criterion_fn = getattr(nn, hparams.loss)(reduction='none')
    autoencoder.eval()
    if alert_filter is not None:
        alert_filter.eval()

    with torch.no_grad(): # Collect results
        for batch in tqdm(dl, desc="Computing predictions"):
            x = batch[0]['encoder_cont'].to(device)
            targets = np.squeeze(batch[1][0])
            labels.extend(targets.cpu().tolist())

            # Get autoencoder outputs with latent representation
            output, latent = autoencoder(x, return_latent=True)
            
            print(x.shape, output.shape, latent.shape)
            
            # Calculate reconstruction error
            batch_rec_errors = criterion_fn(x, output)
            loss_per_sample = batch_rec_errors.mean(dim=(1, 2))
            
            # Determine anomaly decision
            prediction = (loss_per_sample > threshold).float()
            
            
            # Apply alert filter if available and use_filter is True
            if alert_filter is not None and use_filter:
                # Process latent vectors to match the expected dimension
                # If latent is multi-dimensional, take the mean across the sequence dimension
                if latent.dim() > 2:
                    latent_processed = torch.mean(latent, dim=1)
                else:
                    latent_processed = latent
                
                # Ensure the latent vector has the correct dimension (8)
                # If it's larger (e.g., 16), take the first 8 elements
                target_dim = 16  # This should match AlertFilterConfig.latent_dim
                if latent_processed.shape[-1] > target_dim:
                    print(f"Warning: Latent dimension {latent_processed.shape[-1]} is larger than target dimension {target_dim}. Truncating.")
                    latent_processed = latent_processed[:, :target_dim]
                
                allow_alert = alert_filter(latent_processed, prediction, loss_per_sample)
                
                # Final decision: anomaly AND allow_alert
                final_decision = prediction * (allow_alert > 0.5).float()
            else:
                # If no filter, all alerts are allowed
                allow_alert = torch.ones_like(prediction)
                final_decision = prediction
            
            # Collect feedback if enabled
            if collect_feedback and alert_filter is not None:
                for i in range(len(prediction)):
                    if prediction[i] == 1:  # Only collect feedback for anomalies
                        print(f"\nAlert {i+1}:")
                        print(f"Reconstruction error: {loss_per_sample[i].item():.6f}")
                        print(f"Alert filter decision: {'Allow' if allow_alert[i].item() > 0.5 else 'Deny'}")
                        
                        # Ask for user feedback
                        user_input = input("Is this a true positive? (y/n): ").lower()
                        user_feedback = 1 if user_input.startswith('y') else 0
                        
                        # Add feedback to collector
                        feedback_collector.add_feedback(
                            latent=latent[i].cpu(),
                            prediction=prediction[i].item(),
                            reconstruction_error=loss_per_sample[i].item(),
                            user_feedback=user_feedback
                        )
            
            # Store results
            inputs.extend(x.cpu().tolist())
            latents.extend(latent.cpu().tolist())
            losses.extend(loss_per_sample.cpu().tolist())
            predictions.extend(prediction.cpu().tolist())
            # Convert final_decision to a list of scalar values
            filtered_decisions.extend(final_decision.cpu().flatten().tolist())

    # Convert lists to numpy arrays
    inputs_np = np.asarray(inputs)
    latents_np = np.asarray(latents)
    losses_np = np.array(losses)
    predictions_np = np.array(predictions)
    filtered_decisions_np = np.array(filtered_decisions)
    
    # print number of filtered decisions per value
    unique, counts = np.unique(filtered_decisions_np, return_counts=True)
    print(f"Filtered decisions counts: {dict(zip(unique, counts))}")

   
    labels_np = np.array(labels, dtype=int)

    # Print classification report for both unfiltered and filtered decisions
    print("\nUnfiltered Anomaly Detection Results:")
    print_balanced_classification_report(labels_np, predictions_np)
    
    if alert_filter is not None and use_filter:
        print("\nFiltered Anomaly Detection Results:")
        print_balanced_classification_report(labels_np, filtered_decisions_np)
    
    return inputs_np, latents_np, losses_np, predictions_np, filtered_decisions_np, labels_np

def print_balanced_classification_report(labels_np: np.ndarray, predictions_np: np.ndarray) -> None:
    """
    Print a classification report with balanced classes.
    
    Parameters:
    -----------
    labels_np : np.ndarray
        Array of true labels
    predictions_np : np.ndarray
        Array of predicted labels
    """
    # Check if arrays have the same shape
    if len(labels_np) != len(predictions_np):
        logger.warning(f"Labels and predictions have different shapes: {labels_np.shape} vs {predictions_np.shape}")
        # Resize predictions to match labels
        if len(predictions_np) > len(labels_np):
            predictions_np = predictions_np[:len(labels_np)]
        else:
            # This shouldn't happen, but just in case
            logger.error("Predictions array is smaller than labels array, cannot generate report")
            return
    
    # Keep same n samples for balanced metrics
    ben_labels = labels_np[labels_np == 0]
    ben_predictions = predictions_np[labels_np == 0]

    mal_labels = labels_np[labels_np == 1]
    mal_predictions = predictions_np[labels_np == 1]

    _len = min(len(mal_labels), len(ben_labels))
    if _len == 0:
        print("No samples available for balanced evaluation")
        return
        
    ben_labels = ben_labels[:_len]
    ben_predictions = ben_predictions[:_len]
    mal_labels = mal_labels[:_len]
    mal_predictions = mal_predictions[:_len]
    
    labels = np.concatenate([ben_labels, mal_labels])
    predictions = np.concatenate([ben_predictions, mal_predictions])
    
    print(classification_report(labels, predictions))

def main():
    Fire({
        'predict': make_predictions,
        'predict_with_filter': make_predictions_with_filter
    })
