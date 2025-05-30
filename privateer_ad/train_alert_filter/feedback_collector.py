import os
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Union, Optional, Sequence

from privateer_ad.config import AlertFilterConfig, PathsConf, setup_logger

class FeedbackCollector:
    """
    Collects and manages user feedback for the alert filter model.
    
    This class handles storing and retrieving user feedback on alerts,
    which is used to train the alert filter model to reduce false positives.
    
    Attributes:
        storage_path (Path): Path to the directory where feedback data is stored
        feedback_data (Dict): Dictionary containing feedback data
    """
    
    def __init__(self, storage_path: Optional[Path] = None):
        """
        Initialize the feedback collector.
        
        Args:
            storage_path (Path, optional): Path to the directory where feedback data is stored.
                If None, uses the default path from PathsConf.
        """
        paths = PathsConf()
        self.storage_path = storage_path or paths.root.joinpath('hitlad_demo', 'feedback')
        self.storage_file = self.storage_path.joinpath('feedback.json')
        self.logger = setup_logger('feedback-collector')

        self.feedback_data = {
            'latent': [],
            'anomaly_decision': [],
            'reconstruction_error': [],
            'user_feedback': []  # 1 = true positive (allow), 0 = false positive (deny)
        }
        
        # Create storage directory if it doesn't exist
        os.makedirs(self.storage_path, exist_ok=True)
        
        # Load existing data if available
        self._load_existing_data()
        
    def _load_existing_data(self):
        """Load existing feedback data from storage."""
        if self.storage_file.exists():
            try:
                with open(self.storage_file, 'r') as f:
                    
                    data = json.load(f)
                
                # Convert lists to numpy arrays for easier processing
                self.feedback_data = {
                    'latent': [np.array(x) for x in data.get('latent', [])],
                    'anomaly_decision': np.array(data.get('anomaly_decision', [])).tolist(),
                    'reconstruction_error': np.array(data.get('reconstruction_error', [])).tolist(),
                    'user_feedback': np.array(data.get('user_feedback', [])).tolist()
                }
                
                self.logger.info(f"Loaded {len(self.feedback_data['user_feedback'])} feedback entries")
            except Exception as e:
                self.logger.error(f"Error loading feedback data: {e}")
                # Initialize with empty data if loading fails
                self.feedback_data = {
                    'latent': [],
                    'anomaly_decision': [],
                    'reconstruction_error': [],
                    'user_feedback': []
                }
    
    def _save_data(self):
        """Save feedback data to storage."""
        try:
            # Convert numpy arrays to lists for JSON serialization
            data_to_save = {
                'latent': [x.tolist() if isinstance(x, np.ndarray) else x for x in self.feedback_data['latent']],
                'anomaly_decision': self.feedback_data['anomaly_decision'],
                'reconstruction_error': self.feedback_data['reconstruction_error'],
                'user_feedback': self.feedback_data['user_feedback']
            }
            
            with open(self.storage_file, 'w') as f:
                json.dump(data_to_save, f)
                
            # self.logger.info(f"Saved {len(self.feedback_data['user_feedback'])} feedback entries")
        except Exception as e:
            self.logger.error(f"Error saving feedback data: {e}")
    
    def _process_single_feedback(self,
                               latent: Union[np.ndarray, torch.Tensor],
                               anomaly_decision: Union[bool, int, float, np.ndarray, torch.Tensor],
                               reconstruction_error: Union[float, np.ndarray, torch.Tensor],
                               user_feedback: Union[bool, int, float, np.ndarray, torch.Tensor]) -> tuple:
        """
        Process a single feedback item, converting types as needed.
        
        Args:
            latent: Latent representation from the autoencoder
            anomaly_decision: Whether the autoencoder flagged an anomaly
            reconstruction_error: Reconstruction error from the autoencoder
            user_feedback: User feedback (1 = true positive, 0 = false positive)
            
        Returns:
            tuple: Processed (latent, anomaly_decision, reconstruction_error, user_feedback)
        """
        # Convert torch tensors to numpy arrays
        if isinstance(latent, torch.Tensor):
            latent = latent.detach().cpu().numpy()
        if isinstance(anomaly_decision, torch.Tensor):
            anomaly_decision = anomaly_decision.item()
        if isinstance(reconstruction_error, torch.Tensor):
            reconstruction_error = reconstruction_error.item()
        if isinstance(user_feedback, torch.Tensor):
            user_feedback = user_feedback.item()

        # Convert boolean to int
        if isinstance(anomaly_decision, bool):
            anomaly_decision = int(anomaly_decision)
        if isinstance(user_feedback, bool):
            user_feedback = int(user_feedback)
            
        # Convert bool_ numpy arrays to int
        if isinstance(anomaly_decision, np.bool_):
            anomaly_decision = int(anomaly_decision.item())
        if isinstance(user_feedback, np.bool_):
            user_feedback = int(user_feedback.item())
            
        # Convert numpy.int64 to int
        if isinstance(user_feedback, np.int64): 
            user_feedback = int(user_feedback.item())
            
        return latent, anomaly_decision, reconstruction_error, user_feedback

    def add_feedback_batch(self,
                          latents: Sequence[Union[np.ndarray, torch.Tensor]],
                          anomaly_decisions: Sequence[Union[bool, int, float, np.ndarray, torch.Tensor]],
                          reconstruction_errors: Sequence[Union[float, np.ndarray, torch.Tensor]],
                          user_feedbacks: Sequence[Union[bool, int, float, np.ndarray, torch.Tensor]]) -> None:
        """
        Add multiple feedback items in a batch.
        
        Args:
            latents: List of latent representations from the autoencoder
            anomaly_decisions: List of autoencoder anomaly flags
            reconstruction_errors: List of reconstruction errors
            user_feedbacks: List of user feedback values (1 = true positive, 0 = false positive)
        """
        if not (len(latents) == len(anomaly_decisions) == len(reconstruction_errors) == len(user_feedbacks)):
            raise ValueError("All input sequences must have the same length")
            
        for latent, anomaly_decision, reconstruction_error, user_feedback in zip(
            latents, anomaly_decisions, reconstruction_errors, user_feedbacks):
            
            processed_latent, processed_anomaly, processed_recon, processed_feedback = self._process_single_feedback(
                latent, anomaly_decision, reconstruction_error, user_feedback
            )
            
            self.feedback_data['latent'].append(processed_latent)
            self.feedback_data['anomaly_decision'].append(processed_anomaly)
            self.feedback_data['reconstruction_error'].append(processed_recon)
            self.feedback_data['user_feedback'].append(processed_feedback)
        
        # Save all feedback at once
        self._save_data()
        
        self.logger.info(f"Added {len(latents)} feedback entries in batch")
    
    def add_feedback(self, 
                    latent: Union[np.ndarray, torch.Tensor], 
                    anomaly_decision: Union[bool, int, float, np.ndarray, torch.Tensor], 
                    reconstruction_error: Union[float, np.ndarray, torch.Tensor], 
                    user_feedback: Union[bool, int, float, np.ndarray, torch.Tensor]):
        """
        Add user feedback for an alert.
        
        Args:
            latent: Latent representation from the autoencoder
            anomaly_decision: Whether the autoencoder flagged an anomaly
            reconstruction_error: Reconstruction error from the autoencoder
            user_feedback: User feedback (1 = true positive, 0 = false positive)
        """
        self.add_feedback_batch(
            latents=[latent],
            anomaly_decisions=[anomaly_decision],
            reconstruction_errors=[reconstruction_error],
            user_feedbacks=[user_feedback]
        )
    
    def get_training_data(self) -> Dict[str, torch.Tensor]:
        """
        Get training data for the alert filter model.
        
        Returns:
            Dict[str, torch.Tensor]: Dictionary containing training data as torch tensors
        """
        if not self.feedback_data['user_feedback']:
            self.logger.warning("No feedback data available for training")
            return None
        
        # Process latent vectors to ensure they all have the same shape
        processed_latents = []
        for latent in self.feedback_data['latent']:
            # Convert to numpy array if it's not already
            if not isinstance(latent, np.ndarray):
                latent = np.array(latent)
            
            # If the latent vector is multi-dimensional, take the mean across the sequence dimension
            if latent.ndim > 1:
                latent = np.mean(latent, axis=0)
            
            # Ensure the latent vector has the correct dimension 
            target_dim = AlertFilterConfig.input_dim
            if len(latent) > target_dim:
                latent = latent[:target_dim]
            elif len(latent) < target_dim:
                latent = np.pad(latent, (0, target_dim - len(latent)))
            
            processed_latents.append(latent)
        
        # Convert data to torch tensors
        try:
            latent_tensor = torch.tensor(np.array(processed_latents), dtype=torch.float32)
            anomaly_decision_tensor = torch.tensor(np.array(self.feedback_data['anomaly_decision']), dtype=torch.float32)
            reconstruction_error_tensor = torch.tensor(np.array(self.feedback_data['reconstruction_error']), dtype=torch.float32)
            user_feedback_tensor = torch.tensor(np.array(self.feedback_data['user_feedback']), dtype=torch.float32)
            
            return {
                'latent': latent_tensor,
                'anomaly_decision': anomaly_decision_tensor,
                'reconstruction_error': reconstruction_error_tensor,
                'user_feedback': user_feedback_tensor
            }
        except Exception as e:
            self.logger.error(f"Error converting feedback data to tensors: {e}")
            # Print debug information
            for i, latent in enumerate(processed_latents):
                self.logger.info(f"Latent {i} shape: {latent.shape}")
            raise
    
    def get_stats(self) -> Dict[str, int]:
        """
        Get statistics about the feedback data.
        
        Returns:
            Dict[str, int]: Dictionary containing statistics
        """
        if not self.feedback_data['user_feedback']:
            return {
                'total': 0,
                'true_positives': 0,
                'false_positives': 0
            }
            
        user_feedback = np.array(self.feedback_data['user_feedback'])
        return {
            'total': len(user_feedback),
            'true_positives': int(np.sum(user_feedback == 1)),
            'false_positives': int(np.sum(user_feedback == 0))
        }
