import os
import torch
import numpy as np
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from typing import Dict, Optional, Tuple

from privateer_ad.config import AlertFilterAEConfig, PathsConf, setup_logger
from privateer_ad.models import AlertFilterAutoencoder
from privateer_ad.train_alert_filter.feedback_collector import FeedbackCollector

class AlertFilterAETrainer:
    """
    Trainer for the alert filter autoencoder model.
    
    This class handles training the alert filter autoencoder model based on user feedback,
    specifically focusing on false positives.
    
    Attributes:
        model (AlertFilterAutoencoder): The alert filter autoencoder model to train
        optimizer (torch.optim.Optimizer): The optimizer used for training
        criterion (nn.Module): The loss function used for training
        device (torch.device): The device to train on
        config (AlertFilterAEConfig): Configuration for training
    """
    
    def __init__(self, 
                 model: Optional[AlertFilterAutoencoder] = None,
                 config: Optional[AlertFilterAEConfig] = None,
                 device: Optional[torch.device] = None):
        """
        Initialize the alert filter autoencoder trainer.
        
        Args:
            model (AlertFilterAutoencoder, optional): The alert filter autoencoder model to train.
                If None, a new model is created.
            config (AlertFilterAEConfig, optional): Configuration for training.
                If None, default configuration is used.
            device (torch.device, optional): The device to train on.
                If None, uses CUDA if available, otherwise CPU.
        """
        self.config = config or AlertFilterAEConfig()
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model if not provided
        self.model = model or AlertFilterAutoencoder(config=self.config)
        self.model = self.model.to(self.device)
        
        # Initialize optimizer and loss function
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.config.learning_rate
        )
        self.criterion = nn.MSELoss()
        
        # Setup logger
        self.logger = setup_logger('alert-filter-ae-trainer')
        
        self.logger.info(f"Initialized AlertFilterAETrainer with device: {self.device}")
    
    def train(self, feedback_collector: FeedbackCollector, epochs: Optional[int] = None) -> Dict[str, float]:
        """
        Train the alert filter autoencoder model based on user feedback.
        Only uses false positives for training.
        
        Args:
            feedback_collector (FeedbackCollector): The feedback collector containing training data
            epochs (int, optional): Number of epochs to train for.
                If None, uses the value from config.
                
        Returns:
            Dict[str, float]: Dictionary containing training metrics
        """
        # Get training data
        training_data = feedback_collector.get_training_data()
        if training_data is None:
            self.logger.warning("No feedback data available for training")
            return {'loss': float('inf')}
        
        # Extract only false positives (where user_feedback == 0)
        false_positive_mask = training_data['user_feedback'] == 0
        
        # Check if we have any false positives
        if not torch.any(false_positive_mask):
            self.logger.warning("No false positives available for training")
            return {'loss': float('inf')}
        
        # Extract false positive data
        fp_latent = training_data['latent'][false_positive_mask]
        
        # Create dataset and dataloader
        dataset = TensorDataset(fp_latent)
        
        dataloader = DataLoader(
            dataset, 
            batch_size=self.config.batch_size, 
            shuffle=True
        )
        
        # Set number of epochs
        epochs = epochs or self.config.epochs
        
        # Training loop
        self.model.train()
        metrics = {'loss': 0.0}
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            
            for (latent,) in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
                # Move data to device
                latent = latent.to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                
                reconstructed, _ = self.model(latent)
                
                # Calculate loss
                loss = self.criterion(reconstructed, latent)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
            
            # Calculate average loss for epoch
            avg_loss = epoch_loss / len(dataloader)
            metrics['loss'] = avg_loss
            
            self.logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
        
        # After training, calculate reconstruction error statistics
        # to set the threshold for false positive detection
        self.set_reconstruction_threshold(dataloader)
        
        return metrics
    
    def set_reconstruction_threshold(self, dataloader: DataLoader) -> None:
        """
        Calculate reconstruction error statistics and set the threshold.
        
        The threshold is set based on the distribution of reconstruction errors
        for known false positives. We use mean + 2*std as a default threshold.
        
        Args:
            dataloader (DataLoader): DataLoader containing false positive data
        """
        self.model.eval()
        reconstruction_errors = []
        
        with torch.no_grad():
            for (latent,) in dataloader:
                latent = latent.to(self.device)
                reconstructed, _ = self.model(latent)
                
                # Calculate reconstruction error
                error = torch.mean((reconstructed - latent) ** 2, dim=1)
                reconstruction_errors.extend(error.cpu().numpy())
        
        # Calculate statistics
        errors = np.array(reconstruction_errors)
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        
        
       
        # Set threshold at some percentile of errors
        threshold = np.percentile(errors, 38.6)
        
        # Update model config with the calculated threshold
        self.config.reconstruction_threshold = float(threshold)
        self.model.config.reconstruction_threshold = float(threshold)
        
        self.logger.info(f"Set reconstruction error threshold to {threshold:.6f} (mean: {mean_error:.6f}, std: {std_error:.6f})")
    
    def save_model(self, path: Optional[str] = None) -> str:
        """
        Save the alert filter autoencoder model.
        
        Args:
            path (str, optional): Path to save the model to.
                If None, uses the default path.
                
        Returns:
            str: Path where the model was saved
        """
        paths = PathsConf()
        if path is None:
            # Create directory for alert filter models if it doesn't exist
            alert_filter_dir = paths.root.joinpath('alert_filter_models')
            os.makedirs(alert_filter_dir, exist_ok=True)
            
            # Generate filename with timestamp
            timestamp = torch.datetime.now().strftime("%Y%m%d-%H%M%S")
            path = alert_filter_dir.joinpath(f'alert_filter_ae_{timestamp}.pt')
        
        # Save model state dict and config
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'config': self.config
        }
        
        # Save model
        torch.save(save_dict, path)
        self.logger.info(f"Saved alert filter autoencoder model to {path}")
        
        return str(path)
    
    @classmethod
    def load_model(cls, 
                  path: str, 
                  device: Optional[torch.device] = None) -> 'AlertFilterAETrainer':
        """
        Load an alert filter autoencoder model from a saved state dict.
        
        Args:
            path (str): Path to the saved model state dict
            device (torch.device, optional): The device to load the model on.
                If None, uses CUDA if available, otherwise CPU.
                
        Returns:
            AlertFilterAETrainer: Trainer with the loaded model
        """
        # Initialize device
        device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load saved dictionary
        save_dict = torch.load(path, map_location=device)
        
        # Extract config and state dict
        config = save_dict.get('config', AlertFilterAEConfig())
        state_dict = save_dict.get('model_state_dict', None)
        
        if state_dict is None:
            # For backward compatibility with older saved models
            state_dict = save_dict
        
        # Create model and load state dict
        model = AlertFilterAutoencoder(config=config)
        model.load_state_dict(state_dict)
        
        # Create trainer with loaded model
        trainer = cls(model=model, config=config, device=device)
        logger = setup_logger('alert-filter-ae-trainer')
        logger.info(f"Loaded alert filter autoencoder model from {path}")
        
        return trainer
