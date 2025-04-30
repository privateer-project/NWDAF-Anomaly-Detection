import os
import torch
import mlflow
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from typing import Dict, Optional

from privateer_ad.config import AlertFilterConfig, PathsConf, logger
from privateer_ad.models import AlertFilterModel
from privateer_ad.train.feedback_collector import FeedbackCollector

class AlertFilterTrainer:
    """
    Trainer for the alert filter model.
    
    This class handles training the alert filter model based on user feedback.
    
    Attributes:
        model (AlertFilterModel): The alert filter model to train
        optimizer (torch.optim.Optimizer): The optimizer used for training
        criterion (nn.Module): The loss function used for training
        device (torch.device): The device to train on
        config (AlertFilterConfig): Configuration for training
    """
    
    def __init__(self, 
                 model: Optional[AlertFilterModel] = None,
                 config: Optional[AlertFilterConfig] = None,
                 device: Optional[torch.device] = None):
        """
        Initialize the alert filter trainer.
        
        Args:
            model (AlertFilterModel, optional): The alert filter model to train.
                If None, a new model is created.
            config (AlertFilterConfig, optional): Configuration for training.
                If None, default configuration is used.
            device (torch.device, optional): The device to train on.
                If None, uses CUDA if available, otherwise CPU.
        """
        self.config = config or AlertFilterConfig()
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model if not provided
        self.model = model or AlertFilterModel(config=self.config)
        self.model = self.model.to(self.device)
        
        # Initialize optimizer and loss function
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.config.learning_rate
        )
        self.criterion = nn.BCELoss()
        
        logger.info(f"Initialized AlertFilterTrainer with device: {self.device}")
    
    def train(self, feedback_collector: FeedbackCollector, epochs: Optional[int] = None) -> Dict[str, float]:
        """
        Train the alert filter model based on user feedback.
        
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
            logger.warning("No feedback data available for training")
            return {'loss': float('inf')}
        
        # Create dataset and dataloader
        dataset = TensorDataset(
            training_data['latent'],
            training_data['anomaly_decision'],
            training_data['reconstruction_error'],
            training_data['user_feedback']
        )
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
            
            for latent, decision, error, feedback in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
                # Move data to device
                latent = latent.to(self.device)
                decision = decision.to(self.device)
                error = error.to(self.device)
                feedback = feedback.to(self.device).unsqueeze(-1)  # Add dimension for BCE loss
                
                # Forward pass
                self.optimizer.zero_grad()
                output = self.model(latent, decision, error)
                
                # Calculate loss
                loss = self.criterion(output, feedback)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
            
            # Calculate average loss for epoch
            avg_loss = epoch_loss / len(dataloader)
            metrics['loss'] = avg_loss
            
            logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
                    
        return metrics
    
    def save_model(self, path: Optional[str] = None) -> str:
        """
        Save the alert filter model.
        
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
            path = alert_filter_dir.joinpath(f'alert_filter_{timestamp}.pt')
        
        # Save model state dict
        torch.save(self.model.state_dict(), path)
        logger.info(f"Saved alert filter model to {path}")
        
        return str(path)
    
    @classmethod
    def load_model(cls, 
                  path: str, 
                  config: Optional[AlertFilterConfig] = None,
                  device: Optional[torch.device] = None) -> 'AlertFilterTrainer':
        """
        Load an alert filter model from a saved state dict.
        
        Args:
            path (str): Path to the saved model state dict
            config (AlertFilterConfig, optional): Configuration for the model.
                If None, default configuration is used.
            device (torch.device, optional): The device to load the model on.
                If None, uses CUDA if available, otherwise CPU.
                
        Returns:
            AlertFilterTrainer: Trainer with the loaded model
        """
        # Initialize configuration and device
        config = config or AlertFilterConfig()
        device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create model and load state dict
        model = AlertFilterModel(config=config)
        model.load_state_dict(torch.load(path, map_location=device))
        
        # Create trainer with loaded model
        trainer = cls(model=model, config=config, device=device)
        logger.info(f"Loaded alert filter model from {path}")
        
        return trainer
