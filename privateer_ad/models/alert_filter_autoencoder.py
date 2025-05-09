import torch
from torch import nn
from privateer_ad.config.hparams_config import AlertFilterAEConfig

class AlertFilterAutoencoder(nn.Module):
    """
    Alert filter autoencoder for reducing false positives in anomaly detection.
    
    This model takes the latent representation from the main autoencoder as input,
    and learns to reconstruct false positives. The reconstruction error is used
    to determine if a new alert is likely a false positive (low error) or a true
    positive (high error).
    
    Attributes:
        config (AlertFilterAEConfig): Configuration for the alert filter autoencoder
        encoder (nn.Sequential): Encoder neural network layers
        decoder (nn.Sequential): Decoder neural network layers
    """
    
    def __init__(self, config: AlertFilterAEConfig):
        super(AlertFilterAutoencoder, self).__init__()
        self.config = config
        
        # Build encoder
        encoder_layers = []
        prev_dim = self.config.input_dim
        
        for hidden_dim in self.config.hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            encoder_layers.append(nn.ReLU())
            encoder_layers.append(nn.Dropout(self.config.dropout))
            prev_dim = hidden_dim
            
        # Final encoder layer to latent dimension
        encoder_layers.append(nn.Linear(prev_dim, self.config.latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Build decoder (reverse of encoder)
        decoder_layers = []
        prev_dim = self.config.latent_dim
        
        # Go through hidden dimensions in reverse
        for hidden_dim in reversed(self.config.hidden_dims):
            decoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            decoder_layers.append(nn.ReLU())
            decoder_layers.append(nn.Dropout(self.config.dropout))
            prev_dim = hidden_dim
            
        # Final decoder layer back to input dimension
        decoder_layers.append(nn.Linear(prev_dim, self.config.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
        
    def forward(self, x):
        """
        Forward pass of the alert filter autoencoder.
        
        Args:
            x (torch.Tensor): Latent representation from the main autoencoder
                Shape: [batch_size, input_dim]
                
        Returns:
            tuple: (reconstructed, latent)
                - reconstructed (torch.Tensor): Reconstructed input
                - latent (torch.Tensor): Latent representation
        """
        # Ensure inputs have the right shape
        if x.dim() == 3:  # If x has shape [batch_size, seq_len, input_dim]
            # Take the mean across the sequence dimension
            x = torch.mean(x, dim=1)
            
        # Encode
        latent = self.encoder(x)
        
        # Decode
        reconstructed = self.decoder(latent)
        
        return reconstructed, latent
    
    def get_reconstruction_error(self, x):
        """
        Calculate reconstruction error for input.
        
        Args:
            x (torch.Tensor): Latent representation from the main autoencoder
                
        Returns:
            torch.Tensor: Mean squared error between input and reconstruction
        """
        # Forward pass
        reconstructed, _ = self.forward(x)
        
        # Calculate MSE
        mse = torch.mean((reconstructed - x) ** 2, dim=1)
        
        return mse
    
    def is_false_positive(self, x):
        """
        Determine if an alert is likely a false positive based on reconstruction error.
        
        Args:
            x (torch.Tensor): Latent representation from the main autoencoder
                
        Returns:
            torch.Tensor: Boolean tensor indicating if alert is a false positive
                (True = false positive, False = true positive)
        """
        # Get reconstruction error
        rec_error = self.get_reconstruction_error(x)
        
        # Compare to threshold
        if self.config.reconstruction_threshold is None:
            # Default behavior if threshold not set: assume not false positive
            return torch.zeros_like(rec_error, dtype=torch.bool)
        
        # Low reconstruction error indicates false positive
        return rec_error < self.config.reconstruction_threshold
