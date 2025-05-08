import torch
from torch import nn
from privateer_ad.config.hparams_config import AlertFilterConfig

class AlertFilterModel(nn.Module):
    """
    Alert filter model for reducing false positives in anomaly detection.
    
    This model takes the latent representation from the autoencoder, the anomaly decision,
    and the reconstruction error as inputs, and outputs a decision on whether to allow
    or deny the alert. The model starts by allowing all alerts (free pass) and learns
    from user feedback which alerts are false positives.
    
    Attributes:
        config (AlertFilterConfig): Configuration for the alert filter model
        layers (nn.Sequential): Neural network layers
        output (nn.Linear): Output layer
    """
    
    def __init__(self, config: AlertFilterConfig):
        super(AlertFilterModel, self).__init__()
        self.config = config
        
        # Input: latent representation + anomaly decision + reconstruction error
        input_dim = self.config.input_dim + 2
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in self.config.hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.config.dropout))
            prev_dim = hidden_dim
            
        # Output layer - initially biased to allow all alerts
        self.layers = nn.Sequential(*layers)
        self.output = nn.Linear(prev_dim, 1)
        
        # Initialize with bias toward allowing alerts (high positive bias for sigmoid)
        self.output.bias.data.fill_(5.0)
        
    def forward(self, latent, anomaly_decision, reconstruction_error):
        """
        Forward pass of the alert filter model.
        
        Args:
            latent (torch.Tensor): Latent representation from the autoencoder
            anomaly_decision (torch.Tensor): Binary tensor indicating anomaly detection (1 = anomaly, 0 = normal)
            reconstruction_error (torch.Tensor): Reconstruction error from the autoencoder
            
        Returns:
            torch.Tensor: Probability of allowing the alert (1 = allow, 0 = deny)
        """
        # Ensure inputs have the right shape
        if latent.dim() == 3:  # If latent has shape [batch_size, seq_len, input_dim]
            # Take the mean across the sequence dimension
            latent = torch.mean(latent, dim=1)
            
        # Concatenate inputs
        x = torch.cat([
            latent,
            anomaly_decision.unsqueeze(-1).float(),
            reconstruction_error.unsqueeze(-1).float()
        ], dim=-1)
        
        x = self.layers(x)
        # Sigmoid to get probability of allowing the alert
        return torch.sigmoid(self.output(x))
