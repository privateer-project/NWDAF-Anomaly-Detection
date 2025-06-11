from torch import nn
from opacus.layers import DPMultiheadAttention

from privateer_ad.architectures.layers import PositionalEncoding
from privateer_ad.config import ModelConfig


class TransformerAD(nn.Module):
    """
    Transformer-based reconstruction model for privacy-preserving anomaly detection.

    This implementation leverages transformer attention mechanisms to learn temporal
    patterns in sequential network data and reconstruct input sequences. The architecture
    incorporates differential privacy through specialized attention layers while maintaining
    the temporal modeling capabilities essential for network traffic analysis.

    Attributes:
        model_config (ModelConfig): Configuration parameters controlling model architecture
        embed (nn.Linear): Input feature embedding layer transforming raw features
                         to model embedding dimension
        pos_enc (PositionalEncoding): Sinusoidal positional encoding for temporal
                                    sequence understanding
        transformer_encoder (nn.TransformerEncoder): Multi-layer transformer encoder
                                                   with privacy-preserving attention
        compress (nn.Sequential): Compression pathway reducing encoded representations
                                to latent space dimensions
        output (nn.Linear): Reconstruction layer projecting latent representations
                          back to original feature space
    """
    def __init__(self, model_config: ModelConfig = None):
        """
        Initialize the transformer-based anomaly detection architecture.

        The transformer encoder receives specialized differential privacy
        attention layers that replace standard multi-head attention to ensure
        privacy guarantees during federated training scenarios. Layer normalization
        and feed-forward dimensions are configured to balance representational
        capacity with computational efficiency requirements.

        Args:
            model_config (ModelConfig, optional): Architecture configuration specifying embedding dimensions, attention
                                                  heads, layer counts, and privacy parameters.
                                                  If None, uses default configuration with standard transformer settings.
        """
        super(TransformerAD, self).__init__()

        self.model_config = model_config or ModelConfig()

        # Input feature embedding to model dimension
        self.embed = nn.Linear(self.model_config.input_size, self.model_config.embed_dim)

        # Positional encoding for temporal sequence understanding
        self.pos_enc = PositionalEncoding(d_model=self.model_config.embed_dim,
                                          max_seq_length=self.model_config.seq_len,
                                          dropout=self.model_config.dropout)

        # Configure transformer encoder layer with privacy-preserving attention
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.model_config.embed_dim,
            nhead=self.model_config.num_heads,
            dim_feedforward=self.model_config.latent_dim,
            batch_first=True
        )

        # Replace standard attention with opacus, privacy compatible, implementation
        encoder_layer.self_attn = DPMultiheadAttention(
            embed_dim=self.model_config.embed_dim,
            num_heads=self.model_config.num_heads,
            dropout=self.model_config.dropout,
            batch_first=True)

        # Multi-layer transformer encoder with layer normalization
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.model_config.num_layers,
            norm=nn.LayerNorm(self.model_config.embed_dim)
        )

        # Compression path to latent representation
        self.compress = nn.Sequential(nn.Linear(self.model_config.embed_dim, self.model_config.latent_dim),
                                      nn.ReLU())
        # Reconstruction output layer
        self.output = nn.Linear(self.model_config.latent_dim, self.model_config.input_size)

    def forward(self, x):
        """
        Execute forward pass for sequence reconstruction and anomaly detection.

        Processes input network traffic sequences through the transformer-based
        reconstruction pipeline, producing reconstructed outputs for anomaly scoring.
        The forward pass maintains temporal relationships through positional
        encoding while leveraging attention mechanisms to capture complex
        feature interactions within the sequence.

        The reconstruction quality serves as the primary anomaly indicator, where
        high reconstruction errors suggest anomalous behavior patterns that
        deviate from learned benign traffic characteristics. The privacy-preserving
        attention mechanisms ensure that individual sequence contributions remain
        protected throughout the processing pipeline.

        Processing stages include:
        1. Feature embedding to model dimension space
        2. Positional encoding addition for temporal awareness
        3. Multi-layer transformer encoding with privacy-preserving attention
        4. Latent space compression through ReLU activation
        5. Linear reconstruction to original feature space

        Args:
            x (torch.Tensor): Input sequences of shape [batch_size, seq_length, input_size]
                            representing network traffic feature vectors over time.
                            Each sequence contains temporal measurements of network
                            metrics such as throughput, latency, and connection patterns.

        Returns:
            torch.Tensor: Reconstructed sequences of identical shape to input
                         [batch_size, seq_length, input_size]. Reconstruction
                         quality indicates anomaly likelihood, with higher
                         reconstruction errors suggesting anomalous patterns.

        Note:
            The model operates in reconstruction mode where input and output dimensions
            match, enabling direct reconstruction error computation for anomaly
            scoring. During training, reconstruction loss drives the learning of
            normal traffic pattern representations.
        """
        # Transform input features to embedding dimension
        x = self.embed(x)
        # Add positional information for temporal sequence modeling
        x = self.pos_enc(x)
        # Process through a multi-layer transformer encoder with DP attention
        x = self.transformer_encoder(x)
        # Compress to latent representation with non-linear activation
        x = self.compress(x)
        # Reconstruct to the original feature space
        x = self.output(x)
        return x
