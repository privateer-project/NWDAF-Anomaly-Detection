"""Model Training Pipeline

This module handles the complete training workflow: data loading,
normalization, training loop, early stopping, and threshold computation.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from hitl.store.repository import Repository
from hitl.artifacts.manager import Artifacts
from hitl.settings import Config
from hitl.types import TrainParams
from hitl.io.serialization import decode_npy
from hitl.models.ae import build_model
from hitl.utils.time import now_iso


# Helper functions


def mse_per_sample(x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
    """Compute MSE per sample (not averaged over batch).

    Args:
        x: Original tensor (B, ...)
        x_hat: Reconstructed tensor (B, ...)

    Returns:
        Tensor of shape (B,) with MSE for each sample
    """
    # Squared error
    se = (x - x_hat) ** 2

    # Sum over all dimensions except batch (dim 0)
    # For dense (B, D): sum over dim=1
    # For conv1d (B, F, T): sum over dims=(1,2)
    dims = tuple(range(1, se.ndim))
    mse = se.sum(dim=dims)

    # Divide by number of elements per sample
    n_elements = np.prod(x.shape[1:])
    mse = mse / n_elements

    return mse


def get_device() -> torch.device:
    """Get appropriate PyTorch device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Trainer:
    """Handles training of autoencoder models on labeled anomaly data."""

    def __init__(
        self, repository: Repository, artifacts: Artifacts, config: Config, logger
    ):
        """Initialize trainer with dependencies.

        Args:
            repository: Database repository for loading vectors
            artifacts: Artifact manager for saving model
            config: System configuration
            logger: Structured logger instance
        """
        self.repo = repository
        self.artifacts = artifacts
        self.config = config
        self.logger = logger

    def load_dataset(
        self, schema_id: str, mode: str
    ) -> tuple[np.ndarray, list[str], tuple[int, ...]]:
        """Load all vectors for a schema from database.

        Args:
            schema_id: Feature schema to load
            mode: "dense" or "conv1d" - determines shape transformation

        Returns:
            (X, ids, storage_sample_shape) tuple where:
            - X: Data array
              - Dense mode: X has shape (N, D)
              - Conv1d mode: X has shape (N, F, T) - transposed from storage format
            - ids: List of anomaly_ids
            - storage_sample_shape: Original sample shape in storage format
              - Dense: (D,)
              - Conv1d: (T, F)

        Shape Convention:
            Storage format (per sample):
                - Dense: (D,) - feature vector
                - Conv1d: (T, F) - time-series with T timesteps, F features
            
            Training format (batch):
                - Dense: (N, D) - no transformation needed
                - Conv1d: (N, F, T) - transposed from (N, T, F) for PyTorch Conv1d
                  PyTorch Conv1d expects channels-first format (B, C, L)

        Raises:
            ValueError: if no vectors found for schema
        """
        self.logger.info("Loading dataset", schema_id=schema_id, mode=mode)

        vectors = []
        ids = []

        # Iterate over all vectors for this schema
        for anomaly_id, blob in self.repo.iter_vectors(schema_id):
            # Decode the .npy blob
            arr = decode_npy(blob)
            vectors.append(arr)
            ids.append(anomaly_id)

        if not vectors:
            raise ValueError(f"No vectors found for schema {schema_id}")

        # Stack into single array
        X = np.stack(vectors, axis=0)

        # Store original sample shape (storage format) for model building
        storage_sample_shape = X.shape[1:]  # Shape per sample in storage format

        # For Conv1d mode, transpose from storage format (N, T, F) to training format (N, F, T)
        # PyTorch Conv1d expects (Batch, Channels, Length) = (N, F, T)
        if mode == "conv1d" and X.ndim == 3:
            X = np.transpose(X, (0, 2, 1))  # (N, T, F) -> (N, F, T)
            self.logger.info(
                "Transposed data for Conv1d", 
                storage_format="(N, T, F)", 
                training_format="(N, F, T)"
            )

        self.logger.info(
            "Dataset loaded", 
            schema_id=schema_id, 
            n_samples=len(vectors), 
            shape=X.shape,
            mode=mode
        )

        return X, ids, storage_sample_shape

    def fit(
        self,
        X: np.ndarray,
        params: TrainParams,
        storage_sample_shape: tuple[int, ...] | None = None,
    ) -> tuple[dict, dict, dict]:
        """Train autoencoder on data.

        Args:
            X: Training data array (raw, no preprocessing)
               For Conv1d, should already be transposed to (N, F, T)
            params: Training parameters
            storage_sample_shape: Original sample shape in storage format
                                 If None, will use X.shape[1:]

        Returns:
            (state_dict, threshold, metrics) tuple
        """
        self.logger.info("Starting training", shape=X.shape, params=params)

        # Extract parameters
        mode = params["mode"]
        epochs = params["epochs"]
        batch_size = params["batch_size"]
        lr = params["lr"]
        val_split = params["val_split"]
        patience = params["patience"]
        percentile = params["percentile"]

        # Use provided storage shape or infer from X
        if storage_sample_shape is None:
            storage_sample_shape = X.shape[1:]

        device = get_device()
        self.logger.info("Using device", device=str(device))

        # 1. Split data
        X_train, X_val = self._split_data(X, val_split)
        self.logger.info("Data split", train_shape=X_train.shape, val_shape=X_val.shape)

        # 2. Build model using STORAGE format shape (before any transpose)
        #    build_model expects storage format: Dense=(D,), Conv1d=(T,F)
        model = build_model(mode, storage_sample_shape)
        model = model.to(device)

        n_params = sum(p.numel() for p in model.parameters())
        self.logger.info(
            "Model built", mode=mode, input_shape=storage_sample_shape, n_parameters=n_params
        )

        # 3. Setup optimizer and loss
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # 4. Training loop with early stopping
        train_losses = []
        val_losses = []
        best_val_loss = float("inf")
        best_state_dict = None
        epochs_without_improvement = 0

        for epoch in range(epochs):
            # Train
            train_loss = self._train_epoch(
                model, optimizer, X_train, batch_size, device
            )
            train_losses.append(train_loss)

            # Validate
            val_loss = self._validate(model, X_val, batch_size, device)
            val_losses.append(val_loss)

            # Check for improvement
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state_dict = {
                    k: v.cpu().clone() for k, v in model.state_dict().items()
                }
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            # Log progress
            if (epoch + 1) % 10 == 0 or epoch == 0:
                self.logger.info(
                    "Epoch",
                    epoch=epoch + 1,
                    train_loss=f"{train_loss:.6f}",
                    val_loss=f"{val_loss:.6f}",
                    best_val=f"{best_val_loss:.6f}",
                )

            # Early stopping
            if epochs_without_improvement >= patience:
                self.logger.info(
                    "Early stopping triggered", epoch=epoch + 1, patience=patience
                )
                break

        # Restore best model
        if best_state_dict is not None:
            model.load_state_dict(best_state_dict)

        # 5. Compute threshold on train set (raw data, no normalization)
        threshold = self._compute_threshold(model, X_train, percentile, device)

        self.logger.info(
            "Training complete",
            epochs=epoch + 1,
            best_val_loss=f"{best_val_loss:.6f}",
            threshold=f"{threshold['value']:.6f}",
        )

        # 6. Prepare return values
        state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
        metrics = {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "best_val_loss": best_val_loss,
            "epochs_trained": epoch + 1,
            "final_train_loss": train_losses[-1] if train_losses else None,
            "final_val_loss": val_losses[-1] if val_losses else None,
        }

        return state_dict, threshold, metrics

    def train_and_publish(self, schema_id: str, params: TrainParams) -> str:
        """Complete training workflow: load → train → save → register.

        Args:
            schema_id: Feature schema to train on
            params: Training parameters

        Returns:
            model_version string
        """
        self.logger.info(
            "Starting train_and_publish", schema_id=schema_id, mode=params["mode"]
        )

        # 1. Load dataset (with appropriate shape transformation for mode)
        X, ids, storage_sample_shape = self.load_dataset(schema_id, mode=params["mode"])

        # 2. Train model (no normalization, train on raw data)
        state_dict, threshold, metrics = self.fit(X, params, storage_sample_shape)

        # 3. Create artifact version using STORAGE format shape
        model_version, artifact_path = self.artifacts.create_version(
            params["mode"], storage_sample_shape
        )

        self.logger.info(
            "Artifact version created", model_version=model_version, path=artifact_path
        )

        # 4. Save artifacts (no scaler - training on raw data)
        self.artifacts.save_model(model_version, state_dict)

        config = {
            "mode": params["mode"],
            "input_shape": list(storage_sample_shape),
            "dtype": "float32",
        }
        self.artifacts.save_config(model_version, config)

        self.artifacts.save_threshold(model_version, threshold)

        self.logger.info("All artifacts saved")

        # 5. Register model in database
        self.repo.insert_model(
            model_version=model_version,
            kind=params["mode"],  # "dense" or "conv1d"
            schema_id=schema_id,
            artifact_path=artifact_path,
            created_at=now_iso(),
        )

        self.logger.info("Model registered", model_version=model_version)

        return model_version

    def _split_data(
        self, X: np.ndarray, val_split: float = 0.1, shuffle: bool = True
    ) -> tuple[np.ndarray, np.ndarray]:
        """Split data into train and validation sets.

        Args:
            X: Full dataset
            val_split: Fraction for validation
            shuffle: Whether to shuffle before split

        Returns:
            (X_train, X_val) tuple
        """
        n_samples = X.shape[0]
        n_val = int(n_samples * val_split)
        n_train = n_samples - n_val

        if shuffle:
            indices = np.random.permutation(n_samples)
            X = X[indices]

        X_train = X[:n_train]
        X_val = X[n_train:]

        return X_train, X_val

    def _train_epoch(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        X_train: np.ndarray,
        batch_size: int,
        device: torch.device,
    ) -> float:
        """Run one training epoch.

        Args:
            model: PyTorch model
            optimizer: Optimizer
            X_train: Training data (raw, no preprocessing)
            batch_size: Batch size
            device: CPU or CUDA device

        Returns:
            Average training loss
        """
        model.train()

        # Shuffle data
        indices = np.random.permutation(len(X_train))
        X_shuffled = X_train[indices]

        total_loss = 0.0
        n_batches = 0

        # Iterate over batches
        for i in range(0, len(X_shuffled), batch_size):
            batch = X_shuffled[i : i + batch_size]

            # Convert to tensor
            x = torch.FloatTensor(batch).to(device)

            # Forward pass
            x_hat = model(x)

            # Compute loss
            loss = nn.functional.mse_loss(x_hat, x)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches if n_batches > 0 else 0.0
        return avg_loss

    def _validate(
        self, model: nn.Module, X_val: np.ndarray, batch_size: int, device: torch.device
    ) -> float:
        """Run validation pass.

        Args:
            model: PyTorch model
            X_val: Validation data (raw, no preprocessing)
            batch_size: Batch size
            device: Device

        Returns:
            Average validation loss
        """
        model.eval()

        total_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            # Iterate over batches
            for i in range(0, len(X_val), batch_size):
                batch = X_val[i : i + batch_size]

                # Convert to tensor
                x = torch.FloatTensor(batch).to(device)

                # Forward pass
                x_hat = model(x)

                # Compute loss
                loss = nn.functional.mse_loss(x_hat, x)

                total_loss += loss.item()
                n_batches += 1

        avg_loss = total_loss / n_batches if n_batches > 0 else 0.0
        return avg_loss

    def _compute_threshold(
        self,
        model: nn.Module,
        X_train: np.ndarray,
        percentile: float,
        device: torch.device,
    ) -> dict:
        """Compute anomaly detection threshold.

        Args:
            model: Trained model
            X_train: Training data (raw, no preprocessing)
            percentile: Threshold percentile (e.g., 99.5)
            device: Device

        Returns:
            Threshold dict
        """
        model.eval()

        reconstruction_errors = []

        with torch.no_grad():
            # Compute reconstruction for all samples
            # Process in batches to avoid memory issues
            batch_size = 256
            for i in range(0, len(X_train), batch_size):
                batch = X_train[i : i + batch_size]
                x = torch.FloatTensor(batch).to(device)
                x_hat = model(x)

                # Compute per-sample MSE
                mse = mse_per_sample(x, x_hat)
                reconstruction_errors.extend(mse.cpu().numpy())

        # Compute threshold as percentile
        reconstruction_errors = np.array(reconstruction_errors)
        threshold_value = np.percentile(reconstruction_errors, percentile)

        threshold = {
            "value": float(threshold_value),
            "percentile": float(percentile),
            "strategy": "percentile",
            "computed_at": now_iso(),
        }

        return threshold

    def _create_dataloader(
        self, X: np.ndarray, batch_size: int, shuffle: bool = False
    ) -> DataLoader:
        """Create PyTorch DataLoader from NumPy array.

        Args:
            X: Data array
            batch_size: Batch size
            shuffle: Whether to shuffle

        Returns:
            DataLoader
        """
        # Convert to tensor
        tensor = torch.FloatTensor(X)

        # Create dataset and dataloader
        dataset = TensorDataset(tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

        return dataloader
