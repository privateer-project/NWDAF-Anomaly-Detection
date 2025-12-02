import torch
import json
import numpy as np
import sys
from pathlib import Path

# Add project root to path to import hitl modules
# current_file = Path(__file__).resolve()

project_root = Path(__file__).resolve().parent
# project_root = current_file.parent.parent
# sys.path.append(str(project_root))

# from hitl.models.ae import build_model
try:
    from .utils.ae import build_model
except ImportError:
    from utils.ae import build_model

def load_model(model_dir):
    model_dir = Path(model_dir)
    config_path = model_dir / "dense_autoencoder_config.json"
    model_path = model_dir / "dense_autoencoder.pth"

    if not config_path.exists() or not model_path.exists():
        raise FileNotFoundError(f"Model files not found in {model_dir}")

    with open(config_path, "r") as f:
        config = json.load(f)

    # Reconstruct model
    # Note: input_shape expects a tuple. For dense it is (D,)
    model = build_model(
        mode=config["mode"],
        input_shape=(config["input_dim"],),
        hidden_dims=config["hidden_dims"],
        latent_dim=config["latent_dim"]
    )
    
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model, config

def run_inference(model, data):
    """
    Run inference on data (N, D).
    Returns reconstruction error (MSE) per sample.
    """
    with torch.no_grad():
        # Convert to tensor
        x = torch.from_numpy(data).float()
        
        # Forward pass (reconstruction)
        x_hat = model(x)
        
        # Calculate MSE per sample
        mse = torch.mean((x - x_hat) ** 2, dim=1)
        
    return mse.numpy(), x_hat.numpy()

def flatten_data(data, target_dim):
    """
    Flattens input data (N, ...) to (N, D) where D = target_dim.
    Raises ValueError if dimensions don't match.
    """
    # Get batch size
    n_samples = data.shape[0]
    
    # Flatten all other dimensions
    data_flat = data.reshape(n_samples, -1)
    
    # Check if dimensions match
    if data_flat.shape[1] != target_dim:
        raise ValueError(f"Input data shape {data.shape} flattens to {data_flat.shape[1]} features, but expected {target_dim}")
        
    return data_flat.astype(np.float32)

def predict(data: np.ndarray, model_dir: str | Path = None) -> list[dict]:
    """
    Main entry point for inference.
    
    Args:
        data: Input data array. Can be (N, D) or (N, T, F).
        model_dir: Directory containing the 'champion' folder with model files.
                   Defaults to 'saved_models' in the project root.
                   
    Returns:
        List of dictionaries containing inference results per samp`le.
    """
    if model_dir is None:
        model_dir = project_root / "model"
    
    # Load model and config
    model, config = load_model(model_dir)
    
    # Preprocess data
    input_dim = config["input_dim"]
    data_flat = flatten_data(data, input_dim)
    
    # Run inference
    scores, reconstructions = run_inference(model, data_flat)
    
    # Apply threshold
    threshold = config.get("threshold", 0.5)
    
    results = []
    for i, score in enumerate(scores):
        results.append({
            "is_anomaly": bool(score > threshold),
            "score": float(score),
            "threshold": threshold,
            # "reconstruction": reconstructions[i]
        })
        
    return results

if __name__ == "__main__":
    # Example usage
    save_dir = project_root / "model"
    
    if not save_dir.exists():
        print(f"Error: Model directory {save_dir} not found. Run the notebook first to train and save the model.")
        sys.exit(1)

    # Load config just to get input_dim for dummy data generation
    # In a real scenario, you would just pass your data to predict()
    try:
        _, config = load_model(save_dir)
        input_dim = config["input_dim"]
    except Exception as e:
        print(f"Failed to load model config: {e}")
        sys.exit(1)
    
    # Try to load real data from project data directory
    data_path = project_root.parent / "data" / "final_anomalies.npz"
    
    if data_path.exists():
        print(f"Loading data from {data_path}...")
        try:
            with np.load(data_path) as data:
                if 'X' in data:
                    X = data['X']  # Shape: (N, T, F)
                    
                    try:
                        # We don't need to flatten manually anymore, predict handles it
                        # Just selecting random samples
                        indices = np.random.choice(X.shape[0], 5, replace=False)
                        dummy_data = X[indices]
                        print(f"Successfully loaded real data. Using 5 samples. Shape: {dummy_data.shape}")
                    except ValueError as e:
                        print(f"Data mismatch: {e}. Using random data.")
                        dummy_data = np.random.randn(5, input_dim).astype(np.float32)
                else:
                    print("Key 'X' not found in data file. Using random data.")
                    dummy_data = np.random.randn(5, input_dim).astype(np.float32)
        except Exception as e:
            print(f"Error loading data: {e}. Using random data.")
            dummy_data = np.random.randn(5, input_dim).astype(np.float32)
    else:
        print(f"Data file not found at {data_path}. Generating dummy data with dimension {input_dim}...")
        # Create 5 random samples
        dummy_data = np.random.randn(5, input_dim).astype(np.float32)
    
    print("Running inference...")
    
    try:
        results = predict(dummy_data, save_dir)
        print(results)
        print(f"\nInference Results (Threshold: {results[0]['threshold']:.6f}):")
        for i, res in enumerate(results):
            status = "ANOMALY" if res['is_anomaly'] else "Normal"
            print(f"Sample {i}: Anomaly Score (MSE) = {res['score']:.6f} [{status}]")
            
    except Exception as e:
        print(f"Inference failed: {e}")
