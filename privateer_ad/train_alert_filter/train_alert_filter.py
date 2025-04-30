import os
import torch
import mlflow
import argparse
from datetime import datetime
from pathlib import Path

from privateer_ad.config import AlertFilterConfig, PathsConf, MLFlowConfig, logger
from privateer_ad.models import AlertFilterModel
from privateer_ad.train_alert_filter.feedback_collector import FeedbackCollector
from privateer_ad.train_alert_filter.alert_filter_trainer import AlertFilterTrainer

def train_alert_filter(
    feedback_path: str = None,
    model_path: str = None,
    epochs: int = None,
    learning_rate: float = None,
    batch_size: int = None,
    hidden_dims: str = None,
    dropout: float = None,
    track_mlflow: bool = False,
    save_model: bool = True
):
    """
    Train an alert filter model based on collected feedback.
    
    Args:
        feedback_path (str, optional): Path to the feedback data directory.
            If None, uses the default path.
        model_path (str, optional): Path to a saved model state dict to continue training.
            If None, a new model is created.
        epochs (int, optional): Number of epochs to train for.
            If None, uses the value from config.
        learning_rate (float, optional): Learning rate for the optimizer.
            If None, uses the value from config.
        batch_size (int, optional): Batch size for training.
            If None, uses the value from config.
        hidden_dims (str, optional): Comma-separated list of hidden dimensions.
            If None, uses the value from config.
        dropout (float, optional): Dropout rate for regularization.
            If None, uses the value from config.
        track_mlflow (bool): Whether to track the training with MLflow.
        save_model (bool): Whether to save the trained model.
    
    Returns:
        str: Path to the saved model if save_model is True, otherwise None.
    """
    # Initialize paths
    paths = PathsConf()
        
    # Create configuration with overrides
    config = AlertFilterConfig()
    if learning_rate is not None:
        config.learning_rate = learning_rate
    if batch_size is not None:
        config.batch_size = batch_size
    if hidden_dims is not None:
        config.hidden_dims = [int(dim) for dim in hidden_dims.split(',')]
    if dropout is not None:
        config.dropout = dropout
    
    
    # Initialize feedback collector
    feedback_collector = FeedbackCollector(
        storage_path=Path(feedback_path) if feedback_path else None
    )
    
    # Get feedback statistics
    stats = feedback_collector.get_stats()
    logger.info(f"Feedback statistics: {stats}")
    
    if stats['total'] == 0:
        logger.error("No feedback data available for training")
        return None
    
   
    # Initialize trainer
    if model_path:
        # Load existing model
        trainer = AlertFilterTrainer.load_model(
            path=model_path,
            config=config
        )
        logger.info(f"Loaded existing model from {model_path}")
    else:
        # Create new model
        trainer = AlertFilterTrainer(config=config)
        logger.info("Created new alert filter model")
    
    # Train model
    metrics = trainer.train(
        feedback_collector=feedback_collector,
        epochs=epochs or config.epochs
    )
    
   
    # Save model if requested
    saved_path = None
    if save_model:
        # Create directory for alert filter models if it doesn't exist
        alert_filter_dir = paths.root.joinpath('alert_filter_models')
        os.makedirs(alert_filter_dir, exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        saved_path = alert_filter_dir.joinpath(f'alert_filter_{timestamp}.pt')
        
        # Save model
        torch.save(trainer.model.state_dict(), saved_path)
        logger.info(f"Saved alert filter model to {saved_path}")
        
    
    
    return str(saved_path) if saved_path else None

def main():
    """Command line interface for training the alert filter model."""
    parser = argparse.ArgumentParser(description='Train an alert filter model based on collected feedback.')
    parser.add_argument('--feedback-path', type=str, help='Path to the feedback data directory')
    parser.add_argument('--model-path', type=str, help='Path to a saved model state dict to continue training')
    parser.add_argument('--epochs', type=int, help='Number of epochs to train for')
    parser.add_argument('--learning-rate', type=float, help='Learning rate for the optimizer')
    parser.add_argument('--batch-size', type=int, help='Batch size for training')
    parser.add_argument('--hidden-dims', type=str, help='Comma-separated list of hidden dimensions')
    parser.add_argument('--dropout', type=float, help='Dropout rate for regularization')
    parser.add_argument('--no-mlflow', action='store_true', help='Disable MLflow tracking')
    parser.add_argument('--no-save', action='store_true', help='Do not save the trained model')
    
    args = parser.parse_args()
    
    saved_path = train_alert_filter(
        feedback_path=args.feedback_path,
        model_path=args.model_path,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        hidden_dims=args.hidden_dims,
        dropout=args.dropout,
        track_mlflow=not args.no_mlflow,
        save_model=not args.no_save
    )
    
    if saved_path:
        print(f"Model saved to: {saved_path}")

if __name__ == '__main__':
    main()
