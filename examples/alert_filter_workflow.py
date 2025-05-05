#!/usr/bin/env python3
"""
Example workflow for using the alert filter model with the autoencoder for anomaly detection.

This script demonstrates the complete workflow for:
1. Making predictions with the autoencoder
2. Collecting user feedback on alerts
3. Training the alert filter model
4. Making predictions with the alert filter model

Usage:
    python alert_filter_workflow.py --model-path <path_to_autoencoder_model> --data-path <path_to_data>
"""

import os
import argparse
from pathlib import Path

import torch
import numpy as np
from sklearn.metrics import classification_report

from privateer_ad.config import AlertFilterConfig, PathsConf, logger
from privateer_ad.models import AlertFilterModel
from privateer_ad.train_alert_filter.feedback_collector import FeedbackCollector
from privateer_ad.train_alert_filter.alert_filter_trainer import AlertFilterTrainer
from privateer_ad.predict.predict import make_predictions, make_predictions_with_filter

def main():
    parser = argparse.ArgumentParser(description='Alert filter workflow example')
    parser.add_argument('--model-path', type=str, required=True, 
                        help='Path to the autoencoder model')
    parser.add_argument('--data-path', type=str, required=True,
                        help='Path to the data (or "train", "val", "test")')
    parser.add_argument('--threshold', type=float, default=0.026970019564032555,
                        help='Threshold for anomaly detection')
    parser.add_argument('--skip-feedback', action='store_true',
                        help='Skip the feedback collection step')
    parser.add_argument('--skip-training', action='store_true',
                        help='Skip the alert filter training step')
    parser.add_argument('--filter-model-path', type=str,
                        help='Path to an existing alert filter model (skips training)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs to train the alert filter model')
    
    args = parser.parse_args()
    
    # Initialize paths
    paths = PathsConf()
    
    # Step 1: Make predictions with the autoencoder and get latent representations
    print("\n=== Step 1: Making predictions with the autoencoder ===")
    # Use make_predictions_with_filter with use_filter=False to get latent representations
    inputs, latents, losses, predictions, _, _, labels = make_predictions_with_filter(
        model_path=args.model_path,
        data_path=args.data_path,
        threshold=args.threshold,
        use_filter=False
    )
    
    # Step 2: Collect user feedback (if not skipped)
    filter_model_path = args.filter_model_path
    
    if not args.skip_feedback and not filter_model_path:
        print("\n=== Step 2: Collecting user feedback ===")
        # Initialize feedback collector
        feedback_collector = FeedbackCollector()
        
        # Get feedback statistics before collection
        stats_before = feedback_collector.get_stats()
        print(f"Feedback statistics before collection: {stats_before}")
        
        # Collect feedback on a subset of anomalies
        anomaly_indices = np.where(predictions == 1)[0]
        if len(anomaly_indices) > 0:
            # Limit to 10 anomalies for this example
            sample_size = min(10, len(anomaly_indices))
            sampled_indices = np.random.choice(anomaly_indices, size=sample_size, replace=False)
            
            for idx in sampled_indices:
                print(f"\nAlert for sample {idx}:")
                print(f"Reconstruction error: {losses[idx]:.6f}")
                print(f"True label: {'Anomaly' if labels[idx] == 1 else 'Normal'}")
                
                # In a real application, this would be a UI interaction
                # For this example, we'll use the true label as feedback
                # (In practice, user feedback might differ from true labels)
                user_feedback = labels[idx]  # 1 = true positive, 0 = false positive
                
                # Add feedback using the actual latent representation
                # Flatten the latent vector if it's multi-dimensional
                latent_vector = latents[idx]
                if isinstance(latent_vector, np.ndarray) and latent_vector.ndim > 1:
                    # Take the mean across the sequence dimension to get a vector of size latent_dim
                    latent_vector = np.mean(latent_vector, axis=0)
                
                feedback_collector.add_feedback(
                    latent=latent_vector,
                    anomaly_decision=predictions[idx],
                    reconstruction_error=losses[idx],
                    user_feedback=user_feedback
                )
                
                print(f"Added feedback: {'True positive' if user_feedback == 1 else 'False positive'}")
        else:
            print("No anomalies detected to collect feedback on")
        
        # Get feedback statistics after collection
        stats_after = feedback_collector.get_stats()
        print(f"Feedback statistics after collection: {stats_after}")
    
    # Step 3: Train the alert filter model (if not skipped)
    if not args.skip_training and not filter_model_path:
        print("\n=== Step 3: Training the alert filter model ===")
        # Initialize trainer
        trainer = AlertFilterTrainer(config=AlertFilterConfig())
        
        # Train model
        feedback_collector = FeedbackCollector()
        stats = feedback_collector.get_stats()
        
        if stats['total'] > 0:
            print(f"Training with {stats['total']} feedback samples")
            metrics = trainer.train(
                feedback_collector=feedback_collector,
                epochs=args.epochs
            )
            print(f"Training complete. Final loss: {metrics['loss']:.6f}")
            
            # Save model
            alert_filter_dir = paths.root.joinpath('alert_filter_models')
            os.makedirs(alert_filter_dir, exist_ok=True)
            filter_model_path = alert_filter_dir.joinpath('alert_filter_example.pt')
            torch.save(trainer.model.state_dict(), filter_model_path)
            print(f"Saved alert filter model to {filter_model_path}")
        else:
            print("No feedback data available for training")
    
    # Step 4: Make predictions with the alert filter model (if available)
    if filter_model_path and Path(filter_model_path).exists():
        print("\n=== Step 4: Making predictions with the alert filter model ===")
        inputs, latents, losses, predictions, anomaly_decisions, filtered_decisions, labels = make_predictions_with_filter(
            model_path=args.model_path,
            data_path=args.data_path,
            filter_model_path=filter_model_path,
            threshold=args.threshold,
            collect_feedback=False
        )
        
        # Calculate reduction in false positives
        if len(labels) > 0:
            # Ensure arrays have the same shape
            if len(filtered_decisions) != len(labels):
                logger.warning(f"Filtered decisions and labels have different shapes: {filtered_decisions.shape} vs {labels.shape}")
                # Resize filtered_decisions to match labels
                if len(filtered_decisions) > len(labels):
                    filtered_decisions = filtered_decisions[:len(labels)]
                else:
                    # This shouldn't happen, but just in case
                    logger.error("Filtered decisions array is smaller than labels array, cannot calculate reduction")
                    return
            
            # Find false positives in unfiltered predictions
            unfiltered_fp = np.logical_and(anomaly_decisions == 1, labels == 0).sum()
            
            # Find false positives in filtered predictions
            filtered_fp = np.logical_and(filtered_decisions == 1, labels == 0).sum()
            
            # Calculate reduction
            if unfiltered_fp > 0:
                reduction = (unfiltered_fp - filtered_fp) / unfiltered_fp * 100
                print(f"\nFalse positive reduction: {reduction:.2f}%")
                print(f"Unfiltered false positives: {unfiltered_fp}")
                print(f"Filtered false positives: {filtered_fp}")
            else:
                print("\nNo false positives in unfiltered predictions")
    else:
        print("\nSkipping prediction with alert filter (no model available)")
    
    print("\nWorkflow complete!")

if __name__ == '__main__':
    main()
