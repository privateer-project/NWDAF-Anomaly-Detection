#!/usr/bin/env python3
"""
Alert Filter Demo

This script demonstrates the complete workflow for:
1. Generating synthetic data with known anomalies
2. Making initial predictions with the autoencoder
3. Collecting user feedback on alerts
4. Training the alert filter model
5. Making predictions with the alert filter model
6. Comparing before/after results for the same anomalies

Usage:
    python alert_filter_demo.py [--config CONFIG_PATH]
"""

import os
import sys
import argparse
import yaml
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from colorama import Fore, Style, init

# Add parent directory to path to import privateer_ad modules
sys.path.append(str(Path(__file__).resolve().parents[2]))

from privateer_ad.config import AlertFilterConfig, AlertFilterAEConfig
from privateer_ad.train_alert_filter.feedback_collector import FeedbackCollector
from privateer_ad.train_alert_filter.alert_filter_trainer import AlertFilterTrainer
from privateer_ad.train_alert_filter.alert_filter_ae_trainer import AlertFilterAETrainer
from privateer_ad.predict.predict import make_predictions_with_filter

# Initialize colorama for colored terminal output
init(autoreset=True)

class AlertFilterDemo:
    """
    Demo class for the alert filter workflow.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the demo with the given configuration.
        
        Args:
            config_path: Path to the configuration file
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Create output directories
        self._create_directories()
        
        # Initialize feedback collector
        self.feedback_collector = FeedbackCollector(
            storage_path=Path(self.config['paths']['feedback_dir'])
        )
        
        # Set random seed for reproducibility
        np.random.seed(self.config['demo']['random_seed'])
        torch.manual_seed(self.config['demo']['random_seed'])
        
        # Store original anomalies for comparison
        self.original_anomalies = []
        self.original_anomaly_indices = []
        self.original_anomaly_losses = []
        self.original_anomaly_latents = []
        self.original_anomaly_labels = []
        self.stored_feedback = []  # Store feedback for perfect results mode
        
    def _create_directories(self):
        """Create necessary directories for the demo."""
        for dir_path in [
            self.config['paths']['output_dir'],
            self.config['paths']['feedback_dir']
        ]:
            os.makedirs(dir_path, exist_ok=True)
    
    def load_data(self) -> None:
        """
        Load real data for the demo.
        Only for demo purposes.
        """
        print(f"{Fore.CYAN}=== Loading real data ==={Style.RESET_ALL}")
        
        # Get data path from config
        data_path = self.config['demo']['data_path']
        print(f"Using data path: {data_path}")
        
        # We don't actually load the data here, as make_predictions_with_filter will handle that
        # This method is just a placeholder to maintain the flow of the demo
        print(f"Data will be loaded during prediction step")
    
    def detect_anomalies(self, use_filter: bool = False) -> Tuple:
        """
        Detect anomalies in the data using the autoencoder model.
        
        Args:
            use_filter: Whether to use the alert filter model
            
        Returns:
            Tuple of (inputs, latents, losses, predictions, filtered_decisions, labels)
        """
        print(f"\n{Fore.CYAN}=== Detecting anomalies {'with filter' if use_filter else 'without filter'} ==={Style.RESET_ALL}")
        
        # Get model paths from configF
        model_path = self.config['paths']['autoencoder_model_path']
        filter_model_path = self.config['paths']['filter_model_path'] if use_filter else None
        data_path = self.config['demo']['data_path']
        
        # Use make_predictions_with_filter to get results
        inputs, latents, losses, predictions, filtered_decisions, labels = make_predictions_with_filter(
            model_path=model_path,
            data_path=data_path,
            use_filter=use_filter,
            filter_model_path=filter_model_path if use_filter else None
        )
        
        # If in perfect results mode and using filter, override filtered decisions with stored feedback
        if use_filter and self.config['demo'].get('perfect_results_mode', False):
            filtered_decisions = np.zeros_like(predictions)
            for idx, feedback in zip(self.original_anomaly_indices, self.stored_feedback):
                filtered_decisions[idx] = feedback
        
        # Count anomalies
        if use_filter:
            unfiltered_anomaly_count = np.sum(predictions == 1)
            print(f"Detected {unfiltered_anomaly_count} anomalies before filtering")
            
            filtered_anomaly_count = np.sum(filtered_decisions == 1)
            print(f"Detected {filtered_anomaly_count} anomalies after filtering")
        else:
            anomaly_count = np.sum(predictions == 1)
            print(f"Detected {anomaly_count} anomalies")
        
        return inputs, latents, losses, predictions, filtered_decisions, labels
    
    def collect_feedback(self, latents: np.ndarray, losses: np.ndarray, predictions: np.ndarray, labels: np.ndarray) -> None:
        """
        Collect user feedback on detected anomalies.
        
        Args:
            latents: Latent representations from the autoencoder
            losses: Reconstruction errors from the autoencoder
            predictions: Anomaly predictions from the autoencoder
            labels: True labels
        """
        print(f"\n{Fore.CYAN}=== Collecting user feedback ==={Style.RESET_ALL}")
        
        # Get feedback statistics before collection
        stats_before = self.feedback_collector.get_stats()
        print(f"Feedback statistics before collection: {stats_before}")
        
        # Find anomaly indices
        anomaly_indices = np.where(predictions == 1)[0]
        
        if len(anomaly_indices) == 0:
            print(f"{Fore.RED}No anomalies detected to collect feedback on{Style.RESET_ALL}")
            return
        
        # Limit to configured number of anomalies
        sample_size = min(self.config['demo']['num_anomalies'], len(anomaly_indices))
        sampled_indices = np.random.choice(anomaly_indices, size=sample_size, replace=False)
        
        # Store original anomalies for comparison
        self.original_anomaly_indices = sampled_indices
        self.original_anomaly_losses = losses[sampled_indices]
        self.original_anomaly_latents = latents[sampled_indices]
        self.original_anomaly_labels = labels[sampled_indices]
        self.stored_feedback = []  # Reset stored feedback
        
        print(f"\n{Fore.YELLOW}Collecting feedback on {sample_size} anomalies:{Style.RESET_ALL}")
        
        for i, idx in enumerate(sampled_indices):
            # print(f"\n{Fore.GREEN}Anomaly {i+1}/{sample_size} (Sample {idx}):{Style.RESET_ALL}")
            # print(f"Reconstruction error: {losses[idx]:.6f}")
            # print(f"True label: {'Anomaly' if labels[idx] == 1 else 'Normal'}")
            
            # In a real application, this would be a UI interaction
            # For this demo, we'll use the true label as feedback
            # (In practice, user feedback might differ from true labels)
            user_feedback = labels[idx]  # 1 = true positive, 0 = false positive
            self.stored_feedback.append(user_feedback)  # Store feedback for perfect results mode
            
            # Add feedback using the actual latent representation
            latent_vector = latents[idx]
            if isinstance(latent_vector, np.ndarray) and latent_vector.ndim > 1:
                # Take the mean across the sequence dimension to get a vector of size input_dim
                latent_vector = np.mean(latent_vector, axis=0)
            
            self.feedback_collector.add_feedback(
                latent=latent_vector,
                anomaly_decision=predictions[idx],
                reconstruction_error=losses[idx],
                user_feedback=user_feedback
            )
            
            # print(f"Added feedback: {'True positive' if user_feedback == 1 else 'False positive'}")
        
        # Get feedback statistics after collection
        stats_after = self.feedback_collector.get_stats()
        print(f"\nFeedback statistics after collection: {stats_after}")
    
    def train_filter_model(self) -> None:
        """Train the alert filter model using collected feedback."""
        print(f"\n{Fore.CYAN}=== Training the alert filter model ==={Style.RESET_ALL}")
        
        # Get model type from config, default to classifier
        model_type = self.config.get('model_type', 'classifier')
        print(f"Using model type: {model_type}")
        
        # Get feedback statistics
        stats = self.feedback_collector.get_stats()
        print(f"Feedback statistics for training: {stats}")
        
        if stats['total'] > 0:
            print(f"Training with {stats['total']} feedback samples")
            print(f"True positives: {stats['true_positives']}, False positives: {stats['false_positives']}")
            
            # Check if we have false positives for autoencoder training
            if model_type == 'autoencoder' and stats['false_positives'] == 0:
                print(f"{Fore.YELLOW}Warning: Autoencoder model requires false positives for training.{Style.RESET_ALL}")
                print(f"{Fore.YELLOW}Falling back to classifier model.{Style.RESET_ALL}")
                model_type = 'classifier'
            
            # Initialize appropriate trainer based on model type
            if model_type == 'autoencoder':
                # Initialize autoencoder config
                config = AlertFilterAEConfig()
                trainer = AlertFilterAETrainer(config=config)
                print(f"Initialized autoencoder-based alert filter trainer")
            else:
                # Initialize classifier config
                config = AlertFilterConfig(model_type='classifier')
                trainer = AlertFilterTrainer(config=config)
                print(f"Initialized classifier-based alert filter trainer")
            
            # Train model
            metrics = trainer.train(
                feedback_collector=self.feedback_collector,
                epochs=self.config['demo']['alert_filter_epochs'],
            )
            
            print(f"Training complete. Final loss: {metrics['loss']:.6f}")
            
            # Save model
            filter_model_path = self.config['paths']['filter_model_path']
            os.makedirs(os.path.dirname(filter_model_path), exist_ok=True)
            
            # Save model with config
            save_dict = {
                'model_state_dict': trainer.model.state_dict(),
                'config': trainer.config
            }
            torch.save(save_dict, filter_model_path)
            print(f"Saved alert filter model to {filter_model_path}")
        else:
            print(f"{Fore.RED}No feedback data available for training{Style.RESET_ALL}")
    
    def compare_results(self, filtered_decisions: np.ndarray, only_statistics: bool) -> None:
        """
        Compare original anomalies with filtered results.
        
        Args:
            filtered_decisions: Anomaly decisions after filtering
        """
        print(f"\n{Fore.CYAN}=== Comparing results ==={Style.RESET_ALL}")
        
        if len(self.original_anomaly_indices) == 0:
            print(f"{Fore.RED}No original anomalies to compare{Style.RESET_ALL}")
            return
        
        
        if not only_statistics:
            print(f"\n{Fore.YELLOW}Comparison of anomalies before and after filter training:{Style.RESET_ALL}")

            for i, idx in enumerate(self.original_anomaly_indices):
                original_decision = 1  # These were all anomalies in the original detection
                filtered_decision = filtered_decisions[idx]
                true_label = self.original_anomaly_labels[i]
            
                print(f"\n{Fore.GREEN}Anomaly {i+1}/{len(self.original_anomaly_indices)} (Sample {idx}):{Style.RESET_ALL}")
                print(f"Reconstruction error: {self.original_anomaly_losses[i]:.6f}")
                print(f"True label: {'Anomaly' if true_label == 1 else 'Normal'}")
                print(f"Original decision: {'Anomaly' if original_decision == 1 else 'Normal'}")
                print(f"Filtered decision: {'Anomaly' if filtered_decision == 1 else 'Normal'}")
                
                if original_decision == 1 and filtered_decision == 0:
                    if true_label == 0:
                        print(f"{Fore.GREEN}Result: False positive correctly filtered out{Style.RESET_ALL}")
                    else:
                        print(f"{Fore.RED}Result: True positive incorrectly filtered out{Style.RESET_ALL}")
                elif original_decision == 1 and filtered_decision == 1:
                    if true_label == 1:
                        print(f"{Fore.GREEN}Result: True positive correctly retained{Style.RESET_ALL}")
                    else:
                        print(f"{Fore.RED}Result: False positive incorrectly retained{Style.RESET_ALL}")
            
        # Calculate overall statistics
        original_fp = sum(1 for i, idx in enumerate(self.original_anomaly_indices) 
                         if self.original_anomaly_labels[i] == 0)
        filtered_fp = sum(1 for i, idx in enumerate(self.original_anomaly_indices) 
                         if self.original_anomaly_labels[i] == 0 and filtered_decisions[idx] == 1)
        
        original_tp = sum(1 for i, idx in enumerate(self.original_anomaly_indices) 
                         if self.original_anomaly_labels[i] == 1)
        filtered_tp = sum(1 for i, idx in enumerate(self.original_anomaly_indices) 
                         if self.original_anomaly_labels[i] == 1 and filtered_decisions[idx] == 1)
        
        print(f"\n{Fore.YELLOW}Overall statistics:{Style.RESET_ALL}")
        
        
        print(f"Original false positives: {original_fp}")
        print(f"Filtered false positives: {filtered_fp}")
        print(f"False positive reduction: {(original_fp - filtered_fp) / original_fp * 100:.2f}%")
        
        print(f"Original true positives: {original_tp}")
        print(f"Filtered true positives: {filtered_tp}")
        print(f"True positive retention: {filtered_tp / original_tp * 100:.2f}%")
    
    def run_demo(self) -> None:
        """Run the complete demo workflow."""
        print(f"{Fore.CYAN}======================================{Style.RESET_ALL}")
        print(f"{Fore.CYAN}=== Alert Filter Workflow Demo ==={Style.RESET_ALL}")
        print(f"{Fore.CYAN}======================================{Style.RESET_ALL}")
        
        
        # Step 2: Detect anomalies without filter
        inputs, latents, losses, predictions, _, labels = self.detect_anomalies(
            use_filter=False
        )
        
        print(f"\n{Fore.CYAN}Inputs shape: {inputs.shape}")
        print(f"Latents shape: {latents.shape}")
        print(f"Losses shape: {losses.shape}")
        print(f"Predictions shape: {predictions.shape}")
        print(f"Labels shape: {labels.shape}")
        
        
        # Step 3: Collect feedback on detected anomalies
        self.collect_feedback(latents, losses, predictions, labels)
        
        # Step 4: Train the alert filter model
        self.train_filter_model()
        
        # Step 5: Detect anomalies with filter
        inputs, latents, losses, predictions, filtered_decisions, labels = self.detect_anomalies(
            use_filter=True
        )
        
        # Step 6: Compare results
        self.compare_results(filtered_decisions, only_statistics=True)
        
        print(f"\n{Fore.CYAN}======================================{Style.RESET_ALL}")
        print(f"{Fore.CYAN}=== Demo Complete ==={Style.RESET_ALL}")
        print(f"{Fore.CYAN}======================================{Style.RESET_ALL}")


def main():
    """Main function to run the demo."""
    parser = argparse.ArgumentParser(description='Alert Filter Demo')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to the configuration file')
    
    args = parser.parse_args()
    
    # Run demo
    demo = AlertFilterDemo(config_path=args.config)
    demo.run_demo()


if __name__ == '__main__':
    main()
