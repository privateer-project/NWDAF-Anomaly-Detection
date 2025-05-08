#!/usr/bin/env python3
"""
Interactive Alert Filter Demo

This script provides an interactive version of the alert filter workflow demo,
allowing users to provide their own feedback on detected anomalies.

Usage:
    python interactive_demo.py [--config CONFIG_PATH]
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

from privateer_ad.config import AlertFilterConfig, PathsConf, logger
from privateer_ad.models import AlertFilterModel
from privateer_ad.train_alert_filter.feedback_collector import FeedbackCollector
from privateer_ad.train_alert_filter.alert_filter_trainer import AlertFilterTrainer
from privateer_ad.predict.predict import make_predictions, make_predictions_with_filter

# Initialize colorama for colored terminal output
init(autoreset=True)

class InteractiveAlertFilterDemo:
    """
    Interactive demo class for the alert filter workflow.
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
        self.original_anomaly_indices = []
        self.original_anomaly_losses = []
        self.original_anomaly_latents = []
        self.original_anomaly_labels = []
        self.user_feedback_list = []
        
    def _create_directories(self):
        """Create necessary directories for the demo."""
        for dir_path in [
            self.config['paths']['output_dir'],
            self.config['paths']['feedback_dir'],
            os.path.dirname(self.config['model']['autoencoder']['model_path']),
            os.path.dirname(self.config['model']['alert_filter']['model_path'])
        ]:
            # Handle both absolute and relative paths
            if os.path.isabs(dir_path):
                dir_path = Path(dir_path)
            else:
                dir_path = Path(__file__).resolve().parent / dir_path
            
            os.makedirs(dir_path, exist_ok=True)
    
    def load_data(self) -> None:
        """
        Load real data for the demo.
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
            Tuple of (inputs, latents, losses, predictions, anomaly_decisions, filtered_decisions, labels)
        """
        print(f"\n{Fore.CYAN}=== Detecting anomalies {'with filter' if use_filter else 'without filter'} ==={Style.RESET_ALL}")
        
        # Get model paths from config
        model_path = self.config['model']['autoencoder']['model_path']
        threshold = self.config['model']['autoencoder']['threshold']
        filter_model_path = self.config['model']['alert_filter']['model_path'] if use_filter else None
        data_path = self.config['demo']['data_path']
        
        # Use make_predictions_with_filter to get results
        inputs, latents, losses, predictions, anomaly_decisions, filtered_decisions, labels = make_predictions_with_filter(
            model_path=model_path,
            data_path=data_path,
            threshold=threshold,
            use_filter=use_filter,
            filter_model_path=filter_model_path if use_filter else None
        )
        
        # Count anomalies
        if use_filter:
            anomaly_count = np.sum(filtered_decisions == 1)
            print(f"Detected {anomaly_count} anomalies after filtering")
        else:
            anomaly_count = np.sum(predictions == 1)
            print(f"Detected {anomaly_count} anomalies")
        
        return inputs, latents, losses, predictions, anomaly_decisions, filtered_decisions, labels
    
    def collect_interactive_feedback(self, latents: np.ndarray, losses: np.ndarray, predictions: np.ndarray, labels: np.ndarray) -> None:
        """
        Collect interactive user feedback on detected anomalies.
        
        Args:
            latents: Latent representations from the autoencoder
            losses: Reconstruction errors from the autoencoder
            predictions: Anomaly predictions from the autoencoder
            labels: True labels (only used for display, not for feedback)
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
        self.user_feedback_list = []
        
        print(f"\n{Fore.YELLOW}Collecting feedback on {sample_size} anomalies:{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}For each anomaly, indicate if it's a true positive (1) or false positive (0){Style.RESET_ALL}")
        
        for i, idx in enumerate(sampled_indices):
            print(f"\n{Fore.GREEN}Anomaly {i+1}/{sample_size} (Sample {idx}):{Style.RESET_ALL}")
            print(f"Reconstruction error: {losses[idx]:.6f}")
            print(f"True label (for reference): {'Anomaly' if labels[idx] == 1 else 'Normal'}")
            
            # Get user feedback
            while True:
                try:
                    user_input = input(f"{Fore.YELLOW}Is this a true anomaly? (1=yes, 0=no): {Style.RESET_ALL}")
                    user_feedback = int(user_input.strip())
                    if user_feedback not in [0, 1]:
                        print(f"{Fore.RED}Invalid input. Please enter 0 or 1.{Style.RESET_ALL}")
                        continue
                    break
                except ValueError:
                    print(f"{Fore.RED}Invalid input. Please enter 0 or 1.{Style.RESET_ALL}")
            
            # Store user feedback for later comparison
            self.user_feedback_list.append(user_feedback)
            
            # Add feedback using the actual latent representation
            latent_vector = latents[idx]
            if isinstance(latent_vector, np.ndarray) and latent_vector.ndim > 1:
                # Take the mean across the sequence dimension to get a vector of size latent_dim
                latent_vector = np.mean(latent_vector, axis=0)
            
            self.feedback_collector.add_feedback(
                latent=latent_vector,
                anomaly_decision=predictions[idx],
                reconstruction_error=losses[idx],
                user_feedback=user_feedback
            )
            
            print(f"Added feedback: {'True positive' if user_feedback == 1 else 'False positive'}")
        
        # Get feedback statistics after collection
        stats_after = self.feedback_collector.get_stats()
        print(f"\nFeedback statistics after collection: {stats_after}")
    
    def train_filter_model(self) -> None:
        """Train the alert filter model using collected feedback."""
        print(f"\n{Fore.CYAN}=== Training the alert filter model ==={Style.RESET_ALL}")
        
        # Initialize alert filter config
        alert_filter_config = AlertFilterConfig()
        alert_filter_config.latent_dim = self.config['model']['alert_filter']['latent_dim']
        alert_filter_config.hidden_dim = self.config['model']['alert_filter']['hidden_dim']
        alert_filter_config.learning_rate = self.config['model']['alert_filter']['learning_rate']
        
        # Initialize trainer
        trainer = AlertFilterTrainer(config=alert_filter_config)
        
        # Get feedback statistics
        stats = self.feedback_collector.get_stats()
        
        if stats['total'] > 0:
            print(f"Training with {stats['total']} feedback samples")
            print(f"True positives: {stats['true_positives']}, False positives: {stats['false_positives']}")
            
            # Train model
            metrics = trainer.train(
                feedback_collector=self.feedback_collector,
                epochs=self.config['model']['alert_filter']['epochs']
            )
            
            print(f"Training complete. Final loss: {metrics['loss']:.6f}")
            
            # Save model
            filter_model_path = self.config['model']['alert_filter']['model_path']
            os.makedirs(os.path.dirname(filter_model_path), exist_ok=True)
            torch.save(trainer.model.state_dict(), filter_model_path)
            print(f"Saved alert filter model to {filter_model_path}")
        else:
            print(f"{Fore.RED}No feedback data available for training{Style.RESET_ALL}")
    
    def compare_results(self, filtered_decisions: np.ndarray) -> None:
        """
        Compare original anomalies with filtered results.
        
        Args:
            filtered_decisions: Anomaly decisions after filtering
        """
        print(f"\n{Fore.CYAN}=== Comparing results ==={Style.RESET_ALL}")
        
        if len(self.original_anomaly_indices) == 0:
            print(f"{Fore.RED}No original anomalies to compare{Style.RESET_ALL}")
            return
        
        print(f"\n{Fore.YELLOW}Comparison of anomalies before and after filter training:{Style.RESET_ALL}")
        
        for i, idx in enumerate(self.original_anomaly_indices):
            original_decision = 1  # These were all anomalies in the original detection
            filtered_decision = filtered_decisions[idx]
            true_label = self.original_anomaly_labels[i]
            user_feedback = self.user_feedback_list[i]
            
            print(f"\n{Fore.GREEN}Anomaly {i+1}/{len(self.original_anomaly_indices)} (Sample {idx}):{Style.RESET_ALL}")
            print(f"Reconstruction error: {self.original_anomaly_losses[i]:.6f}")
            print(f"True label: {'Anomaly' if true_label == 1 else 'Normal'}")
            print(f"Your feedback: {'True positive' if user_feedback == 1 else 'False positive'}")
            print(f"Original decision: {'Anomaly' if original_decision == 1 else 'Normal'}")
            print(f"Filtered decision: {'Anomaly' if filtered_decision == 1 else 'Normal'}")
            
            if original_decision == 1 and filtered_decision == 0:
                if user_feedback == 0:
                    print(f"{Fore.GREEN}Result: False positive correctly filtered out based on your feedback{Style.RESET_ALL}")
                else:
                    print(f"{Fore.RED}Result: True positive incorrectly filtered out despite your feedback{Style.RESET_ALL}")
            elif original_decision == 1 and filtered_decision == 1:
                if user_feedback == 1:
                    print(f"{Fore.GREEN}Result: True positive correctly retained based on your feedback{Style.RESET_ALL}")
                else:
                    print(f"{Fore.RED}Result: False positive incorrectly retained despite your feedback{Style.RESET_ALL}")
        
        # Calculate overall statistics based on user feedback
        original_fp = sum(1 for i in range(len(self.user_feedback_list)) if self.user_feedback_list[i] == 0)
        filtered_fp = sum(1 for i, idx in enumerate(self.original_anomaly_indices) 
                         if self.user_feedback_list[i] == 0 and filtered_decisions[idx] == 1)
        
        original_tp = sum(1 for i in range(len(self.user_feedback_list)) if self.user_feedback_list[i] == 1)
        filtered_tp = sum(1 for i, idx in enumerate(self.original_anomaly_indices) 
                         if self.user_feedback_list[i] == 1 and filtered_decisions[idx] == 1)
        
        print(f"\n{Fore.YELLOW}Overall statistics based on your feedback:{Style.RESET_ALL}")
        print(f"Original false positives (based on your feedback): {original_fp}")
        print(f"Filtered false positives: {filtered_fp}")
        if original_fp > 0:
            print(f"False positive reduction: {(original_fp - filtered_fp) / original_fp * 100:.2f}%")
        
        print(f"Original true positives (based on your feedback): {original_tp}")
        print(f"Filtered true positives: {filtered_tp}")
        if original_tp > 0:
            print(f"True positive retention: {filtered_tp / original_tp * 100:.2f}%")
    
    def run_demo(self) -> None:
        """Run the complete interactive demo workflow."""
        print(f"{Fore.CYAN}======================================{Style.RESET_ALL}")
        print(f"{Fore.CYAN}=== Interactive Alert Filter Demo ==={Style.RESET_ALL}")
        print(f"{Fore.CYAN}======================================{Style.RESET_ALL}")
        
        # Step 1: Load real data
        self.load_data()
        
        # Step 2: Detect anomalies without filter
        inputs, latents, losses, predictions, anomaly_decisions, _, labels = self.detect_anomalies(
            use_filter=False
        )
        
        # Step 3: Collect interactive feedback on detected anomalies
        self.collect_interactive_feedback(latents, losses, predictions, labels)
        
        # Step 4: Train the alert filter model
        self.train_filter_model()
        
        # Step 5: Detect anomalies with filter
        _, _, _, _, _, filtered_decisions, _ = self.detect_anomalies(
            use_filter=True
        )
        
        # Step 6: Compare results
        self.compare_results(filtered_decisions)
        
        print(f"\n{Fore.CYAN}======================================{Style.RESET_ALL}")
        print(f"{Fore.CYAN}=== Demo Complete ==={Style.RESET_ALL}")
        print(f"{Fore.CYAN}======================================{Style.RESET_ALL}")


def main():
    """Main function to run the interactive demo."""
    parser = argparse.ArgumentParser(description='Interactive Alert Filter Demo')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to the configuration file')
    
    args = parser.parse_args()
    
    # Run demo
    demo = InteractiveAlertFilterDemo(config_path=args.config)
    demo.run_demo()


if __name__ == '__main__':
    main()
