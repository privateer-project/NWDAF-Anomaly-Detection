#!/usr/bin/env python3
"""
Setup script for the Alert Filter Demo

This script prepares the demo environment by:
1. Creating necessary directories
2. Setting up the initial configuration

Usage:
    python setup_demo.py
"""

import os
import sys
import yaml
import shutil
from pathlib import Path

# Add parent directory to path to import privateer_ad modules
sys.path.append(str(Path(__file__).resolve().parents[2]))

def setup_demo():
    """Set up the demo environment."""
    print("Setting up Alert Filter Demo environment...")
    
    # Get the current directory
    current_dir = Path(__file__).resolve().parent
    
    # Load configuration
    config_path = current_dir / "config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create directories
    print("Creating directories...")
    directories = [
        config['paths']['output_dir'],
        config['paths']['feedback_dir'],
        os.path.dirname(config['model']['alert_filter']['model_path'])
    ]
    
    for directory in directories:
        # Handle both absolute and relative paths
        if os.path.isabs(directory):
            dir_path = Path(directory)
        else:
            dir_path = current_dir / directory
        
        os.makedirs(dir_path, exist_ok=True)
        print(f"  Created directory: {dir_path}")
    
    # Check if autoencoder model exists
    model_path = config['model']['autoencoder']['model_path']
    if os.path.isabs(model_path):
        full_model_path = model_path
    else:
        full_model_path = current_dir / model_path
    
    if not os.path.exists(full_model_path):
        print(f"\n{full_model_path} does not exist.")
        print("Please ensure you have a trained autoencoder model at this location.")
        print("You may need to adjust the model path in config.yaml.")
    else:
        print(f"\nFound autoencoder model at: {full_model_path}")
    
    print("\nDemo setup complete!")
    print("\nTo run the demo, use:")
    print(f"  python alert_filter_demo.py")
    print("\nYou can customize the demo by editing config.yaml")


if __name__ == "__main__":
    setup_demo()
