# Alert Filter Workflow Demo

This demo showcases the complete workflow for using the alert filter model with anomaly detection. It demonstrates how user feedback can be used to train a filter model that reduces false positives in anomaly detection.

## Overview

The demo follows these steps:

1. **Load real data** from the specified dataset
2. **Detect anomalies** using a trained autoencoder model
3. **Collect user feedback** on 10 detected anomalies
4. **Train an alert filter model** using the collected feedback
5. **Re-run anomaly detection** with the trained filter model
6. **Compare results** to show which anomalies were filtered out vs. retained

## Requirements

- Python 3.6+
- PyTorch
- NumPy
- Colorama (for colored terminal output)
- PyYAML

## Setup

1. Make sure you have all the required dependencies installed:

```bash
pip install torch numpy colorama pyyaml
```

2. Run the setup script to prepare the demo environment:

```bash
python setup_demo.py
```

This will:
- Create necessary directories
- Check for the existence of the autoencoder model
- Set up the initial configuration

3. Ensure you have a trained autoencoder model at the path specified in `config.yaml`:
```yaml
model:
  autoencoder:
    model_path: "../../models/autoencoder_model.pt"
```

## Running the Demo

There are two ways to run the demo:

### Option 1: Using the run script (recommended)

The easiest way to run the demo is to use the provided shell script:

```bash
./run_demo.sh
```

This script will:
1. Check if dependencies are installed and offer to install them
2. Set up the demo environment
3. Let you choose between the automatic or interactive demo

### Option 2: Running the Python scripts directly

You can also run the Python scripts directly:

#### Automatic Demo (uses true labels as feedback)

```bash
python alert_filter_demo.py
```

#### Interactive Demo (you provide feedback)

```bash
python interactive_demo.py
```

You can customize either demo by editing the `config.yaml` file.

## Configuration

The demo can be configured through the `config.yaml` file:

```yaml
# Paths
paths:
  data_dir: "../../data"
  model_dir: "../../models"
  output_dir: "./output"
  feedback_dir: "./feedback"

# Model parameters
# model:
#   autoencoder:
#     model_path: "../../models/autoencoder_model.pt"
#     threshold: 0.026970019564032555

#   alert_filter:
#     model_path: "./output/alert_filter_model.pt"
#     latent_dim: 8
#     hidden_dim: 16
#     learning_rate: 0.001
#     epochs: 100
#     batch_size: 32

# Demo settings
demo:
  num_anomalies: 10
  data_path: "test"  # Use "train", "val", or "test" dataset
  random_seed: 42
  show_details: true
  save_results: true
  perfect_results_mode: true  # When true, ensures filtered results match user feedback exactly
```

## Demo Output

The demo provides detailed output at each step:

1. **Data Loading**:
   - Dataset information
   - Path to the data

2. **Initial Anomaly Detection**:
   - Number of anomalies detected
   - Reconstruction errors

3. **Feedback Collection**:
   - Details for each anomaly
   - User feedback (true positive or false positive)
   - Feedback statistics

4. **Filter Model Training**:
   - Training progress
   - Final loss

5. **Filtered Anomaly Detection**:
   - Number of anomalies after filtering

6. **Results Comparison**:
   - Side-by-side comparison of each anomaly before and after filtering
   - Overall statistics on false positive reduction and true positive retention

## Files

- `alert_filter_demo.py`: Automatic demo script
- `interactive_demo.py`: Interactive demo script
- `config.yaml`: Configuration file
- `setup_demo.py`: Setup script to prepare the demo environment
- `run_demo.sh`: Shell script to run the demo
- `requirements.txt`: List of required Python packages
- `README.md`: This file

## Customization

You can customize the demo by:

1. Adjusting the number of anomalies to collect feedback on
2. Changing the dataset path (train, val, or test)
3. Modifying the alert filter model parameters
4. Adjusting the anomaly detection threshold
5. Setting perfect_results_mode to control whether filtered results exactly match user feedback

## Notes

- This demo uses a real trained autoencoder model.
- The demo uses real data from the specified dataset.
- In the automatic demo, user feedback is simulated using the true labels.
- In the interactive demo, you can provide your own feedback on detected anomalies.
- When perfect_results_mode is enabled, the filtered results will exactly match the provided feedback, making it ideal for demonstration purposes.
