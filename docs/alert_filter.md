# Alert Filter Model

The Alert Filter Model is designed to reduce false positives in anomaly detection by learning from user feedback. It works alongside the autoencoder-based anomaly detection system to filter out alerts that are likely to be false positives.

## Overview

In anomaly detection systems, false positives can be a significant problem. The Alert Filter Model addresses this by:

1. Starting with a "free pass" approach (allowing all alerts)
2. Learning from user feedback which alerts are true positives and which are false positives
3. Gradually improving its filtering capabilities over time

## Architecture

The Alert Filter Model takes three inputs:
- The latent representation from the autoencoder
- The anomaly decision (1 = anomaly, 0 = normal)
- The reconstruction error

It outputs a decision on whether to allow or deny the alert (1 = allow, 0 = deny).

The model consists of:
- A neural network with configurable hidden layers
- ReLU activations and dropout for regularization
- A sigmoid output layer that produces a probability of allowing the alert

## Workflow

The typical workflow for using the Alert Filter Model is:

1. **Anomaly Detection**: The autoencoder model detects anomalies based on reconstruction error
2. **Alert Filtering**: The Alert Filter Model decides whether to allow or deny each alert
3. **User Feedback**: Users provide feedback on alerts (true positive or false positive)
4. **Model Training**: The Alert Filter Model is trained based on the collected feedback
5. **Improved Filtering**: The updated model provides better filtering of false positives

## Usage

### Making Predictions with the Alert Filter

```python
from privateer_ad.predict.predict import make_predictions_with_filter

# Make predictions with the alert filter
inputs, latents, losses, anomaly_decisions, filtered_decisions = make_predictions_with_filter(
    model_path='path/to/autoencoder/model.pt',
    data_path='test',  # or path to a CSV file
    filter_model_path='path/to/alert_filter/model.pt',
    threshold=0.027,  # Threshold for anomaly detection
    collect_feedback=False  # Set to True to collect feedback during prediction
)
```

### Collecting Feedback

```python
from privateer_ad.train_alert_filter.feedback_collector import FeedbackCollector

# Initialize feedback collector
feedback_collector = FeedbackCollector()

# Add feedback for an alert
feedback_collector.add_feedback(
    latent=latent_representation,  # Latent representation from the autoencoder
    anomaly_decision=1,  # 1 = anomaly, 0 = normal
    reconstruction_error=0.05,  # Reconstruction error from the autoencoder
    user_feedback=0  # 1 = true positive, 0 = false positive
)

# Get statistics about collected feedback
stats = feedback_collector.get_stats()
print(f"Total feedback: {stats['total']}")
print(f"True positives: {stats['true_positives']}")
print(f"False positives: {stats['false_positives']}")
```

### Training the Alert Filter Model

```python
from privateer_ad.train_alert_filter.alert_filter_trainer import AlertFilterTrainer
from privateer_ad.config import AlertFilterConfig

# Initialize trainer with default configuration
trainer = AlertFilterTrainer(config=AlertFilterConfig())

# Train model with collected feedback
metrics = trainer.train(
    feedback_collector=feedback_collector,
    epochs=100
)

# Save the trained model
trainer.save_model('path/to/save/model.pt')
```

### Command-Line Interface

The Alert Filter Model can also be trained using the command-line interface:

```bash
# Train the alert filter model
python -m privateer_ad.train.train_alert_filter --epochs 100 --learning-rate 0.001

# Make predictions with the alert filter
python -m privateer_ad.predict.predict predict_with_filter \
    --model-path path/to/autoencoder/model.pt \
    --data-path test \
    --filter-model-path path/to/alert_filter/model.pt \
    --threshold 0.027 \
    --collect-feedback
```

### Complete Workflow Example

An example script demonstrating the complete workflow is provided in `examples/alert_filter_workflow.py`:

```bash
python examples/alert_filter_workflow.py \
    --model-path path/to/autoencoder/model.pt \
    --data-path test
```

## Configuration

The Alert Filter Model can be configured using the `AlertFilterConfig` class:

```python
from privateer_ad.config import AlertFilterConfig

# Create custom configuration
config = AlertFilterConfig(
    latent_dim=16,  # Should match the autoencoder's latent dimension
    hidden_dims=[32, 16],  # Hidden layer dimensions
    dropout=0.2,  # Dropout rate
    learning_rate=0.001,  # Learning rate for training
    batch_size=32,  # Batch size for training
    epochs=100  # Number of epochs for training
)
```

## Performance Metrics

To evaluate the performance of the Alert Filter Model, you can use the following metrics:

- **False Positive Reduction**: Percentage of false positives that are filtered out
- **True Positive Retention**: Percentage of true positives that are allowed through
- **Overall Accuracy**: Accuracy of the filtered decisions compared to true labels

These metrics can be calculated by comparing the filtered decisions with the true labels.

## Integration with MLflow

The Alert Filter Model training process can be tracked using MLflow:

```python
# Training with MLflow tracking
from privateer_ad.train.train_alert_filter import train_alert_filter

model_path = train_alert_filter(
    epochs=100,
    track_mlflow=True
)
```

This will log parameters, metrics, and the trained model to MLflow for experiment tracking.
