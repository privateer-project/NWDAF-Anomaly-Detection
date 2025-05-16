#!/usr/bin/env python3
"""
HITLAD Demo Web Interface

This application provides a web-based interface for the HITLAD demo,
allowing interactive control of the demo steps and visualization of results.
"""

import os
import sys
import yaml
import numpy as np
from pathlib import Path
from flask import Flask, render_template, jsonify, request
from typing import Dict, List, Optional

# Add parent directory to path to import privateer_ad modules
parent_dir = str(Path(__file__).resolve().parents[2])
sys.path.append(parent_dir)

from hitlad_demo.alert_filter_demo import AlertFilterDemo

app = Flask(__name__)

# Initialize paths
parent_dir = str(Path(__file__).resolve().parents[1])
demo_config_path = os.path.join(parent_dir, "config.yaml")

# Load and modify config to use absolute paths
with open(demo_config_path, 'r') as f:
    config = yaml.safe_load(f)

# Convert relative paths to absolute paths and ensure directories exist
for key in config['paths']:
    if isinstance(config['paths'][key], str):
        if config['paths'][key].startswith('./'):
            config['paths'][key] = os.path.join(parent_dir, config['paths'][key][2:])
        # Create directories for paths that should be directories
        if key.endswith('_dir'):
            os.makedirs(config['paths'][key], exist_ok=True)
        # Create parent directories for file paths
        elif key.endswith('_path'):
            os.makedirs(os.path.dirname(config['paths'][key]), exist_ok=True)

# Initialize demo with modified config
demo = AlertFilterDemo(config_path=demo_config_path, config_dict=config)

print(f"Starting server with config:")
print(f"Model path: {config['paths']['autoencoder_model_path']}")
print(f"Filter path: {config['paths']['filter_model_path']}")
print(f"Data path: {config['demo']['data_path']}")

# Store session data
session_data = {
    'current_step': 0,
    'total_steps': 5,
    'anomaly_data': None,
    'current_anomaly_idx': 0,
    'feedback_collected': 0,
    'results': None,
    'sampled_indices': None
}

@app.route('/')
def index():
    """Render the main demo interface."""
    return render_template('index.html')

@app.route('/api/start_demo', methods=['POST'])
def start_demo():
    """Initialize/reset the demo."""
    global session_data
    session_data = {
        'current_step': 0,
        'total_steps': 5,
        'anomaly_data': None,
        'current_anomaly_idx': 0,
        'feedback_collected': 0,
        'results': None,
        'sampled_indices': None
    }
    return jsonify({'status': 'success', 'step': 0})

@app.route('/api/detect_anomalies', methods=['POST'])
def detect_anomalies():
    """Run initial anomaly detection."""
    try:
        # Run detection without filter
        inputs, latents, losses, predictions, _, labels = demo.detect_anomalies(use_filter=False)
        
        # Store data for feedback collection
        session_data['anomaly_data'] = {
            'inputs': inputs.tolist(),
            'latents': latents.tolist(),
            'losses': losses.tolist(),
            'predictions': predictions.tolist(),
            'labels': labels.tolist()
        }
        
        # Count anomalies
        anomaly_count = np.sum(predictions == 1)
        
        return jsonify({
            'status': 'success',
            'anomaly_count': int(anomaly_count),
            'total_samples': len(predictions)
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/get_next_anomaly', methods=['GET'])
def get_next_anomaly():
    """Get the next anomaly for feedback collection."""
    if not session_data['anomaly_data']:
        return jsonify({'status': 'error', 'message': 'No anomaly data available'})
    
    if session_data.get('sampled_indices') is None:
        # First call - initialize sampled indices
        predictions = np.array(session_data['anomaly_data']['predictions'])
        anomaly_indices = np.where(predictions == 1)[0]
        sample_size = min(config['demo']['num_anomalies'], len(anomaly_indices))
        session_data['sampled_indices'] = np.random.choice(anomaly_indices, size=sample_size, replace=False)
        
    if session_data['current_anomaly_idx'] >= len(session_data['sampled_indices']):
        return jsonify({
            'status': 'complete',
            'message': 'All anomalies reviewed'
        })
    
    idx = session_data['sampled_indices'][session_data['current_anomaly_idx']]
    
    return jsonify({
        'status': 'success',
        'anomaly_index': int(idx),
        'reconstruction_error': float(session_data['anomaly_data']['losses'][idx]),
        'true_label': int(session_data['anomaly_data']['labels'][idx]),
        'progress': {
            'current': session_data['current_anomaly_idx'] + 1,
            'total': len(session_data['sampled_indices'])
        }
    })

@app.route('/api/submit_feedback', methods=['POST'])
def submit_feedback():
    """Submit feedback for an anomaly."""
    data = request.json
    anomaly_idx = data.get('anomaly_index')
    is_true_positive = data.get('is_true_positive')
    
    if anomaly_idx is None or is_true_positive is None:
        return jsonify({'status': 'error', 'message': 'Missing required data'})
    
    try:
        # Add feedback
        demo.feedback_collector.add_feedback(
            latent=np.array(session_data['anomaly_data']['latents'][anomaly_idx]),
            anomaly_decision=1,  # It was detected as anomaly
            reconstruction_error=session_data['anomaly_data']['losses'][anomaly_idx],
            user_feedback=int(is_true_positive)
        )
        
        # Update session
        session_data['current_anomaly_idx'] += 1
        session_data['feedback_collected'] += 1
        
        return jsonify({
            'status': 'success',
            'feedback_collected': session_data['feedback_collected']
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/train_filter', methods=['POST'])
def train_filter():
    """Train the alert filter model."""
    try:
        demo.train_filter_model()
        return jsonify({'status': 'success'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/evaluate_results', methods=['POST'])
def evaluate_results():
    """Run final evaluation with the trained filter."""
    try:
        # Run detection with filter
        _, _, _, predictions, filtered_decisions, labels = demo.detect_anomalies(use_filter=True)
        
        # Store results
        session_data['results'] = {
            'predictions': predictions.tolist(),
            'filtered_decisions': filtered_decisions.tolist(),
            'labels': labels.tolist()
        }
        
        # Calculate statistics
        original_anomalies = np.sum(predictions == 1)
        filtered_anomalies = np.sum(filtered_decisions == 1)
        reduction = ((original_anomalies - filtered_anomalies) / original_anomalies * 100)
        
        return jsonify({
            'status': 'success',
            'statistics': {
                'original_anomalies': int(original_anomalies),
                'filtered_anomalies': int(filtered_anomalies),
                'reduction_percentage': float(reduction)
            }
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
