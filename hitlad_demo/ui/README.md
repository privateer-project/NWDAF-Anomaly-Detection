# HITLAD Demo Web Interface

This is a web-based user interface for the HITLAD (Human-In-The-Loop Anomaly Detection) demo. It provides an interactive way to run the alert filter demo, collect feedback on anomalies, and visualize the results.

## Features

- Step-by-step guided workflow
- Interactive anomaly review interface
- Real-time feedback collection
- Visual progress tracking
- Results visualization
- Clear error handling and status updates

## Prerequisites

- Python 3.x
- pip3
- Web browser (Chrome, Firefox, Safari, or Edge)
- Trained autoencoder model (should be present in the specified path in config.yaml)

## Installation

1. Navigate to the UI directory:
   ```bash
   cd hitlad_demo/ui
   ```

2. Install dependencies:
   ```bash
   pip3 install -r requirements.txt
   ```

## Running the Demo

1. Make sure the run script is executable:
   ```bash
   chmod +x run_ui.sh
   ```

2. Run the UI server:
   ```bash
   ./run_ui.sh
   ```

3. Open your web browser and navigate to:
   ```
   http://localhost:5000
   ```

## Usage Guide

### Step 1: Start Demo
- Click the "Start Demo" button to initialize the demo environment
- The system will load necessary models and prepare for anomaly detection

### Step 2: Initial Anomaly Detection
- Click "Detect Anomalies" to run the initial detection process
- The system will display the number of anomalies found

### Step 3: Feedback Collection
- Review each detected anomaly
- For each anomaly:
  - Review the reconstruction error and other metrics
  - Click "True Positive" if you agree it's an anomaly
  - Click "False Positive" if you believe it's not an anomaly
- Progress bar shows how many anomalies you've reviewed

### Step 4: Train Alert Filter
- After providing feedback, click "Train Filter" to train the alert filter model
- Wait for the training process to complete

### Step 5: View Results
- Review the comparison between initial and filtered results
- See statistics about false positive reduction
- View visualizations of the improvement

## Troubleshooting

### Common Issues

1. **Server won't start**
   - Check if port 5000 is available
   - Ensure all dependencies are installed
   - Verify Python version is compatible

2. **Models not found**
   - Verify paths in config.yaml
   - Ensure autoencoder model is present in the specified location

3. **API errors**
   - Check browser console for detailed error messages
   - Verify server is running and accessible
   - Check network connectivity

### Error Messages

If you encounter error messages:
1. Read the error message carefully - it should indicate the specific issue
2. Check the server terminal for additional details
3. Verify all prerequisites are met
4. Try restarting the server

## Contributing

To contribute to the UI:
1. Follow the project's coding style
2. Test your changes thoroughly
3. Update documentation as needed
4. Submit a pull request

## Architecture

The UI consists of:
- Flask backend (`app.py`)
- HTML templates (`templates/`)
- CSS styles (`static/css/`)
- JavaScript frontend (`static/js/`)
- Configuration and run scripts

The interface communicates with the backend through REST APIs and updates the UI dynamically based on the demo's progress.
