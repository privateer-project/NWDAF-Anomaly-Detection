// Global state
let currentStep = 0;
let currentAnomalyIndex = null;
let totalSteps = 5;

// DOM Elements
const progressBar = document.querySelector('.progress-bar');
const prevButton = document.getElementById('prev-btn');
const nextButton = document.getElementById('next-btn');

// Helper functions
function updateProgress(step) {
    const progress = (step / totalSteps) * 100;
    progressBar.style.width = `${progress}%`;
    progressBar.setAttribute('aria-valuenow', progress);
    progressBar.textContent = `Step ${step}/${totalSteps}`;
}

function showPanel(step) {
    // Hide all panels
    document.querySelectorAll('.panel').forEach(panel => {
        panel.classList.remove('active');
    });
    
    // Show the current panel
    const currentPanel = document.getElementById(`step${step}`);
    if (currentPanel) {
        currentPanel.classList.add('active');
    }
    
    // Update navigation buttons
    prevButton.style.display = step > 0 ? 'block' : 'none';
    nextButton.style.display = step < totalSteps - 1 ? 'block' : 'none';
}

function showError(message, elementId) {
    const element = document.getElementById(elementId);
    element.innerHTML = `
        <div class="status-message status-error">
            ${message}
        </div>
    `;
}

function showLoading(elementId) {
    const element = document.getElementById(elementId);
    element.innerHTML = `
        <div class="d-flex align-items-center">
            <div class="spinner-border text-primary" role="status">
                <span class="visually-hidden">Loading...</span>
            </div>
            <span>Processing...</span>
        </div>
    `;
}

// Step navigation
function nextStep() {
    if (currentStep < totalSteps - 1) {
        currentStep++;
        updateProgress(currentStep);
        showPanel(currentStep);
    }
}

function previousStep() {
    if (currentStep > 0) {
        currentStep--;
        updateProgress(currentStep);
        showPanel(currentStep);
    }
}

// Demo functions
async function startDemo() {
    try {
        const response = await fetch('/api/start_demo', {
            method: 'POST'
        });
        const data = await response.json();
        
        if (data.status === 'success') {
            nextStep();
        } else {
            showError('Failed to start demo', 'step0');
        }
    } catch (error) {
        showError('Failed to start demo: ' + error.message, 'step0');
    }
}

async function detectAnomalies() {
    const resultsDiv = document.getElementById('detection-results');
    showLoading('detection-results');
    
    try {
        const response = await fetch('/api/detect_anomalies', {
            method: 'POST'
        });
        const data = await response.json();
        
        if (data.status === 'success') {
            resultsDiv.innerHTML = `
                <div class="status-message status-success">
                    <h5>Detection Complete</h5>
                    <p>Found ${data.anomaly_count} anomalies in ${data.total_samples} samples.</p>
                    <button class="btn btn-primary mt-2" onclick="nextStep()">Proceed to Feedback Collection</button>
                </div>
            `;
        } else {
            showError(data.message, 'detection-results');
        }
    } catch (error) {
        showError('Error during anomaly detection: ' + error.message, 'detection-results');
    }
}

async function getNextAnomaly() {
    try {
        const response = await fetch('/api/get_next_anomaly');
        const data = await response.json();
        
        if (data.status === 'success') {
            currentAnomalyIndex = data.anomaly_index;
            
            // Update UI
            document.getElementById('reconstruction-error').textContent = data.reconstruction_error.toFixed(6);
            
            // Update progress bar
            const progressBar = document.querySelector('#anomaly-info .progress-bar');
            const progress = (data.progress.current / data.progress.total) * 100;
            progressBar.style.width = `${progress}%`;
            progressBar.textContent = `${data.progress.current}/${data.progress.total} Anomalies Reviewed`;
            
            return true;
        } else if (data.status === 'complete') {
            document.getElementById('anomaly-info').innerHTML = `
                <div class="status-message status-success">
                    <h5>Feedback Collection Complete</h5>
                    <p>All anomalies have been reviewed.</p>
                    <button class="btn btn-primary mt-2" onclick="nextStep()">Proceed to Training</button>
                </div>
            `;
            return false;
        } else {
            showError(data.message, 'anomaly-info');
            return false;
        }
    } catch (error) {
        showError('Error fetching next anomaly: ' + error.message, 'anomaly-info');
        return false;
    }
}

async function submitFeedback(isTrue) {
    if (currentAnomalyIndex === null) {
        showError('No active anomaly to provide feedback for', 'anomaly-info');
        return;
    }
    
    try {
        const response = await fetch('/api/submit_feedback', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                anomaly_index: currentAnomalyIndex,
                is_true_positive: isTrue
            })
        });
        
        const data = await response.json();
        
        if (data.status === 'success') {
            // Get next anomaly
            await getNextAnomaly();
        } else {
            showError(data.message, 'anomaly-info');
        }
    } catch (error) {
        showError('Error submitting feedback: ' + error.message, 'anomaly-info');
    }
}

async function trainFilter() {
    const statusDiv = document.getElementById('training-status');
    showLoading('training-status');
    
    try {
        const response = await fetch('/api/train_filter', {
            method: 'POST'
        });
        const data = await response.json();
        
        if (data.status === 'success') {
            statusDiv.innerHTML = `
                <div class="status-message status-success">
                    <h5>Training Complete</h5>
                    <p>The alert filter model has been trained successfully.</p>
                    <button class="btn btn-primary mt-2" onclick="nextStep()">View Results</button>
                </div>
            `;
        } else {
            showError(data.message, 'training-status');
        }
    } catch (error) {
        showError('Error during training: ' + error.message, 'training-status');
    }
}

async function evaluateResults() {
    const statsDiv = document.getElementById('results-stats');
    const vizDiv = document.getElementById('results-viz');
    showLoading('results-stats');
    
    try {
        const response = await fetch('/api/evaluate_results', {
            method: 'POST'
        });
        const data = await response.json();
        
        if (data.status === 'success') {
            // Display statistics
            statsDiv.innerHTML = `
                <div class="card">
                    <div class="card-body">
                        <h5 class="card-title">Performance Metrics</h5>
                        <p><strong>Original Anomalies:</strong> ${data.statistics.original_anomalies}</p>
                        <p><strong>Filtered Anomalies:</strong> ${data.statistics.filtered_anomalies}</p>
                        <p><strong>Reduction:</strong> ${data.statistics.reduction_percentage.toFixed(2)}%</p>
                    </div>
                </div>
            `;
            
            // Add visualization
            vizDiv.innerHTML = `
                <div class="card">
                    <div class="card-body">
                        <div class="alert alert-info">
                            Reduction in Anomalies
                            <div class="progress mt-2" style="height: 25px;">
                                <div class="progress-bar bg-success" role="progressbar" 
                                     style="width: ${100 - data.statistics.reduction_percentage}%">
                                    After Filter (${data.statistics.filtered_anomalies})
                                </div>
                            </div>
                            <div class="progress mt-2" style="height: 25px;">
                                <div class="progress-bar bg-primary" role="progressbar" style="width: 100%">
                                    Before Filter (${data.statistics.original_anomalies})
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            `;
        } else {
            showError(data.message, 'results-stats');
        }
    } catch (error) {
        showError('Error evaluating results: ' + error.message, 'results-stats');
    }
}

// Initialize UI
document.addEventListener('DOMContentLoaded', () => {
    updateProgress(currentStep);
    showPanel(currentStep);
    
    // Start getting the first anomaly when step 2 becomes active
    const observer = new MutationObserver((mutations) => {
        mutations.forEach((mutation) => {
            if (mutation.target.classList.contains('active') && 
                mutation.target.id === 'step2') {
                getNextAnomaly();
            }
        });
    });
    
    const step2Panel = document.getElementById('step2');
    observer.observe(step2Panel, { 
        attributes: true, 
        attributeFilter: ['class']
    });
});
