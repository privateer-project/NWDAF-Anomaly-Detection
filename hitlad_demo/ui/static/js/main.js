// Global state
let currentStep = 0;
let currentAnomalyIndex = null;
let totalSteps = 5;
let session_data = {
    'current_step': 0,
    'total_steps': 5,
    'anomaly_data': null,
    'current_anomaly_idx': 0,
    'feedback_collected': 0,
    'results': null,
    'sampled_indices': null
};

// DOM Elements
const progressBar = document.querySelector('.progress-bar');
const prevButton = document.getElementById('prev-btn');
const nextButton = document.getElementById('next-btn');

// HTML Templates
const ANOMALY_INFO_TEMPLATE = `
    <h4>Anomaly Details</h4>
    <p>Reconstruction Error: <span id="reconstruction-error"></span></p>
    <div class="progress mb-3">
        <div class="progress-bar bg-info" role="progressbar" style="width: 0%">
            0/0 Anomalies Reviewed
        </div>
    </div>
`;

// Panel IDs to reset on restart
const panelsToReset = [
    'detection-results',
    'training-status',
    'results-stats'
];

// Helper functions
function updateProgress(step) {
    // Calculate progress percentage
    const progress = step === totalSteps ? 100 : (step / totalSteps) * 100;
    
    // Update progress bar with animation
    progressBar.style.transition = 'width 0.6s ease, background-color 0.6s ease';
    progressBar.style.width = `${progress}%`;
    progressBar.setAttribute('aria-valuenow', progress);
    
    // Update text and style for completion
    if (step === totalSteps) {
        progressBar.textContent = 'Complete!';
        progressBar.style.backgroundColor = '#28a745';  // Bootstrap success color
    } else {
        progressBar.textContent = `Step ${step}/${totalSteps}`;
        progressBar.style.backgroundColor = '#3498db';  // Original color
    }
}

function showPanel(step) {
    // First, start hiding all panels by removing active class
    document.querySelectorAll('.panel').forEach(panel => {
        panel.classList.remove('active');
    });
    
    // After opacity transition, hide panels completely
    setTimeout(() => {
        document.querySelectorAll('.panel').forEach(panel => {
            if (!panel.classList.contains('active')) {
                panel.style.display = 'none';
            }
        });
        
        // Show the new panel
        const currentPanel = document.getElementById(`step${step}`);
        if (currentPanel) {
            currentPanel.style.display = 'block';
            // Use requestAnimationFrame to ensure display is set before adding active class
            requestAnimationFrame(() => {
                currentPanel.classList.add('active');
            });
        }
    }, 500);
    
    // Update navigation buttons immediately
    if (step === totalSteps) {
        // Final success panel - hide both buttons
        prevButton.style.display = 'none';
        nextButton.style.display = 'none';
    } else {
        // Normal navigation
        prevButton.style.display = step > 0 ? 'block' : 'none';
        nextButton.style.display = step < totalSteps - 1 ? 'block' : 'none';
    }
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
    // This function is called when clicking "Start Over"
    try {
        // First hide all panels with fade-out animation
        document.querySelectorAll('.panel').forEach(panel => {
            panel.style.display = 'none';
            panel.classList.remove('active');
        });

        // Reset all state variables
        currentStep = 0;
        currentAnomalyIndex = null;
        
        // Reset session data
        session_data = {
            'current_step': 0,
            'total_steps': 5,
            'anomaly_data': null,
            'current_anomaly_idx': 0,
            'feedback_collected': 0,
            'results': null,
            'sampled_indices': null
        };
        
        // Reset progress bar
        updateProgress(currentStep);
        
        // Clear all panel contents
        panelsToReset.forEach(panelId => {
            const panel = document.getElementById(panelId);
            if (panel) {
                panel.innerHTML = '';
            }
        });

        const response = await fetch('/api/start_demo', {
            method: 'POST'
        });
        const data = await response.json();
        
        if (data.status === 'success') {
            // Show welcome panel with animation
            const welcomePanel = document.getElementById('step0');
            if (welcomePanel) {
                welcomePanel.style.display = 'block';
                // Allow display:block to take effect before adding active class
                requestAnimationFrame(() => {
                    welcomePanel.classList.add('active');
                });
            }
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
            const errorSpan = document.getElementById('reconstruction-error');
            const progressBar = document.querySelector('#anomaly-info .progress-bar');
            
            if (!errorSpan || !progressBar) {
                // Reset the HTML structure if elements are missing
                const anomalyInfo = document.getElementById('anomaly-info');
                if (anomalyInfo) {
                    anomalyInfo.innerHTML = ANOMALY_INFO_TEMPLATE;
                }
            }
            
            // Try updating elements again after potential reset
            document.getElementById('reconstruction-error').textContent = data.reconstruction_error.toFixed(6);
            
            // Update progress bar
            const updatedProgressBar = document.querySelector('#anomaly-info .progress-bar');
            const progress = (data.progress.current / data.progress.total) * 100;
            updatedProgressBar.style.width = `${progress}%`;
            updatedProgressBar.textContent = `${data.progress.current}/${data.progress.total} Anomalies Reviewed`;
            
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
            // Display performance metrics cards
            statsDiv.innerHTML = `
                <div class="row mb-4">
                    <div class="col-md-4">
                        <div class="card text-center h-100">
                            <div class="card-body">
                                <h5 class="card-title">False Positive Reduction</h5>
                                <p class="display-4 text-success">${data.statistics.false_positive_reduction.toFixed(1)}%</p>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-4">
                        <div class="card text-center h-100">
                            <div class="card-body">
                                <h5 class="card-title">True Positive Retention</h5>
                                <p class="display-4 text-primary">${data.statistics.true_positive_retention.toFixed(1)}%</p>
                            </div>
                        </div>
                    </div>
                    </div>
                </div>
                
                <div class="card mb-4">
                    <div class="card-header">
                        <h5 class="mb-0">Detailed Comparison</h5>
                    </div>
                    <div class="card-body">
                        <div class="table-responsive">
                            <table class="table table-hover">
                                <thead>
                                    <tr>
                                        <th>Sample</th>
                                        <th>Initial Label</th>
                                        <th>Feedback</th>
                                        <th>Final Decision</th>
                                        <th>Status</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    ${data.reviewed_anomalies.map(anomaly => `
                                        <tr class="${anomaly.matches_feedback ? 'table-success' : 'table-danger'}">
                                            <td>${anomaly.index}</td>
                                            <td>
                                                <span class="badge bg-warning text-dark">
                                                    <i class="bi bi-exclamation-triangle me-1"></i>
                                                    ${anomaly.initial_label}
                                                </span>
                                            </td>
                                            <td>
                                                <span class="badge ${anomaly.feedback === 'True Positive' ? 'bg-success' : 'bg-danger'}">
                                                    ${anomaly.feedback}
                                                </span>
                                            </td>
                                            <td>
                                                <span class="badge ${anomaly.final_decision === 'Anomaly' ? 'bg-warning text-dark' : 'bg-secondary'}">
                                                    ${anomaly.final_decision}
                                                </span>
                                            </td>
                                            <td>
                                                <i class="bi ${anomaly.matches_feedback ? 'bi-check-circle-fill text-success' : 'bi-x-circle-fill text-danger'}"></i>
                                                ${anomaly.matches_feedback ? 'Matches Feedback' : 'Mismatch'}
                                            </td>
                                        </tr>
                                    `).join('')}
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
            `;
            
            // Clear visualization div
            if (vizDiv) {
                vizDiv.innerHTML = '';
            }

            // Add end demo button
            statsDiv.insertAdjacentHTML('afterend', `
                <div class="text-center mt-4">
                    <button class="btn btn-success btn-lg" onclick="endDemo()">
                        <i class="bi bi-flag-fill me-2"></i>End Demo
                    </button>
                </div>
            `);
        } else {
            showError(data.message, 'results-stats');
        }
    } catch (error) {
        showError('Error evaluating results: ' + error.message, 'results-stats');
    }
}

// Initialize and start demo
async function initializeAndStartDemo() {
    try {
        // Initialize demo backend and wait for response
        const response = await fetch('/api/start_demo', {
            method: 'POST'
        });
        const data = await response.json();
        
        if (data.status === 'success') {
            // Only show a success message and let user start when ready
            const startButton = document.querySelector('#step0 .btn-primary');
            if (startButton) {
                startButton.innerHTML = 'Click to Begin Demo';
                startButton.onclick = nextStep;
            }
        } else {
            showError('Failed to initialize demo', 'step0');
        }
    } catch (error) {
        showError('Failed to initialize demo: ' + error.message, 'step0');
    }
}

// Handle demo completion
function endDemo() {
    // Animate the end button
    const endButton = document.querySelector('.btn-success.btn-lg');
    if (endButton) {
        endButton.disabled = true;
        // endButton.innerHTML = `
        //     <span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span>
        //     Finishing Demo...
        // `;
    }
    
    // Transition to completion state after a short delay
    setTimeout(() => {
        currentStep++;
        updateProgress(currentStep);
        showPanel(currentStep);
    }, 800);
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
                // Initialize the anomaly info panel with template
                const anomalyInfo = document.getElementById('anomaly-info');
                if (anomalyInfo) {
                    anomalyInfo.innerHTML = ANOMALY_INFO_TEMPLATE;
                }
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
