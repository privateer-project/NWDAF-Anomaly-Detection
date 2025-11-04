# Development Scripts

This directory contains functional test scripts for debugging and verifying the HITL system components.

## Available Scripts

### 1. `test_database.py`
Tests the complete database layer including SQLite, repository, and all CRUD operations.

```bash
python dev_scripts/test_database.py
```

**Tests:**
- Database initialization and schema creation
- Anomaly upsert and retrieval
- Feedback submission
- Schema management
- Vector storage and retrieval
- Model registration
- Settings and live model management

### 2. `test_schema_registry.py`
Tests schema registration and validation for different tensor shapes.

```bash
python dev_scripts/test_schema_registry.py
```

**Tests:**
- 1D tensor schema registration (dense mode)
- 2D tensor schema registration (conv1d mode)
- Schema ID determinism
- Shape validation
- Schema retrieval from anomalies

### 3. `test_serialization.py`
Tests tensor serialization to/from .npy format.

```bash
python dev_scripts/test_serialization.py
```

**Tests:**
- NumPy array encoding/decoding
- Round-trip preservation
- Shape validation
- NaN/Inf detection
- List to array conversion

### 4. `test_artifacts.py`
Tests model artifact management and versioning.

```bash
python dev_scripts/test_artifacts.py
```

**Tests:**
- Version creation and sequencing
- Model saving/loading
- Config, scaler, threshold management
- Artifact listing and deletion

### 5. `test_full_workflow.py`
End-to-end workflow demonstrating complete integration.

```bash
python dev_scripts/test_full_workflow.py
```

**Demonstrates:**
- Anomaly ingestion and storage
- Schema registration
- Vector serialization and storage
- Feedback collection
- Model artifact management

### 6. `test_utils.py`
Tests utility functions (time, IDs, logging).

```bash
python dev_scripts/test_utils.py
```

**Tests:**
- Timestamp generation and parsing
- ID generation (UUID, schema ID, model version)
- Logging configuration
- Config management

## Running All Scripts

```bash
# Run all dev scripts
for script in dev_scripts/test_*.py; do
    echo "Running $script..."
    python "$script"
    echo "---"
done
```

## Environment Setup

Make sure to set up your environment first:

```bash
# From project root
cd human_in_the_loop
pip install -e .
```

## Cleanup

Some scripts create temporary files. To clean up:

```bash
rm -rf /tmp/hitl_dev_*
rm -rf dev_artifacts/
rm -rf dev_data.db
```
