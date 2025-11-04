# Dev Scripts Quick Reference

This directory contains functional development scripts for testing and debugging the HITL system.

## Quick Start

```bash
# From human_in_the_loop directory
cd dev_scripts

# Run all tests
./run_all.sh

# Or run individual tests
python test_utils.py
python test_serialization.py
python test_database.py
python test_schema_registry.py
python test_artifacts.py
python test_full_workflow.py
```

## Test Coverage

| Script | What It Tests | Runtime |
|--------|---------------|---------|
| `test_utils.py` | Timestamps, IDs, logging, config | ~2s |
| `test_serialization.py` | NumPy encoding/decoding, validation | ~3s |
| `test_database.py` | SQLite, repository, CRUD operations | ~2s |
| `test_schema_registry.py` | Schema management, validation | ~2s |
| `test_artifacts.py` | Model versioning, save/load | ~3s |
| `test_full_workflow.py` | End-to-end integration (all phases) | ~5s |

**Total runtime: ~17 seconds**

## What Gets Tested

### Phase 1: Foundation & Utilities ✓
- ISO 8601 timestamp generation
- UUID generation
- Deterministic ID generation
- Structured logging configuration
- TOML configuration loading

### Phase 2: Type System ✓
- Pydantic models (covered in unit tests)
- TypedDict structures (covered in unit tests)

### Phase 3: Database Layer ✓
- SQLite initialization with WAL mode
- Anomaly CRUD operations
- Feedback submission and retrieval
- Schema registration
- Vector storage
- Model registration
- Settings management
- Transaction handling

### Phase 4: I/O & Serialization ✓
- NumPy array encoding to .npy format
- NumPy array decoding from .npy format
- Shape preservation
- Dtype preservation
- NaN/Inf detection
- Array validation
- Type casting

### Phase 5: Schema Registry ✓
- 1D schema registration (dense mode)
- 2D schema registration (conv1d mode)
- Schema ID determinism
- Shape validation
- Dtype validation
- Schema retrieval from anomalies

### Phase 6: Artifacts Manager ✓
- Date-based versioning (AE-YYYY.MM.DD-N)
- PyTorch model saving/loading
- Config JSON persistence
- Scaler JSON persistence
- Threshold management
- Complete artifact loading
- Version listing and deletion
- weights_only security

### Integration Test (test_full_workflow.py) ✓
Complete lifecycle simulation:
1. **Detection**: Ingest 10 network anomalies
2. **Feedback**: Collect 10 expert reviews (70% TP, 30% FP)
3. **Training**: Train autoencoder on 7 true positives
4. **Versioning**: Save model + config + scaler + threshold
5. **Deployment**: Set as live model
6. **Inference**: Process 10 new samples

## Debugging Tips

### Database Issues
```bash
python test_database.py
# Check: Database creation, CRUD operations, transactions
```

### Serialization Problems
```bash
python test_serialization.py
# Check: Array encoding, NaN/Inf handling, shape preservation
```

### Schema Errors
```bash
python test_schema_registry.py
# Check: Schema registration, validation, ID determinism
```

### Artifact Loading Failures
```bash
python test_artifacts.py
# Check: Model persistence, version management
```

### Integration Issues
```bash
python test_full_workflow.py
# Check: Complete pipeline from detection to inference
```

## Environment

All tests use temporary directories and clean up automatically.
No persistent state is created.

## Requirements

All dependencies from `pyproject.toml`:
- Python 3.10+
- PyTorch
- NumPy
- Pydantic 2.7+
- structlog 24.1+
- pytest (for unit tests)

## Next Steps

After verifying all dev scripts pass:
1. Run full unit test suite: `make test` (from parent directory)
2. Check coverage: `make coverage`
3. Continue with Phase 7: Model Architectures

## Troubleshooting

**Import errors**: Ensure you're running from `human_in_the_loop` directory
**Module not found**: Check virtual environment is activated
**Test failures**: Check the specific error message - scripts include detailed diagnostics

---

Created: 2025-11-04
Last Updated: 2025-11-04
