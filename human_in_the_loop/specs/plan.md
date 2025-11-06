# HITL Implementation Plan

**Project:** Human-in-the-Loop Anomaly Filtering System
**Version:** 0.1.0 (MVP v0.4T-MD)
**Created:** November 4, 2025
**Last Updated:** November 4, 2025
**Estimated Duration:** 4-5 weeks (single developer)

---

## Progress Summary

**Completed Phases:** 7 of 14 (50%)
**Total Tests Passing:** 212 tests
**Test Coverage:** Comprehensive unit and integration tests

### Completed ✅
- **Phase 1:** Foundation & Utilities (29 tests)
- **Phase 2:** Type System (28 tests)
- **Phase 3:** Database Layer (29 tests)
- **Phase 4:** I/O & Serialization (32 tests)
- **Phase 5:** Schema Registry (32 tests)
- **Phase 6:** Artifacts Manager (27 tests)
- **Phase 7:** Model Architectures (35 tests)
- **Bonus:** Dev Scripts (6 functional test scripts)

### In Progress 🚧
- **Phase 8:** Training Pipeline (next up)

### Remaining 📋
- Phase 9: Inference Service
- Phase 10: HITL Orchestrator
- Phase 11: CLI Interface
- Phase 12: HTTP API
- Phase 13: Testing & Documentation
- Phase 14: End-to-End Validation

---

## Overview

This document outlines the implementation plan for the HITL system, organized into 14 phases based on logical dependencies and complexity. Each phase builds upon previous phases, ensuring a solid foundation before adding complexity.

---

## Phase 1: Foundation & Utilities ⚡

**Duration:** 1-2 days
**Dependencies:** None
**Priority:** CRITICAL - Start here!

### Files to Implement

1. **`hitl/utils/time.py`**
   - `utcnow()` - Return timezone-aware UTC datetime
   - `now_iso()` - Return ISO 8601 formatted string
   - `parse_iso()` - Parse ISO string to datetime
   - `to_iso()` - Convert datetime to ISO string
   - `validate_iso()` - Check if string is valid ISO timestamp

2. **`hitl/utils/ids.py`**
   - `uuid_str()` - Generate UUID4 strings
   - `sha1_bytes()` - Hash bytes with SHA-1
   - `schema_id()` - Generate deterministic schema ID from shape+dtype
   - `model_version()` - Generate model version string (AE-YYYY.MM.DD-N)
   - `feedback_id()` - Generate feedback IDs

3. **`hitl/utils/logging.py`**
   - `configure_logging()` - Setup structlog with processors
   - `get_logger()` - Return logger bound to module name
   - `bind_context()` - Bind context variables
   - `clear_context()` - Clear thread-local context

4. **`hitl/settings.py`**
   - `Config` dataclass with fields: sqlite_path, artifacts_dir, mode, log_level
   - `get_env_config()` - Read environment variables (HITL_*)
   - `paths()` - Resolve and create directory paths
   - `validate_config()` - Validate configuration values

5. **`hitl/errors.py`**
   - `HITLError` - Base exception
   - `UnsupportedShape` - Invalid tensor shape
   - `ShapeMismatch` - Shape doesn't match schema
   - `SchemaNotFound` - Schema ID not found
   - `NoLiveModel` - No live model set
   - `ArtifactMissing` - Model artifacts not found
   - `DBError` - Database errors
   - `ValidationError` - Data validation failures

### Testing
- Unit tests for each utility function
- No integration needed yet
- Focus on edge cases (invalid inputs, timezone handling, etc.)

### Success Criteria
- [x] All utility functions implemented and tested
- [x] Config can be loaded from environment
- [x] Logger produces structured output
- [x] All exceptions have clear error messages

### Status: ✅ COMPLETED (29 tests passing)

---

## Phase 2: Type System 📋

**Duration:** 1 day
**Dependencies:** Phase 1
**Priority:** HIGH - Establishes contracts

### Files to Implement

1. **`hitl/types.py`**
   - **TypedDicts:**
     - `SchemaInfo` - shape, ndim, numel, dtype, schema_id
     - `TrainParams` - mode, epochs, batch_size, lr, val_split, patience, percentile
     - `PredictResult` - label, score, threshold, model_version
     - `AnomalyRecord` - anomaly_id, occurred_at, source, created_at, updated_at
     - `FeedbackRecord` - feedback_id, anomaly_id, user_id, label, confidence, note, created_at

   - **Pydantic Models:**
     - `AnomalyUpsert` - Request for upserting anomaly
     - `FeedbackIn` - Request for submitting feedback
     - `TrainRequest` - Request for training
     - `PredictIn` - Request for prediction (with validation)
     - `PredictOut` - Response for prediction

   - **Protocols:**
     - `RepositoryProtocol` - Interface for repository

   - **Type Aliases:**
     - `Tensor1D`, `Tensor2D`

### Testing
- Pydantic model validation tests
- TypedDict structure tests
- No logic to test, mainly structure

### Success Criteria
- [x] All TypedDicts defined with correct fields
- [x] All Pydantic models with validators
- [x] Type hints work with mypy --strict
- [x] Clear documentation strings

### Status: ✅ COMPLETED (28 tests passing)

---

## Phase 3: Database Layer 🗄️

**Duration:** 2-3 days
**Dependencies:** Phase 1, 2
**Priority:** CRITICAL - Foundation for persistence

### Files to Implement

1. **`hitl/ddl.sql`**
   - Complete SQL schema for 6 tables:
     - `anomalies` - Anomaly metadata
     - `feedback` - Human labels
     - `feature_schemas` - Tensor schemas
     - `raw_vectors` - Tensor BLOBs
     - `models` - Model registry
     - `settings` - Key-value config
   - All foreign keys and indexes
   - WAL mode pragma

2. **`hitl/store/sqlite.py`**
   - `SQLite` class:
     - `__init__()` - Create DB file, apply DDL
     - `connect()` - Return configured connection
     - `execute()` - Execute statement
     - `fetchone()` - Query single row
     - `fetchall()` - Query multiple rows
     - `tx()` - Transaction context manager
     - `_apply_ddl()` - Read and execute DDL
     - `_ensure_wal_mode()` - Set WAL mode

3. **`hitl/store/repository.py`**
   - `Repository` class with methods:
     - **Anomalies:** `upsert_anomaly()`, `get_anomaly()`, `list_anomalies()`
     - **Feedback:** `insert_feedback()`, `latest_feedback()`, `list_feedback()`
     - **Schemas:** `get_schema()`, `insert_schema()`, `list_schemas()`
     - **Vectors:** `put_vector()`, `get_vector()`, `iter_vectors()`, `count_vectors()`
     - **Models:** `insert_model()`, `get_model()`, `list_models()`
     - **Settings:** `set_setting()`, `get_setting()`, `set_live_model()`, `get_live_model()`

### Testing
- **`tests/test_store.py`** - Comprehensive CRUD tests
- Transaction rollback tests
- Foreign key constraint tests
- Concurrent access tests (if needed)

### Success Criteria
- [x] Database schema applied successfully
- [x] All CRUD operations work
- [x] Transactions commit/rollback properly
- [x] Foreign keys enforced
- [x] 100% test coverage for store module

### Status: ✅ COMPLETED (29 tests passing)

---

## Phase 4: I/O & Serialization 💾

**Duration:** 1-2 days
**Dependencies:** Phase 1, 2
**Priority:** HIGH - Required for tensor storage

### Files to Implement

1. **`hitl/io/serialization.py`**
   - `encode_npy()` - NumPy array → .npy bytes
   - `decode_npy()` - .npy bytes → NumPy array
   - `ensure_shape()` - Validate array shape
   - `validate_tensor()` - Check for NaN, Inf, empty, 3D+
   - `tensor_from_list()` - Python list → NumPy array
   - `tensor_to_list()` - NumPy array → Python list
   - `estimate_blob_size()` - Calculate storage size

### Testing
- **`tests/test_serialization.py`**
- Round-trip tests (encode → decode)
- Shape validation tests
- NaN/Inf detection tests
- 1D and 2D array tests
- Dtype conversion tests

### Success Criteria
- [x] Round-trip preserves data exactly
- [x] Invalid tensors caught early
- [x] Works with both 1D and 2D arrays
- [x] Efficient serialization (no unnecessary copies)

### Status: ✅ COMPLETED (32 tests passing)

---

## Phase 5: Schema Registry 📐

**Duration:** 1-2 days
**Dependencies:** Phase 3, 4
**Priority:** HIGH - Connects storage and validation

### Files to Implement

1. **`hitl/schemas/registry.py`**
   - `SchemaRegistry` class:
     - `ensure()` - Create or retrieve schema
     - `get()` - Retrieve schema by ID
     - `from_anomaly()` - Get schema from anomaly's vector
     - `list_all()` - List all schemas
     - `validate_shape()` - Check 1D or 2D only
     - `validate_dtype()` - Check valid NumPy dtype
     - `_compute_metadata()` - Calculate ndim, numel
     - `_row_to_schema_info()` - Convert DB row to TypedDict
   - Helper functions:
     - `shape_to_str()` - Serialize shape for DB
     - `str_to_shape()` - Deserialize shape from DB

### Testing
- **`tests/test_registry.py`**
- Schema creation and retrieval
- Idempotent behavior (same input → same schema_id)
- Shape validation (accept 1D/2D, reject 3D+)
- Dtype validation

### Success Criteria
- [x] Same shape+dtype always produces same schema_id
- [x] Schemas persisted and retrievable
- [x] Validation catches invalid shapes early
- [x] Works with both dense and conv1d modes

### Status: ✅ COMPLETED (32 tests passing)

---

## Phase 6: Artifacts Manager 📦

**Duration:** 1-2 days
**Dependencies:** Phase 1
**Priority:** MEDIUM - Can develop in parallel with Phase 3-5

### Files to Implement

1. **`hitl/artifacts/manager.py`**
   - `Artifacts` class:
     - `create_version()` - Generate model version, create directory
     - `save_model()` - Save PyTorch state_dict
     - `save_config()` - Save model config JSON
     - `save_scaler()` - Save normalization params JSON
     - `save_threshold()` - Save threshold JSON
     - `load_all()` - Load complete artifact set
     - `load_model()`, `load_config()`, `load_scaler()`, `load_threshold()`
     - `exists()` - Check if artifacts complete
     - `list_versions()` - List all model versions
     - `delete()` - Remove artifacts
     - `get_path()` - Get artifact directory path
   - Helper functions:
     - `_find_next_sequence()` - Increment sequence number for date

### Testing
- **`tests/test_artifacts.py`**
- Version creation and sequencing
- Save and load round-trips
- Missing file detection
- Directory listing

### Success Criteria
- [x] Artifacts saved with correct structure
- [x] Loading retrieves all components
- [x] Version numbering works correctly
- [x] Handles missing files gracefully

### Status: ✅ COMPLETED (27 tests passing)

---

## Phase 7: Model Architectures 🧠

**Duration:** 2-3 days
**Dependencies:** Phase 1 (minimal)
**Priority:** MEDIUM - Can develop early

### Files to Implement

1. **`hitl/models/ae.py`**
   - `DenseAE` class:
     - `__init__()` - Build encoder/decoder layers
     - `forward()` - Full forward pass
     - `encode()` - Get latent representation
     - `decode()` - Reconstruct from latent

   - `Conv1dAE` class:
     - `__init__()` - Build conv/deconv layers
     - `forward()` - Full forward pass (B, F, T)
     - `encode()` - Get latent representation
     - `decode()` - Reconstruct from latent

   - Factory function:
     - `build_model()` - Create model based on mode and shape

   - Helper functions:
     - `count_parameters()` - Count trainable params
     - `init_weights()` - Initialize with Xavier/Kaiming
     - `model_summary()` - Generate summary string

### Testing
- **`tests/test_models.py`**
- Model initialization
- Forward pass shape preservation
- Encode/decode methods
- Parameter counting
- Both dense and conv1d modes

### Success Criteria
- [x] DenseAE works with 1D inputs
- [x] Conv1dAE works with 2D inputs
- [x] Forward pass preserves shape exactly
- [x] Models can be saved and loaded
- [x] Factory function selects correct architecture

### Status: ✅ COMPLETED (35 tests passing)

---

## Phase 8: Training Pipeline 🎓

**Duration:** 3-4 days
**Dependencies:** Phase 3, 4, 5, 6, 7
**Priority:** CRITICAL - Core ML functionality

### Files to Implement

1. **`hitl/training/trainer.py`**
   - `Trainer` class:
     - `load_dataset()` - Load vectors from DB by schema
     - `fit()` - Complete training loop
     - `train_and_publish()` - Train + save + register
     - `_split_data()` - Train/val split
     - `_compute_scaler()` - Calculate mean/std
     - `_normalize()` - Apply normalization
     - `_train_epoch()` - One epoch of training
     - `_validate()` - Validation pass
     - `_compute_threshold()` - Calculate anomaly threshold
     - `_create_dataloader()` - PyTorch DataLoader

   - Helper functions:
     - `mse_per_sample()` - Per-sample MSE
     - `get_device()` - Get CUDA or CPU

### Testing
- **`tests/test_trainer.py`**
- Dataset loading
- Scaler computation
- Normalization
- Training loop (small dataset)
- Threshold computation
- Early stopping
- Complete workflow

### Success Criteria
- [ ] Can load vectors from database
- [ ] Training reduces loss
- [ ] Model converges on toy dataset
- [ ] Artifacts saved correctly
- [ ] Threshold computed at specified percentile
- [ ] Early stopping triggers when needed

---

## Phase 9: Inference Service 🔮

**Duration:** 2 days
**Dependencies:** Phase 3, 4, 6, 7
**Priority:** HIGH - Required for predictions

### Files to Implement

1. **`hitl/inference/serve.py`**
   - `LiveModel` class:
     - `load_live()` - Load and cache model
     - `predict_tensor()` - Score input tensor
     - `reload_if_changed()` - Detect model updates
     - `clear_cache()` - Clear cached model
     - `get_info()` - Return model metadata
     - `_normalize()` - Apply scaler to input
     - `_compute_reconstruction_error()` - Calculate MSE

   - Helper functions:
     - `mse_per_sample()` - Per-sample MSE
     - `get_device()` - Get device
     - `validate_input_shape()` - Validate and batch

### Testing
- **`tests/test_infer.py`**
- Model loading and caching
- Prediction accuracy
- Threshold comparison
- Shape validation
- Reload detection

### Success Criteria
- [ ] Model loaded once and cached
- [ ] Predictions return correct format
- [ ] Label=1 when score > threshold
- [ ] Label=0 when score <= threshold
- [ ] Handles shape mismatches gracefully
- [ ] Detects live model changes

---

## Phase 10: HITL Orchestrator 🎯

**Duration:** 3-4 days
**Dependencies:** Phase 3, 4, 5, 6, 7, 8, 9
**Priority:** CRITICAL - Main public API

### Files to Implement

1. **`hitl/core/hitl.py`**
   - `HITL` class with public methods:
     - **Anomaly Management:**
       - `upsert_anomaly()` - Insert/update with tensor
       - `get_anomaly()` - Retrieve metadata
       - `get_anomaly_with_tensor()` - Retrieve with vector

     - **Feedback:**
       - `submit_feedback()` - Submit human label
       - `get_feedback()` - Get feedback history

     - **Training:**
       - `train_model()` - Train new model
       - `list_models()` - List trained models

     - **Model Management:**
       - `set_live_model()` - Activate model
       - `get_live_model()` - Get current live version

     - **Inference:**
       - `filter_predict()` - Predict single
       - `batch_predict()` - Predict multiple

     - **Utility:**
       - `get_stats()` - System statistics
       - `health_check()` - System health
       - `close()` - Cleanup

   - Helper functions:
     - `_validate_anomaly_dict()` - Validate required fields
     - `_merge_train_params()` - Merge with defaults

### Testing
- **`tests/test_hitl.py`** - Integration tests
- All public methods
- Complete workflows
- Error handling
- Edge cases

### Success Criteria
- [ ] All public methods implemented
- [ ] Anomaly ingestion works
- [ ] Training completes successfully
- [ ] Predictions work end-to-end
- [ ] Error handling is robust
- [ ] System stats are accurate

---

## Phase 11: CLI Interface 💻

**Duration:** 2-3 days
**Dependencies:** Phase 10
**Priority:** HIGH - User-facing

### Files to Implement

1. **`hitl/cli/main.py`**
   - Typer app with commands:
     - `init-db` - Initialize database
     - `upsert` - Add/update anomaly from .npy file
     - `feedback` - Submit feedback
     - `train` - Train model
     - `set-live` - Set live model
     - `predict` - Predict from .npy or anomaly_id
     - `list-anomalies` - List anomalies
     - `list-models` - List models
     - `stats` - Show statistics
     - `health` - Check system health
     - `get-anomaly` - Get anomaly details
     - `export-vector` - Export vector to .npy

   - Helper functions:
     - `load_npy()` - Load from file
     - `save_npy()` - Save to file
     - `print_json()` - Pretty JSON output
     - `print_table()` - Tabular output
     - `handle_error()` - User-friendly errors

### Testing
- **`tests/test_cli.py`**
- All commands
- Argument parsing
- File I/O
- Output formatting
- Error messages

### Success Criteria
- [ ] All commands work
- [ ] .npy file I/O works
- [ ] Output is readable
- [ ] Error messages are helpful
- [ ] `--help` is comprehensive

---

## Phase 12: HTTP API 🌐

**Duration:** 2-3 days
**Dependencies:** Phase 10
**Priority:** HIGH - External integration

### Files to Implement

1. **`hitl/api/schemas.py`**
   - Pydantic models (already defined in types.py)
   - Additional response models
   - Error response model

2. **`hitl/api/server.py`**
   - FastAPI app with endpoints:
     - `GET /health` - Health check
     - `GET /stats` - Statistics
     - `POST /anomalies` - Upsert anomaly
     - `GET /anomalies/{id}` - Get anomaly
     - `POST /feedback` - Submit feedback
     - `POST /train` - Train model
     - `GET /models` - List models
     - `POST /models/live` - Set live model
     - `GET /models/live` - Get live model
     - `POST /predict` - Predict

   - Error handlers for all custom exceptions
   - Startup/shutdown hooks
   - HITL dependency injection

### Testing
- **`tests/test_api.py`**
- All endpoints
- Request validation
- Response serialization
- Error handling
- Integration workflow

### Success Criteria
- [ ] All endpoints implemented
- [ ] Request validation works
- [ ] Responses are JSON serializable
- [ ] Error handlers return proper status codes
- [ ] Can be started with uvicorn
- [ ] OpenAPI docs are complete

---

## Phase 13: Testing & Documentation ✅

**Duration:** 2-3 days
**Dependencies:** ALL phases
**Priority:** CRITICAL - Quality assurance

### Tasks

1. **Complete Test Fixtures**
   - Implement all fixtures in `tests/conftest.py`
   - Temporary database fixtures
   - Sample data fixtures
   - Trained model fixtures

2. **Run Full Test Suite**
   ```bash
   pytest --cov=hitl --cov-report=html
   ```
   - Target: >90% coverage
   - Fix all failing tests
   - Add missing tests

3. **Update Documentation**
   - Complete `README.md`:
     - Installation instructions with `uv`
     - Quick start guide
     - CLI usage examples
     - Python API examples
     - HTTP API examples
   - Add docstrings to public methods
   - Architecture overview diagram
   - Contribution guidelines

4. **Code Quality**
   - Run `mypy --strict` and fix type issues
   - Run `ruff` linter and fix issues
   - Format with `black` or `ruff format`

### Success Criteria
- [ ] >90% test coverage
- [ ] All tests pass
- [ ] README is comprehensive
- [ ] Code passes type checking
- [ ] Code passes linting
- [ ] API docs generated

---

## Phase 14: End-to-End Validation 🚀

**Duration:** 2-3 days
**Dependencies:** ALL phases
**Priority:** CRITICAL - Production readiness

### Validation Tests

1. **CLI Workflow**
   ```bash
   # Initialize
   hitl init-db

   # Ingest data
   hitl upsert --id A1 --source unit-1 --when 2025-11-04T10:00:00Z --npy data.npy
   hitl upsert --id A2 --source unit-1 --when 2025-11-04T10:05:00Z --npy data2.npy

   # Submit feedback
   hitl feedback --anomaly A1 --user analyst-1 --label TP --confidence 0.9

   # Train model
   hitl train --mode dense --epochs 50

   # Set live model
   hitl set-live --model AE-2025.11.04-1

   # Predict
   hitl predict --anomaly A1
   hitl predict --npy new_data.npy

   # Statistics
   hitl stats
   hitl health
   ```

2. **Python API Workflow**
   ```python
   from hitl import HITL
   import numpy as np

   # Initialize
   hitl = HITL()

   # Ingest
   for i in range(100):
       tensor = np.random.randn(128)
       hitl.upsert_anomaly(
           anomaly={"anomaly_id": f"A{i}", ...},
           tensor=tensor
       )

   # Feedback
   hitl.submit_feedback("A0", label="TP", user_id="user1")

   # Train
   model_version = hitl.train_model(mode="dense")

   # Deploy
   hitl.set_live_model(model_version)

   # Predict
   result = hitl.filter_predict(tensor=new_data)
   print(f"Anomaly: {result['label']}, Score: {result['score']}")

   # Stats
   stats = hitl.get_stats()
   ```

3. **HTTP API Workflow**
   ```bash
   # Start server
   uvicorn hitl.api.server:app --reload

   # Health check
   curl http://localhost:8000/health

   # Upsert anomaly
   curl -X POST http://localhost:8000/anomalies \
     -H "Content-Type: application/json" \
     -d '{"anomaly_id":"A1","occurred_at":"2025-11-04T10:00:00Z","source":"unit-1","tensor":[...]}'

   # Train
   curl -X POST http://localhost:8000/train \
     -H "Content-Type: application/json" \
     -d '{"mode":"dense","params":{"epochs":50}}'

   # Predict
   curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"tensor":[...]}'
   ```

4. **Performance Testing**
   - Load 1000+ anomalies
   - Train on realistic dataset size
   - Measure inference latency (<100ms target)
   - Check database size growth
   - Memory usage during training
   - Concurrent prediction throughput

5. **Integration Testing**
   - Database recovery after crash
   - Model hotswap (live model update)
   - Multiple schemas handling
   - Large tensor handling
   - Edge cases (empty DB, no feedback, etc.)

### Success Criteria
- [ ] Complete workflow works via CLI
- [ ] Complete workflow works via Python API
- [ ] Complete workflow works via HTTP API
- [ ] Performance meets targets
- [ ] System handles edge cases
- [ ] No memory leaks
- [ ] Database remains consistent

---

## Development Schedule

### Week 1: Core Foundation
- **Days 1-2:** Phases 1-2 (Utilities, Types)
- **Days 3-4:** Phase 3 (Database)
- **Day 5:** Phase 4 (I/O)

### Week 2: Business Logic
- **Days 1-2:** Phases 5-6 (Schema Registry, Artifacts)
- **Days 3-4:** Phase 7 (Models)
- **Day 5:** Start Phase 8 (Training)

### Week 3: ML Pipeline
- **Days 1-3:** Complete Phase 8 (Training)
- **Days 4-5:** Phase 9 (Inference)

### Week 4: Integration
- **Days 1-2:** Phase 10 (HITL Orchestrator)
- **Days 3-4:** Phases 11-12 (CLI & API)
- **Day 5:** Phase 13 (Testing)

### Week 5: Polish & Launch
- **Days 1-2:** Phase 14 (E2E Validation)
- **Days 3-5:** Bug fixes, documentation, demos

---

## Milestones

| Milestone | Completion | Description |
|-----------|-----------|-------------|
| **M1: Storage** | End Week 1 | Database can store/retrieve anomalies and vectors |
| **M2: Training** | End Week 3 | Can train model on stored data |
| **M3: Inference** | End Week 3 | Can make predictions with trained model |
| **M4: Core API** | End Week 4 | Complete Python API works |
| **M5: Interfaces** | End Week 4 | CLI and HTTP API functional |
| **M6: Production** | End Week 5 | Production-ready system |

---

## Parallel Development Opportunities

The following can be worked on simultaneously by different developers:

- **Track A:** Phases 3, 4, 5 (Database path)
- **Track B:** Phases 6, 7 (ML path - can start early)
- **Track C:** Phases 11, 12 (After Phase 10 - UI paths)

---

## Risk Management

### High-Risk Areas
1. **Training Pipeline (Phase 8)** - Complex ML logic, performance critical
2. **Database Concurrency (Phase 3)** - SQLite limitations with multiple writers
3. **Model Convergence (Phase 7-8)** - May need hyperparameter tuning

### Mitigation Strategies
- Start with toy datasets for training tests
- Use WAL mode and single-writer pattern for SQLite
- Test models on synthetic data first
- Have fallback architectures ready

---

## Dependencies & Tools

### Core Dependencies
- Python 3.11+
- numpy >= 1.24
- torch >= 2.0 (PyTorch)
- pydantic >= 2.7
- structlog >= 24.1
- fastapi >= 0.115.0
- uvicorn >= 0.30.0
- typer (for CLI)

### Dev Dependencies
- pytest >= 8.0
- pytest-cov >= 5.0
- ruff >= 0.6
- mypy >= 1.10
- httpx >= 0.27 (for API tests)

### Build Tool
- uv (package management and build)

---

## Success Metrics

### Code Quality
- [ ] >90% test coverage
- [ ] Zero type errors with mypy --strict
- [ ] Zero linter errors with ruff
- [ ] All public APIs documented

### Performance
- [ ] Inference latency <100ms
- [ ] Training completes in reasonable time
- [ ] Database scales to 10k+ anomalies
- [ ] Memory usage <2GB during training

### Functionality
- [ ] All 14 phases completed
- [ ] All CLI commands work
- [ ] All API endpoints work
- [ ] Complete workflow tested

---

## Next Steps

1. **Review this plan** with team/stakeholders
2. **Set up development environment:**
   ```bash
   cd human_in_the_loop
   uv venv
   uv pip install -e .[torch]
   ```
3. **Start Phase 1** - Foundation & Utilities
4. **Follow the plan** phase by phase
5. **Update this document** as you progress

---

## Notes

- This is an MVP. Future enhancements can include:
  - Multi-model support (ensemble)
  - Advanced thresholding strategies
  - GPU optimization
  - Distributed training
  - Real-time streaming
  - Web UI dashboard

- Keep scope focused on MVP for first iteration
- Defer optimizations until after working prototype
- Gather user feedback early and iterate

---

**Last Updated:** November 4, 2025
**Status:** Ready to Start Implementation
