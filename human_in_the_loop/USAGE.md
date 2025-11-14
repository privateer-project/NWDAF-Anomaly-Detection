# HITL — Usage Guide

This document explains how to install and use the `hitl` package (the
Human-in-the-Loop orchestrator) from this repository as a dependency in
another project. It includes installation options, configuration, and
concise code examples demonstrating common workflows.

## Overview

`hitl` exposes a single high-level orchestrator class `HITL` that wires the
database, schema registry, artifacts manager, trainer and live inference
service. Typical workflows are:

- Ingest / upsert anomaly vectors
- Query anomaly metadata and stored tensors
- Submit human feedback
- Train and publish models
- Set a live model and run inference

The main public API is available from the package root. Example:

```python
from hitl import HITL
from hitl.types import PredictResult
```

## Installation

Recommended: install into your consuming project's virtual environment.

- Editable (development) install from local repository:

```bash
cd /path/to/NWDAF-Anomaly-Detection/human_in_the_loop
pip install -e .
```

- With optional torch support (required for training/inference):

```bash
pip install -e .[torch]
```

- Install from Git (useful for CI or non-local installs):

```bash
pip install "git+https://github.com/<org>/<repo>.git@<branch>#subdirectory=human_in_the_loop"
```

## Configuration

Two ways to configure `HITL`:

1. Environment variables (used by default when calling `HITL()`):

   - `HITL_SQLITE_PATH` — path to SQLite DB file (default `./hitl.db`)
   - `HITL_ARTIFACTS_DIR` — artifacts directory (default `./artifacts`)
   - `HITL_MODE` — model mode: `dense` or `conv1d` (default `dense`)
   - `HITL_LOG_LEVEL` — logging level (default `INFO`)
   - `HITL_DEV_MODE` — enable dev logging (set to `1`/`true`/`yes`)

   Example:

   ```bash
   export HITL_SQLITE_PATH=/var/lib/hitl/hitl.db
   export HITL_ARTIFACTS_DIR=/var/lib/hitl/artifacts
   export HITL_MODE=dense
   ```

2. Programmatic `Config` object (recommended when embedding):

```python
from hitl.settings import Config, paths, validate_config
cfg = Config(sqlite_path="/tmp/hitl.db", artifacts_dir="/tmp/hit_artifacts", mode="dense")
validate_config(cfg)
cfg = paths(cfg)  # ensure directories exist and are resolved
from hitl import HITL
hitl = HITL(config=cfg)
```

Note: call `paths(cfg)` to create necessary directories when providing a
`Config` programmatically.

## Quickstart — Common Workflows

Below are short examples that cover the most common operations.

1) Initialize `HITL` (env-config or programmatic):

```python
from hitl import HITL
hitl = HITL()  # uses environment configuration by default
```

2) Upsert an anomaly with a NumPy tensor (automatically registers schema):

```python
import numpy as np
anomaly = {
    "anomaly_id": "A1",
    "occurred_at": "2025-11-13T12:00:00Z",
    "source": "sensor-1",
}
tensor = np.random.rand(128).astype("float32")  # dense example
hitl.upsert_anomaly(anomaly, tensor)
```

3) Retrieve anomaly metadata and stored tensor:

```python
record, arr = hitl.get_anomaly_with_tensor("A1")
print(record)
print(arr.shape)
```

4) Submit feedback (human label):

```python
fid = hitl.submit_feedback("A1", label="TP", user_id="analyst-1", confidence=0.95, note="Verified")
```

5) Train and publish a model (may require `torch`):

```python
model_version = hitl.train_model(mode="dense", params={"epochs": 50, "batch_size": 64})
hitl.set_live_model(model_version)
```

6) Run inference on a new tensor or an existing anomaly:

```python
res = hitl.filter_predict(tensor=tensor)  # or anomaly_id="A1"
print(res)  # dict-like PredictResult: label, score, threshold, model_version
```

7) Shutdown/cleanup

```python
hitl.close()
```

## Packaging & Dependency Notes

- Core dependencies are defined in `pyproject.toml` under `[project]`.
- Optional (training/inference) dependency: `torch` (install with the
  extra `torch` as shown above).
- The package targets Python 3.11+ (see `pyproject.toml`).

If you plan to consume `hitl` from another project's `pyproject.toml`, you
can reference it as a local file dependency during development:

```toml
[project]
dependencies = [
  "hitl @ file:///absolute/path/to/NWDAF-Anomaly-Detection/human_in_the_loop"
]
```

Or reference a Git URL with `#subdirectory=human_in_the_loop`.

## Testing & Development

- Run unit tests for `human_in_the_loop`:

```bash
cd human_in_the_loop
pytest -q
```

- Run tests with coverage:

```bash
pytest --cov=hitl --cov-report=term-missing
```

- Run type checks and linters (recommended):

```bash
mypy --strict hitl
ruff check --fix hitl
```

## Caveats & Recommendations

- Storage: the default storage backend is SQLite. For heavy concurrent
  ingestion consider a server-grade DB and adapt the `Repository` if
  necessary.
- Concurrency: avoid re-instantiating `HITL` per-request in web servers;
  create a singleton and reuse it.
- Artifacts: model files are stored in `artifacts_dir`. Ensure proper
  backup policies for production.

## Troubleshooting

- If model training or inference errors mention missing `torch`, install the
  optional dependency with `pip install -e .[torch]`.
- If you see schema or shape errors, validate that tensors are the expected
  storage shape: 1D vectors for `dense` mode, and `(T, F)` storage format
  for `conv1d` mode (trainer/data loader transposes to `(B, F, T)` for the
  model).

## Contributing

- Follow code style and run tests before submitting PRs.
- Update this `USAGE.md` if you add/modify API surface.

---

File: `human_in_the_loop/USAGE.md`
