# ARCHITECTURE.md — HITL v0.4T‑MD MVP

**Target**: Production‑ready minimal system for Human‑in‑the‑Loop anomaly filtering using a PyTorch Autoencoder with tensor storage in SQLite and local artifacts.
**Packaging/Build**: `uv` + `pyproject.toml`.
**APIs**: CLI and optional FastAPI HTTP server.
**Tensor support**: 1D `(D,)` or 2D `(T,F)` via one active AE mode (`dense` or `conv1d`).

---

## 0. High‑Level Design

**Data path**
1. Upsert anomaly + tensor → SQLite (`anomalies`, `feature_schemas`, `raw_vectors`).
2. Train AE from `raw_vectors` for a schema → save artifacts → set live pointer.
3. Inference: load live model + threshold → score new tensor or stored anomaly (no preprocessing).

**Key constraints**
- SQLite with WAL. Single writer, many readers.
- Tensors persisted as `.npy` BLOBs with native shape and dtype.
- One live model at a time (MVP).

---

## 1. Repository Layout

```
hitl/
├─ pyproject.toml
├─ README.md
├─ ARCHITECTURE.md
├─ hitl/                       # package
│  ├─ __init__.py
│  ├─ settings.py              # global config and paths
│  ├─ errors.py                # typed exceptions
│  ├─ ddl.sql                  # minimal DDL schema
│  ├─ types.py                 # Protocols, TypedDicts, pydantic models
│  ├─ utils/
│  │   ├─ time.py              # utcnow(), isoformat helpers
│  │   ├─ ids.py               # uuid4(), sha1 helpers
│  │   └─ logging.py           # structlog/logger factory
│  ├─ store/
│  │   ├─ __init__.py
│  │   ├─ sqlite.py            # SQLite connection + CRUD
│  │   └─ repository.py        # Repo layer over sqlite.py
│  ├─ schemas/
│  │   └─ registry.py          # feature schema registry (shape, dtype)
│  ├─ io/
│  │   └─ serialization.py     # npy encode/decode + validation
│  ├─ artifacts/
│  │   └─ manager.py           # artifact IO (model.pt, threshold.json, config.json)
│  ├─ models/
│  │   └─ ae.py                # AE architectures: DenseAE or Conv1dAE
│  ├─ training/
│  │   └─ trainer.py           # training loop + early stop + thresholding
│  ├─ inference/
│  │   └─ serve.py             # LiveModel cache + scoring
│  ├─ core/
│  │   └─ hitl.py              # Orchestrator class exposing public API
│  ├─ api/
│  │   ├─ __init__.py
│  │   ├─ schemas.py           # pydantic request/response models
│  │   └─ server.py            # FastAPI app (optional)
│  └─ cli/
│      └─ main.py              # CLI using argparse or typer
├─ artifacts/                  # runtime (gitignored)
├─ data/                       # optional seed data (gitignored)
└─ tests/
   ├─ conftest.py
   ├─ test_store.py
   ├─ test_registry.py
   ├─ test_artifacts.py
   ├─ test_trainer.py
   ├─ test_infer.py
   └─ test_api.py
```

---

## 2. `pyproject.toml` (uv)

```toml
[project]
name = "hitl"
version = "0.1.0"
description = "HITL anomaly filtering with tensor storage in SQLite and PyTorch AE"
readme = "README.md"
requires-python = ">=3.11"
authors = [{ name = "Your Name", email = "you@example.com" }]
dependencies = [
  "numpy>=1.24",
  "pydantic>=2.7",
  "structlog>=24.1",
  "fastapi>=0.115.0",
  "uvicorn>=0.30.0",
]

[project.optional-dependencies]
torch = ["torch>=2.0"]

[project.scripts]
hitl = "hitl.cli.main:app"   # if using Typer; or "hitl.cli.main:main" for argparse

[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[tool.uv]
dev-dependencies = [
  "pytest>=8.0",
  "pytest-cov>=5.0",
  "ruff>=0.6",
  "mypy>=1.10",
  "types-requests",
  "httpx>=0.27"
]
```

> With `uv`: `uv venv`, `uv pip install -e .[torch]` or `uv sync -E torch`.

---

## 3. Database Schema (`hitl/ddl.sql`)

Matches the MVP DDL:

```sql
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS anomalies (
  anomaly_id TEXT PRIMARY KEY,
  occurred_at TEXT NOT NULL,
  source TEXT NOT NULL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS feedback (
  feedback_id TEXT PRIMARY KEY,
  anomaly_id TEXT NOT NULL,
  user_id TEXT NOT NULL,
  label TEXT NOT NULL,
  confidence REAL,
  note TEXT,
  created_at TEXT NOT NULL,
  FOREIGN KEY(anomaly_id) REFERENCES anomalies(anomaly_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS feature_schemas (
  schema_id TEXT PRIMARY KEY,
  shape TEXT NOT NULL,
  ndim INTEGER NOT NULL,
  numel INTEGER NOT NULL,
  dtype TEXT NOT NULL DEFAULT 'float32',
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS raw_vectors (
  anomaly_id TEXT PRIMARY KEY,
  schema_id TEXT NOT NULL,
  tensor_blob BLOB NOT NULL,
  created_at TEXT NOT NULL,
  FOREIGN KEY(anomaly_id) REFERENCES anomalies(anomaly_id) ON DELETE CASCADE,
  FOREIGN KEY(schema_id) REFERENCES feature_schemas(schema_id)
);

CREATE TABLE IF NOT EXISTS models (
  model_version TEXT PRIMARY KEY,
  kind TEXT NOT NULL,
  artifact_path TEXT NOT NULL,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS settings (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_fb_anom_time ON feedback(anomaly_id, created_at DESC);
```

---

## 4. Core Types (`hitl/types.py`)

- `SchemaInfo(TypedDict)`: `{"schema_id": str, "shape": tuple[int,...], "ndim": int, "numel": int, "dtype": str}`
- `TrainParams(TypedDict)`: `{"mode": Literal["dense","conv1d"], "epochs": int, "batch_size": int, "lr": float, "val_split": float, "patience": int, "percentile": float}`
- `PredictResult(TypedDict)`: `{"label": int, "score": float, "threshold": float, "model_version": str}`
- Pydantic models for API: `AnomalyUpsert`, `FeedbackIn`, `PredictIn`, `PredictOut`.

---

## 5. Errors (`hitl/errors.py`)

- `UnsupportedShape(Exception)`
- `ShapeMismatch(Exception)`
- `SchemaNotFound(Exception)`
- `NoLiveModel(Exception)`
- `ArtifactMissing(Exception)`
- `DBError(Exception)`

---

## 6. Settings (`hitl/settings.py`)

- `Config` dataclass:
  - `sqlite_path: str = "hitl.db"`
  - `artifacts_dir: str = "artifacts"`
  - `mode: Literal["dense","conv1d"] = "dense"`
- `get_env_config()` reads env overrides.
- `paths()` helper returns resolved paths.

---

## 7. Utilities

### `utils/time.py`
- `utcnow() -> datetime`
- `now_iso() -> str`
- `parse_iso(s: str) -> datetime`

### `utils/ids.py`
- `uuid_str() -> str`
- `sha1_bytes(b: bytes) -> str`
- `schema_id(shape: tuple[int,...], dtype: str) -> str`

### `utils/logging.py`
- `get_logger(name: str)` returning a structlog logger.

---

## 8. Storage Layer

### `store/sqlite.py`
- `class SQLite`
  - `__init__(self, path: str)` → ensures file and applies DDL.
  - `connect(self) -> sqlite3.Connection` with `row_factory=sqlite3.Row`.
  - `execute(self, sql: str, params: tuple=()) -> None`
  - `fetchone(self, sql: str, params: tuple=()) -> Row|None`
  - `fetchall(self, sql: str, params: tuple=()) -> list[Row]`
  - Transaction context manager `tx()`.

### `store/repository.py`
- `class Repository` using `SQLite`
  - **Anomalies**
    - `upsert_anomaly(anomaly_id, occurred_at, source, created_at, updated_at)`
    - `get_anomaly(anomaly_id) -> Row|None`
  - **Feedback**
    - `insert_feedback(feedback_id, anomaly_id, user_id, label, confidence, note, created_at)`
    - `latest_feedback(anomaly_id) -> Row|None`
  - **Schemas**
    - `get_schema(schema_id) -> Row|None`
    - `insert_schema(schema_id, shape, ndim, numel, dtype, created_at)`
  - **Raw vectors**
    - `put_vector(anomaly_id, schema_id, blob, created_at)`
    - `get_vector(anomaly_id) -> (schema_id:str, blob:bytes)`
    - `iter_vectors(schema_id) -> Iterator[(anomaly_id, blob)]`
  - **Models & settings**
    - `insert_model(model_version, kind, artifact_path, created_at)`
    - `set_live_model(model_version)`  → `settings["live_model_version"]=...`
    - `get_live_model() -> str|None`

---

## 9. Schema Registry (`schemas/registry.py`)

- `class SchemaRegistry`
  - `ensure(shape: tuple[int,...], dtype: str) -> SchemaInfo`
    Computes `schema_id` from shape+dtype, inserts if missing.
  - `from_anomaly(anomaly_id) -> SchemaInfo`
    Looks up raw vector to infer shape and dtype.
  - Validates `(D,)` or `(T,F)` only for MVP. Raises `UnsupportedShape` otherwise.

**Input:** `shape`, `dtype`.
**Output:** `SchemaInfo` with `schema_id` and metadata.

---

## 10. IO Serialization (`io/serialization.py`)

- `encode_npy(arr: np.ndarray, dtype="float32") -> bytes`
  Validates contiguous 1D/2D, casts dtype, returns `.npy` bytes.
- `decode_npy(blob: bytes) -> np.ndarray`
- `ensure_shape(arr: np.ndarray, shape: tuple[int,...]) -> np.ndarray`
  Raises `ShapeMismatch` if not exact.

---

## 11. Artifacts Manager (`artifacts/manager.py`)

- `class Artifacts`
  - `create_version(mode: str, input_shape: tuple[int,...]) -> str` → returns `model_version` like `AE-YYYY.MM.DD-N` and artifact directory path.
  - `save_model(state_dict: dict, path: str) -> None` → writes `model.pt`
  - `save_config(path: str, config: dict) -> None`
  - `save_threshold(path: str, threshold: dict) -> None`
  - `load_all(path: str) -> dict` → returns `{"model": state_dict, "config":..., "threshold":...}`

**Input:** state dict, config, threshold.
**Output:** persisted files and paths.

---

## 12. Models (`models/ae.py`)

- `class DenseAE(nn.Module)`  — for `(D,)`
  - `encoder`: Linear/ReLU stacks to latent
  - `decoder`: Linear/ReLU mirror to `D`
  - `forward(x: Tensor[B,D]) -> Tensor[B,D]`

- `class Conv1dAE(nn.Module)` — for `(T,F)` with channels=`F`
  - Accepts `(B,F,T)`; encoder uses `Conv1d`, decoder `ConvTranspose1d`
  - `forward(x: Tensor[B,F,T]) -> Tensor[B,F,T]`

- `build_model(mode: Literal["dense","conv1d"], input_shape: tuple[int,...]) -> nn.Module`

**Input:** mode and input shape.
**Output:** initialized AE model instance.

---

## 13. Training (`training/trainer.py`)

- `class Trainer`
  - `__init__(repo: Repository, artifacts: Artifacts, cfg: Config, logger)`
  - `load_dataset(schema_id: str, mode: str) -> tuple[np.ndarray, list[str]]`
    Returns `X` and `ids` in consistent shape `(N,D)` for dense or `(N,F,T)` for conv1d.
  - `fit(X, params: TrainParams) -> tuple[state_dict, threshold, metrics]`
    - Split 90/10.
    - Train with MSE + Adam(lr) on raw data (no preprocessing/normalization).
    - Early stop with patience on val loss.
    - Threshold: 99.5th percentile of train reconstruction MSE.
  - `train_and_publish(schema_id: str, params: TrainParams) -> str`
    - Create artifact version.
    - Save artifacts (model, config, threshold only).
    - Insert model row and return `model_version`.

**Inputs:** `schema_id`, training params.
**Output:** `model_version` string.

**Note:** Training is performed directly on raw vectors from database with no preprocessing.

---

## 14. Inference (`inference/serve.py`)

- `class LiveModel`
  - `__init__(repo: Repository, artifacts: Artifacts, cfg: Config, logger)`
  - `load_live() -> None` loads live model once and caches.
  - `predict_tensor(arr: np.ndarray) -> PredictResult`
    - Validate shape
    - Forward pass (on raw data, no preprocessing)
    - Compute per‑sample MSE → scalar `score`
    - Compare to threshold

- `mse_per_sample(x, x_hat) -> np.ndarray[(N,)]` implemented per mode.

**Input:** tensor (raw, no preprocessing).
**Output:** `PredictResult` dict.

---

## 15. Orchestrator (`core/hitl.py`)

- `class HITL`
  - `upsert_anomaly(anomaly: dict, tensor, dtype="float32") -> str`
    - Insert/Update `anomalies` row
    - Ensure schema via `SchemaRegistry`
    - Encode `.npy` and write to `raw_vectors`
  - `submit_feedback(anomaly_id: str, label: str, user_id: str, confidence: float|None=None, note: str|None=None) -> None`
  - `train_model(mode: str, schema_id: str|None=None, params: dict|None=None) -> str`
    - Resolve schema if not provided (error if multiple exist)
    - Delegate to `Trainer.train_and_publish`
  - `set_live_model(model_version: str) -> None` → `Repository.set_live_model`
  - `filter_predict(tensor=None, anomaly_id: str|None=None) -> PredictResult`
    - If `tensor` given, use that; else load from DB by `anomaly_id`
    - Delegate to `LiveModel.predict_tensor`

**Inputs** and **Outputs** are identical to the MVP spec. Errors bubble as typed exceptions.

---

## 16. HTTP API (optional) (`api/server.py`, `api/schemas.py`)

- `POST /anomalies`
  **Body**: `AnomalyUpsert { anomaly_id:str, occurred_at:str, source:str, tensor: list|nested list|base64 npy }`
  **Out**: `{ anomaly_id }`
- `POST /feedback`
  **Body**: `FeedbackIn { anomaly_id, user_id, label, confidence?, note? }`
  **Out**: `{ ok: true }`
- `POST /train`
  **Body**: `{ mode: "dense"|"conv1d", schema_id?, params? }`
  **Out**: `{ model_version }`
- `POST /live/{model_version}`
  **Out**: `{ ok: true }`
- `POST /predict`
  **Body**: `PredictIn { tensor? , anomaly_id? }` exactly one is required
  **Out**: `PredictOut { label:int, score:float, threshold:float, model_version:str }`
- `GET /health`
  **Out**: `{ status:"ok" }`

Validation uses Pydantic. Server uses Uvicorn with `--workers 1` for MVP.

---

## 17. CLI (`cli/main.py`)

Commands (Typer or argparse):
- `hitl init-db` → applies DDL
- `hitl upsert --id A1 --source unit-1 --when 2025-11-04T10:00:00Z --npy path.npy`
- `hitl feedback --anomaly A1 --user u1 --label TP --confidence 0.9`
- `hitl train --mode dense` (or `conv1d`) → prints model_version
- `hitl set-live --model AE-2025.11.04-1`
- `hitl predict --npy path.npy` or `--anomaly A1`

**Inputs:** flags and `.npy` files.
**Outputs:** stdout JSON dicts.

---

## 18. Coding Guidelines

- Type hints and `mypy --strict` where feasible.
- Pure functions for math transforms. Minimal side effects.
- Small modules with single responsibility.
- Clear exception boundaries. Convert low‑level errors to `DBError` or domain errors.
- Unit tests for each module. Property tests for serialization round‑trip.
- Avoid global state. Inject `Repository`, `Artifacts`, and config explicitly.

---

## 19. Edge Cases

- Multiple schemas present but `schema_id` omitted in training → raise `SchemaNotFound("ambiguous")`.
- Tensor shape mismatch at insert or predict → `ShapeMismatch`.
- No live model at predict → `NoLiveModel`.
- Empty training set for schema → error with guidance.
- NaNs/inf in data → sanitize or raise with explicit message.
- SQLite busy → retry with backoff in `SQLite.tx()`.

---

## 20. Minimal Data Flows

### A. Ingest + Train + Serve
1. `HITL.upsert_anomaly({...}, tensor)` → rows in `anomalies`, `feature_schemas`, `raw_vectors`.
2. `HITL.train_model(mode="dense")` → artifacts written, `models` row created.
3. `HITL.set_live_model(mv)` → settings updated.
4. `HITL.filter_predict(anomaly_id="A1")` → returns `PredictResult`.

### B. Direct Predict
- `HITL.filter_predict(tensor=[...])` → immediate result using live model.

---

## 21. Tests (high level)

- `test_store.py`: DDL, CRUD, transactions.
- `test_registry.py`: schema id stability and shape validation.
- `test_artifacts.py`: save/load artifacts; version naming.
- `test_trainer.py`: overfit tiny dataset; threshold behavior.
- `test_infer.py`: shape validation, predict path; threshold crossing.
- `test_api.py`: contract tests for endpoints.

---

## 22. Non‑Goals (MVP)

- Multiple live models or per‑schema live pointers.
- `conv2d` image support.
- Run history and training metrics persistence.
- Distributed training or GPU orchestration.
- Advanced thresholding strategies.

---

## 23. Ready‑to‑Implement Checklist

- [ ] Implement `SQLite` and `Repository` with DDL bootstrap.
- [ ] Implement `SchemaRegistry` and `.ensure()`.
- [ ] Implement `.npy` `encode/decode` and shape guards.
- [ ] Implement `DenseAE` and `Conv1dAE`; `build_model()`.
- [ ] Implement `Trainer.train_and_publish()` with early stop + percentile threshold (no preprocessing).
- [ ] Implement `LiveModel` with in‑memory cache and reload on `set_live_model()`.
- [ ] Implement `HITL` façade.
- [ ] Wire CLI. Optionally wire FastAPI server.
- [ ] Add tests and CI workflow.
