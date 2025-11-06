# Human-in-the-Loop (HITL) — v0.4T‑MD **MVP**

**Status:** Draft
**Date:** 2025-11-04
**Goal:** Small, operational MVP for AE-based anomaly filtering with tensor storage in SQLite. Multi‑dimensional tensors supported, but we keep one AE mode active to cut surface area.

---

## 1) What’s included

- SQLite (`hitl.db`, WAL on) + artifacts on local FS.
- One active AE path: **choose one** mode and hardcode for MVP:
  - `dense` for 1D vectors `(D,)`, or
  - `conv1d` for time×feature `(T,F)` where features are channels.
- Tensors stored as `.npy` BLOBs in native shape and dtype.
- Single live model pointer in DB.
- Minimal API for insert, label, train, set live, and predict.

---

## 2) What’s removed

- Additional model kinds (`threshold`, `linear_sgd`), `conv2d` mode.
- Threshold strategies beyond `percentile=0.995`.
- Model lifecycle states (`staging/retired`), run history, manifests.
- Extra artifacts (`metrics.json`, `manifest.json`, `train_index.json`).
- Denormalized fields on `anomalies` (latest feedback cache).
- Nonessential indices and config flags.

---

## 3) Minimal schema (DDL)

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
  label TEXT NOT NULL,                 -- TP|FP|TN|FN|0|1
  confidence REAL,
  note TEXT,
  created_at TEXT NOT NULL,
  FOREIGN KEY(anomaly_id) REFERENCES anomalies(anomaly_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS feature_schemas (
  schema_id TEXT PRIMARY KEY,          -- sha1(JSON: {"shape":[...],"dtype":...})
  shape TEXT NOT NULL,                 -- JSON array, e.g. [240,84]
  ndim INTEGER NOT NULL,
  numel INTEGER NOT NULL,
  dtype TEXT NOT NULL DEFAULT 'float32',
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS raw_vectors (
  anomaly_id TEXT PRIMARY KEY,
  schema_id TEXT NOT NULL,
  tensor_blob BLOB NOT NULL,           -- .npy bytes
  created_at TEXT NOT NULL,
  FOREIGN KEY(anomaly_id) REFERENCES anomalies(anomaly_id) ON DELETE CASCADE,
  FOREIGN KEY(schema_id) REFERENCES feature_schemas(schema_id)
);

CREATE TABLE IF NOT EXISTS models (
  model_version TEXT PRIMARY KEY,      -- e.g. AE-2025.11.04-1
  kind TEXT NOT NULL,                  -- 'ae_torch'
  artifact_path TEXT NOT NULL,         -- ./artifacts/<model_version>
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS settings (   -- single live pointer
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_fb_anom_time ON feedback(anomaly_id, created_at DESC);
```

---

## 4) Artifacts

```
artifacts/
  AE-YYYY.MM.DD-1/
    model.pt
    config.json        # {"mode":"dense"|"conv1d","input_shape":[...],"dtype":"float32"}
    scaler.json        # {"kind":"per_feature","mean":[...],"std":[...],"eps":1e-8}
    threshold.json     # {"strategy":"percentile","value": <float>}
```

---

## 5) API surface (MVP)

```python
class HITL:
    def upsert_anomaly(self, anomaly: dict, tensor, dtype: str = "float32") -> str: ...
    def submit_feedback(self, anomaly_id: str, label: str, user_id: str,
                        confidence: float | None = None, note: str | None = None) -> None: ...

    def train_model(self,
                    mode: str = "dense",          # or "conv1d"
                    schema_id: str | None = None,
                    params: dict | None = None) -> str: ...
    def set_live_model(self, model_version: str) -> None: ...
    def live_model(self) -> dict: ...             # {"model_version","artifact_path"}

    def filter_predict(self, tensor=None, anomaly_id=None) -> dict: ...
```

**Label mapping for optional eval:** `TP,FN,1 → 1` and `FP,TN,0 → 0`.

---

## 6) Fixed defaults

- `mode`: pick one and hardcode for MVP (`dense` or `conv1d`).
- Scaler: `per_feature`.
- Loss/opt: `MSE` + `Adam(lr=1e-3)`.
- Early stopping: `patience=5` on val loss.
- Threshold: `percentile=0.995` on train reconstruction errors.
- Batch size: `128`. Epochs: `20`.

---

## 7) Training loop (minimal)

1. Identify `schema_id`:
   - `schema_id = sha1(json.dumps({"shape": shape, "dtype": dtype}).encode())`.
2. Load tensors with this `schema_id` from `raw_vectors` → `X` with shape `(N, *)`.
3. Split 90/10 train/val.
4. Fit scaler on train and transform both sets.
5. Build AE:
   - **dense**: MLP encoder/decoder (Linear + ReLU), final Linear to input size.
   - **conv1d**: Conv1d blocks with features as channels (input `(B,F,T)`), decoder via ConvTranspose1d, reshape to `(T,F)` if needed.
6. Train with early stopping on val MSE.
7. Compute train reconstruction MSE per sample. Set threshold at 99.5th percentile.
8. Save artifacts (`model.pt`, `config.json`, `scaler.json`, `threshold.json`) under a new `model_version`.
9. Insert one row into `models` and set `settings("live_model_version") = model_version` when promoted.

---

## 8) Inference (minimal)

1. Resolve `live_model_version` from `settings` and load cached artifacts.
2. Accept `tensor` or `anomaly_id`:
   - Read and decode `.npy` if from DB.
   - Validate exact shape against `config["input_shape"]`.
3. Standardize via `scaler.json`.
4. Forward pass → reconstruction MSE → `score`.
5. Decision: `label = int(score >= threshold)`.
6. Return:
```json
{"label": 0, "score": 0.013, "threshold": 0.042, "model_version": "AE-2025.11.04-1"}
```

---

## 9) Helper notes

- **Serialization:** save tensors with `np.save(BytesIO(), arr.astype(dtype))`; load with `np.load(BytesIO(blob), allow_pickle=False)`.
- **Axis convention for conv1d:** expect `(F,T)` on input, transpose to `(B,F,T)` before Conv1d; store `input_shape` exactly as provided.
- **Errors:** `UnsupportedShape`, `ShapeMismatch`, `NoLiveModel`.

---

## 10) Example

```python
# Vector example (dense)
hitl.upsert_anomaly(
  {"anomaly_id": "A1", "occurred_at": "...", "source": "unit-12",
   "created_at": "...", "updated_at": "..."},
  tensor=np.random.randn(84).astype("float32")
)
mv = hitl.train_model(mode="dense")
hitl.set_live_model(mv)
pred = hitl.filter_predict(tensor=np.random.randn(84).astype("float32"))

# Time-series example (conv1d)
X = np.random.randn(240, 84).astype("float32")  # (T,F)
hitl.upsert_anomaly({"anomaly_id": "A2", "occurred_at": "...", "source": "unit-34",
                     "created_at": "...", "updated_at": "..."},
                    tensor=X)
mv = hitl.train_model(mode="conv1d")
hitl.set_live_model(mv)
pred = hitl.filter_predict(anomaly_id="A2")
```

---

## 11) Dependencies

- numpy>=1.24, torch>=2.0
- stdlib: sqlite3, json, hashlib, dataclasses, uuid

Install:
```bash
pip install hitl[torch]
```
