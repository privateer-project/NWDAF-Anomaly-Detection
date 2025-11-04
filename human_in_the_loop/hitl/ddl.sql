-- HITL Database Schema Definition
-- 
-- This SQL file defines the complete database schema for the HITL system.
-- It is idempotent (safe to run multiple times).
--
-- Tables:
--   1. anomalies - Anomaly metadata
--   2. feedback - Human feedback on anomalies
--   3. feature_schemas - Tensor shape/dtype schemas
--   4. raw_vectors - Tensor data as .npy BLOBs
--   5. models - Trained model metadata
--   6. settings - Key-value configuration store

-- Enable WAL mode for concurrent reads
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

-- =============================================================================
-- 1. ANOMALIES TABLE
-- =============================================================================
CREATE TABLE IF NOT EXISTS anomalies (
    anomaly_id TEXT PRIMARY KEY,
    occurred_at TEXT NOT NULL,
    source TEXT NOT NULL,
    schema_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (schema_id) REFERENCES feature_schemas(schema_id)
);

CREATE INDEX IF NOT EXISTS idx_anomalies_occurred ON anomalies(occurred_at DESC);
CREATE INDEX IF NOT EXISTS idx_anomalies_source ON anomalies(source);
CREATE INDEX IF NOT EXISTS idx_anomalies_schema ON anomalies(schema_id);

-- =============================================================================
-- 2. FEEDBACK TABLE
-- =============================================================================
CREATE TABLE IF NOT EXISTS feedback (
    feedback_id TEXT PRIMARY KEY,
    anomaly_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    label TEXT NOT NULL,
    confidence REAL CHECK (confidence IS NULL OR (confidence >= 0.0 AND confidence <= 1.0)),
    note TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (anomaly_id) REFERENCES anomalies(anomaly_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_feedback_anomaly ON feedback(anomaly_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_feedback_user ON feedback(user_id);
CREATE INDEX IF NOT EXISTS idx_feedback_label ON feedback(label);

-- =============================================================================
-- 3. FEATURE_SCHEMAS TABLE
-- =============================================================================
CREATE TABLE IF NOT EXISTS feature_schemas (
    schema_id TEXT PRIMARY KEY,
    shape TEXT NOT NULL,
    ndim INTEGER NOT NULL CHECK (ndim > 0),
    numel INTEGER NOT NULL CHECK (numel > 0),
    dtype TEXT NOT NULL DEFAULT 'float32',
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_schemas_ndim ON feature_schemas(ndim);

-- =============================================================================
-- 4. RAW_VECTORS TABLE
-- =============================================================================
CREATE TABLE IF NOT EXISTS raw_vectors (
    anomaly_id TEXT PRIMARY KEY,
    schema_id TEXT NOT NULL,
    tensor_blob BLOB NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (anomaly_id) REFERENCES anomalies(anomaly_id) ON DELETE CASCADE,
    FOREIGN KEY (schema_id) REFERENCES feature_schemas(schema_id)
);

CREATE INDEX IF NOT EXISTS idx_vectors_schema ON raw_vectors(schema_id);

-- =============================================================================
-- 5. MODELS TABLE
-- =============================================================================
CREATE TABLE IF NOT EXISTS models (
    model_version TEXT PRIMARY KEY,
    kind TEXT NOT NULL CHECK (kind IN ('dense', 'conv1d')),
    schema_id TEXT NOT NULL,
    artifact_path TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (schema_id) REFERENCES feature_schemas(schema_id)
);

CREATE INDEX IF NOT EXISTS idx_models_kind ON models(kind);
CREATE INDEX IF NOT EXISTS idx_models_schema ON models(schema_id);
CREATE INDEX IF NOT EXISTS idx_models_created ON models(created_at DESC);

-- =============================================================================
-- 6. SETTINGS TABLE
-- =============================================================================
CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
