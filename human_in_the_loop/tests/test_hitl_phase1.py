import numpy as np
from pathlib import Path
from hitl.settings import Config
from hitl.core.hitl import HITL
from hitl.utils.time import now_iso
from hitl.models.ae import build_model, init_weights


def test_hitl_upsert_predict(tmp_path):
    artifacts_dir = tmp_path / "artifacts"
    db_path = tmp_path / "hitl.db"

    config = Config(sqlite_path=str(db_path), artifacts_dir=str(artifacts_dir), mode="conv1d")

    hitl = HITL(config=config)

    # Create a small conv1d sample (T, F)
    sample = np.random.randn(4, 2).astype(np.float32)

    aid = "test-1"
    anomaly = {"anomaly_id": aid, "occurred_at": now_iso(), "source": "unittest"}

    hitl.upsert_anomaly(anomaly, sample)

    got = hitl.get_anomaly(aid)
    assert got is not None
    assert got["anomaly_id"] == aid

    # Build a model compatible with (T=4, F=2)
    model = build_model("conv1d", (4, 2))
    init_weights(model)
    state = model.state_dict()

    # Create artifact version and save
    version, _ = hitl.artifacts.create_version("conv1d", (4, 2))
    hitl.artifacts.save_model(version, state)
    hitl.artifacts.save_config(version, {"mode": "conv1d", "input_shape": (4, 2)})
    # Use a very large threshold to classify as normal
    hitl.artifacts.save_threshold(version, {"value": 1e6, "percentile": 99.0})

    # Register model in repo and set live
    schema = hitl.registry.from_anomaly(aid)
    hitl.repo.insert_model(version, "conv1d", schema["schema_id"], version, now_iso())
    hitl.repo.set_live_model(version)

    # Predict on tensor
    res = hitl.filter_predict(tensor=sample)
    assert "label" in res and "score" in res
    assert res["label"] in (0, 1)

    # Predict on anomaly id
    res2 = hitl.filter_predict(anomaly_id=aid)
    assert res2["model_version"] == version

    hitl.close()
