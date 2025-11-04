# Tests for Inference Service
#
# Test coverage:
# - LiveModel loading
# - Caching behavior
# - Prediction on tensors
# - Normalization during inference
# - Threshold comparison
# - Reload detection
# - Error cases (no live model, shape mismatch)
#
# Test functions to implement:
#
# def test_load_live_loads_model(live_model, trained_model_artifacts, repository):
#   """Test live model loading."""
#
# def test_load_live_caches_model(live_model, trained_model_artifacts, repository):
#   """Test model is cached after first load."""
#
# def test_load_live_raises_if_not_set(live_model):
#   """Test NoLiveModel raised when not set."""
#
# def test_predict_tensor_returns_result(live_model, trained_model_artifacts, sample_vector_1d):
#   """Test prediction on valid tensor."""
#
# def test_predict_tensor_labels_anomaly_if_above_threshold(live_model):
#   """Test label=1 when score > threshold."""
#
# def test_predict_tensor_labels_normal_if_below_threshold(live_model):
#   """Test label=0 when score <= threshold."""
#
# def test_predict_tensor_raises_shape_mismatch(live_model, trained_model_artifacts):
#   """Test ShapeMismatch for wrong shape."""
#
# def test_reload_if_changed_detects_change(live_model, repository, artifacts):
#   """Test reload when live model version changes."""
#
# def test_reload_if_changed_skips_if_same(live_model, trained_model_artifacts, repository):
#   """Test no reload when version unchanged."""
#
# def test_clear_cache_removes_model(live_model, trained_model_artifacts):
#   """Test cache clearing."""
#
# def test_get_info_returns_metadata(live_model, trained_model_artifacts):
#   """Test model info retrieval."""
#
# def test_normalize_applies_scaler(live_model):
#   """Test normalization with scaler."""
#
# def test_mse_per_sample_dense():
#   """Test per-sample MSE for dense mode."""
#
# def test_mse_per_sample_conv1d():
#   """Test per-sample MSE for conv1d mode."""
