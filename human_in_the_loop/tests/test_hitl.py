# Tests for HITL Orchestrator
#
# Test coverage:
# - Complete workflows (ingest → train → predict)
# - All public API methods
# - Error handling and validation
# - Integration of subsystems
#
# Test functions to implement:
#
# def test_hitl_initializes(hitl):
#   """Test HITL can be instantiated."""
#
# def test_upsert_anomaly_creates_record(hitl, sample_anomaly, sample_vector_1d):
#   """Test upserting anomaly with vector."""
#
# def test_upsert_anomaly_accepts_list(hitl, sample_anomaly):
#   """Test upserting with Python list tensor."""
#
# def test_get_anomaly_returns_dict(hitl, sample_anomaly, sample_vector_1d):
#   """Test anomaly retrieval."""
#
# def test_get_anomaly_with_tensor_returns_both(hitl, sample_anomaly, sample_vector_1d):
#   """Test retrieving anomaly with tensor."""
#
# def test_submit_feedback_creates_record(hitl, sample_anomaly, sample_vector_1d):
#   """Test feedback submission."""
#
# def test_get_feedback_returns_list(hitl, sample_anomaly, sample_vector_1d):
#   """Test feedback retrieval."""
#
# def test_train_model_returns_version(hitl, sample_anomaly, sample_vector_1d):
#   """Test model training."""
#
# def test_train_model_auto_detects_schema(hitl, sample_anomaly, sample_vector_1d):
#   """Test training without explicit schema_id."""
#
# def test_train_model_raises_if_ambiguous_schema(hitl):
#   """Test error when multiple schemas exist."""
#
# def test_set_live_model_activates_model(hitl, trained_model_artifacts):
#   """Test setting live model."""
#
# def test_get_live_model_returns_version(hitl, trained_model_artifacts):
#   """Test retrieving live model version."""
#
# def test_filter_predict_with_tensor(hitl, trained_model_artifacts, sample_vector_1d):
#   """Test prediction on direct tensor."""
#
# def test_filter_predict_with_anomaly_id(hitl, trained_model_artifacts, sample_anomaly, sample_vector_1d):
#   """Test prediction on stored anomaly."""
#
# def test_filter_predict_raises_if_no_live_model(hitl):
#   """Test NoLiveModel error."""
#
# def test_batch_predict(hitl, trained_model_artifacts):
#   """Test batch prediction."""
#
# def test_get_stats(hitl):
#   """Test system statistics."""
#
# def test_health_check(hitl):
#   """Test health check."""
#
# def test_list_models(hitl, trained_model_artifacts):
#   """Test model listing."""
#
# def test_close_cleans_up(hitl):
#   """Test graceful shutdown."""
#
# @pytest.mark.integration
# def test_complete_workflow(hitl):
#   """
#   Integration test: complete workflow from ingest to predict.
#   
#   Steps:
#   1. Upsert anomaly with vector
#   2. Submit feedback
#   3. Train model
#   4. Set live model
#   5. Predict on anomaly
#   6. Verify result
#   """
