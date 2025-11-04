# Tests for FastAPI Server
#
# Test coverage:
# - All API endpoints
# - Request validation
# - Response serialization
# - Error handling
# - Authentication (if implemented)
#
# Uses httpx TestClient for synchronous testing.
#
# Test setup:
#
# from fastapi.testclient import TestClient
# from hitl.api.server import app
#
# @pytest.fixture
# def client(hitl):
#   """Provide TestClient with HITL instance."""
#   app.state.hitl = hitl
#   return TestClient(app)
#
# Test functions to implement:
#
# def test_health_endpoint(client):
#   """Test GET /health returns 200."""
#
# def test_stats_endpoint(client):
#   """Test GET /stats returns statistics."""
#
# def test_upsert_anomaly_endpoint(client):
#   """Test POST /anomalies creates anomaly."""
#
# def test_upsert_anomaly_validates_request(client):
#   """Test validation errors return 400."""
#
# def test_get_anomaly_endpoint(client):
#   """Test GET /anomalies/{id} returns anomaly."""
#
# def test_get_anomaly_not_found(client):
#   """Test 404 for missing anomaly."""
#
# def test_submit_feedback_endpoint(client):
#   """Test POST /feedback creates feedback."""
#
# def test_train_model_endpoint(client):
#   """Test POST /train starts training."""
#
# def test_list_models_endpoint(client):
#   """Test GET /models lists models."""
#
# def test_set_live_model_endpoint(client):
#   """Test POST /models/live sets live model."""
#
# def test_get_live_model_endpoint(client):
#   """Test GET /models/live returns live model."""
#
# def test_predict_endpoint_with_tensor(client):
#   """Test POST /predict with tensor."""
#
# def test_predict_endpoint_with_anomaly_id(client):
#   """Test POST /predict with anomaly_id."""
#
# def test_predict_endpoint_validates_exactly_one(client):
#   """Test validation for tensor XOR anomaly_id."""
#
# def test_error_handler_no_live_model(client):
#   """Test 503 returned for NoLiveModel."""
#
# def test_error_handler_shape_mismatch(client):
#   """Test 400 returned for ShapeMismatch."""
#
# def test_error_handler_generic(client):
#   """Test 500 returned for unexpected errors."""
#
# @pytest.mark.integration
# def test_api_complete_workflow(client):
#   """
#   Integration test via API.
#   
#   1. POST anomaly
#   2. POST feedback
#   3. POST train
#   4. POST set live
#   5. POST predict
#   """
