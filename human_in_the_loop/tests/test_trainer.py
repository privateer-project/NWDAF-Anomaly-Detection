# Tests for Training Pipeline
#
# Test coverage:
# - Dataset loading from database
# - Data splitting
# - Scaler computation
# - Normalization
# - Training loop (integration test on small data)
# - Threshold computation
# - Early stopping
# - train_and_publish workflow
#
# Test functions to implement:
#
# def test_load_dataset_returns_array_and_ids(trainer, repository, registry, sample_vector_1d):
#   """Test loading vectors from database."""
#
# def test_load_dataset_raises_if_empty(trainer):
#   """Test error when no vectors for schema."""
#
# def test_split_data(trainer):
#   """Test train/val splitting."""
#
# def test_compute_scaler_dense(trainer):
#   """Test scaler computation for dense mode."""
#
# def test_compute_scaler_conv1d(trainer):
#   """Test scaler computation for conv1d mode."""
#
# def test_normalize_dense(trainer):
#   """Test normalization for dense mode."""
#
# def test_normalize_conv1d(trainer):
#   """Test normalization for conv1d mode."""
#
# def test_fit_trains_model(trainer, sample_vector_1d):
#   """Test model training completes."""
#
# def test_fit_returns_artifacts(trainer, sample_vector_1d):
#   """Test fit returns state_dict, scaler, threshold, metrics."""
#
# def test_compute_threshold(trainer, sample_vector_1d):
#   """Test threshold computation from train errors."""
#
# def test_train_and_publish_creates_version(trainer, repository, registry, sample_vector_1d):
#   """Test complete train and publish workflow."""
#
# def test_train_epoch_decreases_loss(trainer):
#   """Test training reduces loss (sanity check)."""
#
# def test_validate_computes_loss(trainer):
#   """Test validation loss computation."""
#
# def test_early_stopping_triggers(trainer):
#   """Test early stopping with patience."""
#
# @pytest.mark.slow
# def test_overfit_tiny_dataset(trainer):
#   """Test model can overfit small dataset (convergence test)."""
