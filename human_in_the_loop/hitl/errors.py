"""
Custom exceptions for HITL system.

This module defines a hierarchy of exceptions used throughout the system for
error handling.

Exception Hierarchy:
    HITLError (base)
    ├── UnsupportedShape: Invalid tensor shape (not 1D or 2D)
    ├── ShapeMismatch: Shape doesn't match registered schema
    ├── SchemaNotFound: Schema ID not found in registry
    ├── NoLiveModel: No live model currently set
    ├── ArtifactMissing: Model artifacts not found
    ├── DBError: Database operation failed
    └── ValidationError: Data validation failed
"""


class HITLError(Exception):
    """
    Base exception for all HITL-specific errors.

    All custom exceptions in the HITL system inherit from this base class,
    making it easy to catch all HITL-related errors.

    Example:
        >>> try:
        ...     raise HITLError("Something went wrong")
        ... except HITLError as e:
        ...     print(f"HITL error: {e}")
        HITL error: Something went wrong
    """

    pass


class UnsupportedShape(HITLError):
    """
    Raised when tensor has invalid shape (not 1D or 2D).

    The HITL system only supports:
        * 1D vectors for "dense" models (shape: (n,))
        * 2D tensors for "conv1d" models (shape: (features, timesteps))

    Example:
        >>> raise UnsupportedShape("Expected 1D or 2D, got shape (2, 3, 4)")
        Traceback (most recent call last):
        ...
        hitl.errors.UnsupportedShape: Expected 1D or 2D, got shape (2, 3, 4)
    """

    pass


class ShapeMismatch(HITLError):
    """
    Raised when tensor shape doesn't match registered schema.

    When an anomaly is associated with a schema, all operations must use
    tensors with the same shape as defined by that schema.

    Example:
        >>> raise ShapeMismatch("Expected shape (128,), got (64,)")
        Traceback (most recent call last):
        ...
        hitl.errors.ShapeMismatch: Expected shape (128,), got (64,)
    """

    pass


class SchemaNotFound(HITLError):
    """
    Raised when schema ID is not found in the registry.

    This typically indicates an anomaly references a schema that was never
    registered or has been deleted.

    Example:
        >>> raise SchemaNotFound("Schema abc123 not found")
        Traceback (most recent call last):
        ...
        hitl.errors.SchemaNotFound: Schema abc123 not found
    """

    pass


class NoLiveModel(HITLError):
    """
    Raised when attempting prediction but no live model is set.

    Before making predictions, a trained model must be designated as the
    "live" model using set_live_model().

    Example:
        >>> raise NoLiveModel("No live model set for predictions")
        Traceback (most recent call last):
        ...
        hitl.errors.NoLiveModel: No live model set for predictions
    """

    pass


class ArtifactMissing(HITLError):
    """
    Raised when model artifacts are not found or incomplete.

    A complete model requires at least:
        * weights (.pth)
        * config (.json)
        * scaler (.json)
        * threshold (.json)

    Example:
        >>> raise ArtifactMissing("Missing scaler.json for model AE-2025.11.04-1")
        Traceback (most recent call last):
        ...
        hitl.errors.ArtifactMissing: Missing scaler.json for model AE-2025.11.04-1
    """

    pass


# Backwards-compatible alias for older test/dev scripts
class ArtifactNotFound(ArtifactMissing):
    """Alias for ArtifactMissing kept for backward compatibility."""


class ValidationError(HITLError):
    """
    Raised when data validation fails.

    This includes:
        * Pydantic validation errors
        * Custom validation logic (e.g. checking for NaN/Inf values)
        * Schema-level inconsistencies in anomaly payloads

    Example:
        >>> raise ValidationError("Tensor contains NaN values")
        Traceback (most recent call last):
        ...
        hitl.errors.ValidationError: Tensor contains NaN values
    """

    pass


# InvalidArray used by older serialization tests
class InvalidArray(ValidationError):
    """
    Raised when an input array is invalid.

    Typical reasons:
        * Contains NaN or Inf values
        * Has zero length
        * Has unexpected dimensions

    Example:
        >>> raise InvalidArray("Input array is empty")
        Traceback (most recent call last):
        ...
        hitl.errors.InvalidArray: Input array is empty
    """

    pass


class DBError(HITLError):
    """
    Raised when a database operation fails.

    This wraps SQLite (or other driver) errors with additional context about
    what operation was being attempted.

    Example:
        >>> raise DBError("Failed to insert anomaly: UNIQUE constraint failed")
        Traceback (most recent call last):
        ...
        hitl.errors.DBError: Failed to insert anomaly: UNIQUE constraint failed
    """

    pass
