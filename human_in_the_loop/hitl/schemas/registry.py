"""
Feature Schema Registry

This module manages feature vector schemas, ensuring consistent
tensor shapes and types throughout the system.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np

from hitl.errors import SchemaNotFound, UnsupportedShape, ValidationError
from hitl.types import SchemaInfo
from hitl.utils.ids import schema_id

if TYPE_CHECKING:
    import sqlite3
    from hitl.store.repository import Repository

__all__ = ["SchemaRegistry", "shape_to_str", "str_to_shape"]


def shape_to_str(shape: tuple[int, ...]) -> str:
    """
    Convert shape tuple to string for database storage.

    Args:
        shape: Tensor dimensions

    Returns:
        JSON string representation

    Example:
        >>> shape_to_str((128,))
        '[128]'
        >>> shape_to_str((10, 8))
        '[10, 8]'
    """
    return json.dumps(list(shape))


def str_to_shape(s: str) -> tuple[int, ...]:
    """
    Parse stored shape string back to tuple.

    Args:
        s: JSON or comma-separated string

    Returns:
        Tuple of integers

    Example:
        >>> str_to_shape('[128]')
        (128,)
        >>> str_to_shape('10,8')
        (10, 8)
    """
    # Try JSON first
    try:
        parsed = json.loads(s)
        if isinstance(parsed, list):
            return tuple(int(x) for x in parsed)
    except (json.JSONDecodeError, ValueError):
        pass

    # Fall back to comma-separated
    parts = s.strip().split(",")
    return tuple(int(x.strip()) for x in parts if x.strip())


class SchemaRegistry:
    """Registry for feature tensor schemas with validation."""

    def __init__(self, repository: Repository) -> None:
        """
        Initialize registry with repository for persistence.

        Args:
            repository: Repository instance for database access
        """
        self.repo = repository

    def ensure(self, shape: tuple[int, ...], dtype: str = "float32") -> SchemaInfo:
        """
        Ensure schema exists, creating if needed.

        This is the main entry point for schema registration.

        Args:
            shape: Tensor dimensions as tuple (e.g., (128,) or (10, 8))
            dtype: NumPy dtype string (default: "float32")

        Returns:
            SchemaInfo dict with schema_id and metadata

        Raises:
            UnsupportedShape: if shape not (D,) or (T,F)
            ValidationError: if dtype invalid
            DBError: on database errors

        Example:
            >>> schema = registry.ensure(shape=(128,), dtype="float32")
            >>> schema["schema_id"]
            'schema_abc123...'
        """
        # Validate inputs
        self.validate_shape(shape)
        self.validate_dtype(dtype)

        # Compute schema_id
        sid = schema_id(shape, dtype)

        # Check if already exists
        existing = self.repo.get_schema(sid)
        if existing is not None:
            return self._row_to_schema_info(existing)

        # Insert new schema
        metadata = self._compute_metadata(shape)
        shape_str = shape_to_str(shape)

        from hitl.utils.time import now_iso

        self.repo.insert_schema(
            schema_id=sid,
            shape=shape_str,
            ndim=metadata["ndim"],
            numel=metadata["numel"],
            dtype=dtype,
            created_at=now_iso(),
        )

        # Return SchemaInfo
        return SchemaInfo(
            schema_id=sid,
            shape=shape,
            ndim=metadata["ndim"],
            numel=metadata["numel"],
            dtype=dtype,
        )

    def get(self, schema_id: str) -> SchemaInfo:
        """
        Retrieve schema by ID.

        Args:
            schema_id: Schema identifier

        Returns:
            SchemaInfo dict

        Raises:
            SchemaNotFound: if ID doesn't exist
        """
        row = self.repo.get_schema(schema_id)
        if row is None:
            raise SchemaNotFound(f"Schema not found: {schema_id}")
        return self._row_to_schema_info(row)

    def from_anomaly(self, anomaly_id: str) -> SchemaInfo:
        """
        Get schema for an anomaly's stored vector.

        Args:
            anomaly_id: Anomaly identifier

        Returns:
            SchemaInfo dict

        Raises:
            ValueError: if anomaly_id not found or has no vector
            SchemaNotFound: if schema_id invalid
        """
        # Get vector record (includes schema_id)
        vector_data = self.repo.get_vector(anomaly_id)
        if vector_data is None:
            raise ValueError(
                f"No vector stored for anomaly: {anomaly_id}. "
                "Call put_vector() first."
            )

        sid, _ = vector_data  # Unpack (schema_id, blob) tuple
        return self.get(sid)

    def list_all(self) -> list[SchemaInfo]:
        """
        List all registered schemas.

        Returns:
            List of SchemaInfo dicts

        Useful for training when schema_id not specified.
        """
        rows = self.repo.list_schemas()
        return [self._row_to_schema_info(row) for row in rows]

    def validate_shape(self, shape: tuple[int, ...]) -> None:
        """
        Validate that shape is supported (1D or 2D).

        Args:
            shape: Tensor dimensions

        Raises:
            UnsupportedShape: with detailed message

        Validation rules:
            - Must be 1D: (D,) where D > 0
            - OR 2D: (T, F) where T > 0 and F > 0
            - No 0 dimensions allowed
            - No 3D+ tensors
        """
        if not isinstance(shape, tuple):
            raise UnsupportedShape(f"Shape must be a tuple, got {type(shape).__name__}")

        if len(shape) == 0:
            raise UnsupportedShape(
                "Shape cannot be empty (0D scalar). "
                "Supported shapes: 1D (D,) for dense mode, 2D (T, F) for conv1d mode."
            )

        if len(shape) > 2:
            raise UnsupportedShape(
                f"Shape must be 1D or 2D, got {len(shape)}D: {shape}. "
                "Supported shapes: 1D (D,) for dense mode, 2D (T, F) for conv1d mode."
            )

        # Check for zero dimensions
        if any(dim <= 0 for dim in shape):
            raise UnsupportedShape(
                f"All dimensions must be positive, got shape: {shape}"
            )

    def validate_dtype(self, dtype: str) -> None:
        """
        Validate NumPy dtype string.

        Args:
            dtype: NumPy dtype string

        Raises:
            ValidationError: if invalid

        Supported dtypes:
            - "float32" (primary)
            - "float64"
            - "int32", "int64"
        """
        try:
            np_dtype = np.dtype(dtype)

            # Check if supported
            supported = {"float32", "float64", "int32", "int64"}
            if np_dtype.name not in supported:
                raise ValidationError(
                    f"Unsupported dtype: {dtype}. "
                    f"Supported dtypes: {', '.join(sorted(supported))}"
                )
        except (TypeError, ValueError) as e:
            raise ValidationError(f"Invalid dtype: {dtype}") from e

    def _compute_metadata(self, shape: tuple[int, ...]) -> dict:
        """
        Compute ndim and numel from shape.

        Args:
            shape: Tensor dimensions

        Returns:
            dict with "ndim" and "numel"

        Example:
            >>> _compute_metadata((128,))
            {'ndim': 1, 'numel': 128}
        """
        ndim = len(shape)
        numel = 1
        for dim in shape:
            numel *= dim

        return {"ndim": ndim, "numel": numel}

    def _row_to_schema_info(self, row: sqlite3.Row) -> SchemaInfo:
        """
        Convert database row to SchemaInfo dict.

        Args:
            row: Row from feature_schemas table

        Returns:
            SchemaInfo TypedDict
        """
        shape = str_to_shape(row["shape"])

        return SchemaInfo(
            schema_id=row["schema_id"],
            shape=shape,
            ndim=row["ndim"],
            numel=row["numel"],
            dtype=row["dtype"],
        )
