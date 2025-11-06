"""
ID generation utilities for HITL system.

This module provides functions for generating various types of identifiers used throughout
the system.

Functions:
    - uuid_str() -> str: Generate UUID4 string
    - sha1_bytes(data: bytes) -> str: Hash bytes with SHA-1
    - schema_id(shape: tuple, dtype: str) -> str: Generate deterministic schema ID
    - model_version(date: Optional[datetime], sequence: int) -> str: Generate model version
    - feedback_id(anomaly_id: str, user_id: str, timestamp: datetime) -> str: Generate feedback ID
"""

import hashlib
import uuid
from datetime import datetime
from typing import Optional

from .time import utcnow


def uuid_str() -> str:
    """
    Generate a random UUID4 string.

    Returns:
        str: UUID4 as lowercase string without hyphens

    Example:
        >>> id1 = uuid_str()
        >>> id2 = uuid_str()
        >>> id1 != id2
        True
        >>> len(id1) == 32
        True
    """
    return uuid.uuid4().hex


def sha1_bytes(data: bytes) -> str:
    """
    Hash bytes with SHA-1 and return hex digest.

    Args:
        data: Bytes to hash

    Returns:
        str: SHA-1 hex digest (40 characters)

    Example:
        >>> sha1_bytes(b"hello")
        'aaf4c61ddcc5e8a2dabede0f3b482cd9aea9434d'
    """
    return hashlib.sha1(data).hexdigest()


def schema_id(shape: tuple[int, ...], dtype: str) -> str:
    """
    Generate deterministic schema ID from shape and dtype.

    The schema ID is a SHA-1 hash of the serialized shape and dtype, ensuring
    that the same shape+dtype always produces the same ID.

    Args:
        shape: Tuple of dimensions (e.g., (128,) or (8, 128))
        dtype: NumPy dtype as string (e.g., "float32")

    Returns:
        str: SHA-1 hex digest (40 characters)

    Example:
        >>> id1 = schema_id((128,), "float32")
        >>> id2 = schema_id((128,), "float32")
        >>> id1 == id2
        True
        >>> id3 = schema_id((64,), "float32")
        >>> id1 != id3
        True
    """
    # Serialize shape and dtype to bytes
    shape_str = ",".join(map(str, shape))
    key = f"{shape_str}:{dtype}"
    return sha1_bytes(key.encode("utf-8"))


def model_version(date: Optional[datetime] = None, sequence: int = 1) -> str:
    """
    Generate model version string in format: AE-YYYY.MM.DD-N

    Args:
        date: Date for version (defaults to current UTC date)
        sequence: Sequence number for same day (starts at 1)

    Returns:
        str: Model version string

    Example:
        >>> from datetime import datetime, timezone
        >>> dt = datetime(2025, 11, 4, tzinfo=timezone.utc)
        >>> model_version(dt, 1)
        'AE-2025.11.04-1'
        >>> model_version(dt, 2)
        'AE-2025.11.04-2'
    """
    if date is None:
        date = utcnow()

    date_str = date.strftime("%Y.%m.%d")
    return f"AE-{date_str}-{sequence}"


def feedback_id(anomaly_id: str, user_id: str, timestamp: datetime) -> str:
    """
    Generate deterministic feedback ID from anomaly, user, and timestamp.

    This ensures that the same user providing feedback on the same anomaly at the
    same time produces the same ID (useful for idempotent operations).

    Args:
        anomaly_id: ID of the anomaly
        user_id: ID of the user providing feedback
        timestamp: When feedback was submitted

    Returns:
        str: SHA-1 hex digest (40 characters)

    Example:
        >>> from datetime import datetime, timezone
        >>> dt = datetime(2025, 11, 4, 10, 30, 0, tzinfo=timezone.utc)
        >>> fid = feedback_id("A1", "user1", dt)
        >>> len(fid) == 40
        True
    """
    # Serialize to string and hash
    key = f"{anomaly_id}:{user_id}:{timestamp.isoformat()}"
    return sha1_bytes(key.encode("utf-8"))
