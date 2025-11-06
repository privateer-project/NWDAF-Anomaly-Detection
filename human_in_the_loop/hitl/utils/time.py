"""
Time utilities for HITL system.

This module provides timezone-aware datetime utilities for consistent timestamp handling
across the system.

Functions:
    - utcnow() -> datetime: Return current UTC time with timezone info
    - now_iso() -> str: Return current UTC time as ISO 8601 string
    - parse_iso(s: str) -> datetime: Parse ISO 8601 string to datetime
    - to_iso(dt: datetime) -> str: Convert datetime to ISO 8601 string
    - validate_iso(s: str) -> bool: Check if string is valid ISO 8601 timestamp
"""

from datetime import datetime, timezone


def utcnow() -> datetime:
    """
    Return current UTC time with timezone info.

    Returns:
        datetime: Current UTC time with tzinfo=timezone.utc

    Example:
        >>> now = utcnow()
        >>> now.tzinfo == timezone.utc
        True
    """
    return datetime.now(timezone.utc)


def now_iso() -> str:
    """
    Return current UTC time as ISO 8601 string.

    Returns:
        str: ISO 8601 formatted timestamp (e.g., "2025-11-04T10:30:45.123456+00:00")

    Example:
        >>> timestamp = now_iso()
        >>> "T" in timestamp and "+" in timestamp
        True
    """
    return utcnow().isoformat()


def parse_iso(s: str) -> datetime:
    """
    Parse ISO 8601 string to timezone-aware datetime.

    Args:
        s: ISO 8601 formatted string

    Returns:
        datetime: Parsed datetime with timezone info

    Raises:
        ValueError: If string is not valid ISO 8601 format

    Example:
        >>> dt = parse_iso("2025-11-04T10:30:45+00:00")
        >>> dt.year == 2025
        True
    """
    try:
        dt = datetime.fromisoformat(s)
        # Ensure timezone-aware
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError) as e:
        raise ValueError(f"Invalid ISO 8601 timestamp: {s}") from e


def to_iso(dt: datetime) -> str:
    """
    Convert datetime to ISO 8601 string.

    Args:
        dt: Datetime object (timezone-aware or naive)

    Returns:
        str: ISO 8601 formatted timestamp

    Example:
        >>> dt = datetime(2025, 11, 4, 10, 30, 45, tzinfo=timezone.utc)
        >>> iso = to_iso(dt)
        >>> "2025-11-04" in iso
        True
    """
    # If naive, assume UTC
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()


def validate_iso(s: str) -> bool:
    """
    Check if string is valid ISO 8601 timestamp.

    Args:
        s: String to validate

    Returns:
        bool: True if valid ISO 8601, False otherwise

    Example:
        >>> validate_iso("2025-11-04T10:30:45+00:00")
        True
        >>> validate_iso("not a timestamp")
        False
    """
    try:
        parse_iso(s)
        return True
    except (ValueError, TypeError):
        return False
