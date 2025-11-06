"""
Structured logging utilities for HITL system.

This module configures structured logging using structlog for consistent, machine-readable
logs across the system.

Functions:
    - configure_logging(level: str, dev_mode: bool): Setup structlog with processors
    - get_logger(name: str): Return logger bound to module name
    - bind_context(**kwargs): Bind context variables to current thread
    - clear_context(): Clear thread-local context
"""

import logging
import sys
from typing import Any

import structlog
from structlog.typing import Processor


def configure_logging(level: str = "INFO", dev_mode: bool = False) -> None:
    """
    Configure structlog with appropriate processors.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        dev_mode: If True, use human-readable console output; if False, use JSON

    Example:
        >>> configure_logging(level="DEBUG", dev_mode=True)
    """
    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=numeric_level,
    )

    # Choose renderer based on mode
    if dev_mode:
        renderer: Processor = structlog.dev.ConsoleRenderer()
    else:
        renderer = structlog.processors.JSONRenderer()

    # Configure structlog
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            renderer,
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """
    Get a logger instance bound to a specific name.

    Args:
        name: Logger name (typically __name__ of the module)

    Returns:
        structlog.stdlib.BoundLogger: Configured logger instance

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("operation_started", operation="train")
    """
    return structlog.get_logger(name)


def bind_context(**kwargs: Any) -> None:
    """
    Bind context variables that will be included in all log entries.

    Uses structlog's context variables feature to add fields to all logs
    in the current context (thread/async task).

    Args:
        **kwargs: Key-value pairs to bind to context

    Example:
        >>> bind_context(request_id="abc123", user_id="user1")
        >>> logger = get_logger(__name__)
        >>> logger.info("processing")  # Will include request_id and user_id
    """
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(**kwargs)


def clear_context() -> None:
    """
    Clear all thread-local context variables.

    Should be called at the end of request/task processing to avoid
    context leaking between requests.

    Example:
        >>> bind_context(request_id="abc123")
        >>> clear_context()
        >>> # request_id no longer in context
    """
    structlog.contextvars.clear_contextvars()
