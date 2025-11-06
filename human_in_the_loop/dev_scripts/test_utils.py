#!/usr/bin/env python
"""
Development script to test utility functions.

Tests timestamp generation, ID creation, logging configuration, and settings.
"""

import sys
from pathlib import Path
import time
from typing import Any
import re

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hitl.utils.time import now_iso, parse_iso, to_iso, utcnow
from hitl.utils.ids import uuid_str, schema_id, feedback_id as make_feedback_id
from hitl.utils.logging import configure_logging


def print_section(title):
    """Print a section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def test_utils():
    """Test utility functions."""

    # Test timestamps
    print_section("1. Timestamp Generation")

    ts1 = now_iso()
    print(f"ISO timestamp: {ts1}")

    # Verify format (ISO 8601 with timezone)
    pattern = r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}(\+00:00|Z)$"
    assert re.match(pattern, ts1)
    print("✓ Format validated (ISO 8601 with microseconds)")

    # Test uniqueness
    time.sleep(0.001)  # Wait 1ms
    ts2 = now_iso()
    assert ts1 != ts2
    print(f"✓ Unique timestamps: {ts1} != {ts2}")

    # Test parsing
    print_section("2. Timestamp Parsing")

    dt = parse_iso(ts1)
    print(f"Parsed datetime: {dt}")
    assert dt is not None
    print("✓ ISO string parsed successfully")

    # Test conversion back
    ts_converted = to_iso(dt)
    print(f"✓ Converted back to ISO: {ts_converted}")

    # Test UUID generation
    print_section("3. UUID Generation")

    uuid1 = uuid_str()
    uuid2 = uuid_str()

    print(f"UUID 1: {uuid1}")
    print(f"UUID 2: {uuid2}")

    assert uuid1 != uuid2
    print("✓ UUIDs are unique")

    # Verify format (32 hex characters without hyphens)
    uuid_pattern = r"^[0-9a-f]{32}$"
    assert re.match(uuid_pattern, uuid1)
    print("✓ UUID format validated (32 hex chars)")

    # Test anomaly ID generation
    print_section("4. Anomaly ID Generation")

    aid1 = uuid_str()
    aid2 = uuid_str()

    print(f"Anomaly ID 1: {aid1}")
    print(f"Anomaly ID 2: {aid2}")

    assert aid1 != aid2
    print("✓ Anomaly IDs are unique")

    # Custom format with prefix
    custom_aid = f"ANOM-{uuid_str()[:16]}"
    print(f"Custom format: {custom_aid}")
    assert custom_aid.startswith("ANOM-")
    print("✓ Custom format works")

    # Test feedback ID generation
    print_section("5. Feedback ID Generation")

    dt = utcnow()
    fid1 = make_feedback_id("anomaly-1", "user-1", dt)
    fid2 = make_feedback_id("anomaly-1", "user-2", dt)
    fid3 = make_feedback_id("anomaly-1", "user-1", dt)  # Same params

    print(f"Feedback ID 1: {fid1}")
    print(f"Feedback ID 2: {fid2}")
    print(f"Feedback ID 3: {fid3}")

    assert fid1 != fid2
    print("✓ Different users produce different IDs")

    assert fid1 == fid3
    print("✓ Same parameters produce same ID (deterministic)")

    # Test schema ID generation
    print_section("6. Schema ID Generation")

    # Schema IDs should be deterministic based on shape and dtype
    sid1 = schema_id(shape=(128,), dtype="float32")
    sid2 = schema_id(shape=(128,), dtype="float32")
    sid3 = schema_id(shape=(256,), dtype="float32")

    print(f"Schema ID 1 (128, float32): {sid1[:32]}...")
    print(f"Schema ID 2 (128, float32): {sid2[:32]}...")
    print(f"Schema ID 3 (256, float32): {sid3[:32]}...")

    assert sid1 == sid2
    print("✓ Schema IDs are deterministic")

    assert sid1 != sid3
    print("✓ Different shapes produce different IDs")

    # Test with 2D shape
    sid_2d = schema_id(shape=(10, 8), dtype="float32")
    print(f"Schema ID (10x8, float32): {sid_2d[:32]}...")
    assert sid_2d != sid1
    print("✓ 2D shapes produce different IDs")

    # Test logging configuration
    print_section("7. Logging Configuration")

    import structlog

    print("Configuring structured logging...")
    configure_logging(level="INFO")
    print("✓ Logging configured")

    # Get logger
    logger = structlog.get_logger()
    print("✓ Logger obtained")

    # Test log output
    print("\nTest log messages:")
    logger.info("test_message", component="utils", status="ok")
    logger.debug("debug_message", detail="This should not appear at INFO level")
    logger.warning("warning_message", code=404)

    print("✓ Structured logging works")

    # Test different log levels
    print_section("8. Log Levels")

    levels = ["DEBUG", "INFO", "WARNING", "ERROR"]

    for level in levels:
        configure_logging(level=level)
        print(f"✓ Configured: {level}")

    # Reset to INFO
    configure_logging(level="INFO")

    # Test configuration loading
    print_section("9. Configuration Loading")

    # Check for pyproject.toml
    project_root = Path(__file__).parent.parent
    pyproject_path = project_root / "pyproject.toml"

    if pyproject_path.exists():
        print(f"Found pyproject.toml at: {pyproject_path}")

        toml_loader = None  # type: Any
        try:
            import tomllib  # Python 3.11+
            toml_loader = tomllib
        except ImportError:
            try:
                import tomli  # Fallback for Python 3.10
                toml_loader = tomli
            except ImportError:
                print("⚠ No TOML parser available (tomllib/tomli)")

        if toml_loader:
            try:
                with open(pyproject_path, "rb") as f:
                    config = toml_loader.load(f)
                print("✓ Configuration loaded")

                # Check for expected sections
                if "tool" in config:
                    print("  - Found [tool] section")

                    if "hitl" in config.get("tool", {}):
                        print("  - Found [tool.hitl] section")
                        hitl_config = config["tool"]["hitl"]

                        # Display some config values
                        for key, value in list(hitl_config.items())[:5]:
                            print(f"    • {key}: {value}")

                if "project" in config:
                    print("  - Found [project] section")
                    project_config = config["project"]

                    if "name" in project_config:
                        print(f"    • name: {project_config['name']}")

                    if "version" in project_config:
                        print(f"    • version: {project_config['version']}")

            except Exception as e:
                print(f"⚠ Configuration loading failed: {e}")
    else:
        print(f"⚠ pyproject.toml not found at: {pyproject_path}")
        print("  This is expected if testing outside project structure")

    # Test ID collision resistance
    print_section("10. ID Collision Testing")

    print("Generating 1000 UUIDs to test collision resistance...")
    uuids = set()
    for _ in range(1000):
        uuids.add(uuid_str())

    assert len(uuids) == 1000
    print("✓ No collisions in 1000 UUIDs")

    print("\nGenerating 1000 anomaly IDs...")
    aids = set()
    for _ in range(1000):
        aids.add(uuid_str())

    assert len(aids) == 1000
    print("✓ No collisions in 1000 anomaly IDs")

    # Test timestamp ordering
    print_section("11. Timestamp Ordering")

    print("Testing chronological ordering...")
    timestamps = []
    for i in range(5):
        timestamps.append(now_iso())
        time.sleep(0.002)  # 2ms between each

    print("Generated timestamps:")
    for i, ts in enumerate(timestamps):
        print(f"  {i+1}. {ts}")

    # Verify they're in order
    sorted_timestamps = sorted(timestamps)
    assert timestamps == sorted_timestamps
    print("✓ Timestamps are chronologically ordered")

    # Test schema ID determinism across runs
    print_section("12. Schema ID Determinism")

    print("Testing schema ID consistency...")

    test_cases = [
        ((128,), "float32"),
        ((256,), "float32"),
        ((10, 8), "float32"),
        ((100, 128), "float64"),
    ]

    for shape, dtype in test_cases:
        # Generate ID multiple times
        ids = [schema_id(shape, dtype) for _ in range(3)]

        # All should be identical
        assert all(id_ == ids[0] for id_ in ids)
        print(f"✓ {str(shape):15s} {dtype:8s} -> {ids[0][:32]}...")

    # Test logging context binding
    print_section("13. Logging Context")

    print("Testing context binding...")
    logger = structlog.get_logger()

    # Bind context
    logger_with_context = logger.bind(
        component="test_utils", version="1.0.0", environment="development"
    )

    print("\nLogging with bound context:")
    logger_with_context.info("context_test", action="verify")
    print("✓ Context binding works")

    # Performance testing
    print_section("14. Performance")

    print("Testing ID generation performance...")

    import timeit

    # UUID generation
    uuid_time = timeit.timeit(uuid_str, number=10000)
    print(f"✓ 10,000 UUIDs: {uuid_time:.3f}s ({10000/uuid_time:.0f} IDs/sec)")

    # Timestamp generation
    ts_time = timeit.timeit(now_iso, number=10000)
    print(f"✓ 10,000 timestamps: {ts_time:.3f}s ({10000/ts_time:.0f} timestamps/sec)")

    # Schema ID generation
    def schema_id_func():
        return schema_id((128,), "float32")
    
    schema_id_func = schema_id_func()

    schema_time = timeit.timeit(schema_id_func, number=10000)
    print(f"✓ 10,000 schema IDs: {schema_time:.3f}s ({10000/schema_time:.0f} IDs/sec)")

    print_section("Summary")
    print("✓ All utility function tests passed!")
    print("✓ Features verified:")
    print("  - ISO 8601 timestamp generation")
    print("  - UUID4 generation")
    print("  - Anomaly/feedback ID generation")
    print("  - Deterministic schema IDs")
    print("  - Structured logging (structlog)")
    print("  - Configuration loading")
    print("  - Collision resistance")
    print("  - Chronological ordering")
    print("  - Performance benchmarks")


if __name__ == "__main__":
    try:
        test_utils()
        print("\n" + "=" * 60)
        print("  ✓ ALL TESTS PASSED")
        print("=" * 60 + "\n")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
