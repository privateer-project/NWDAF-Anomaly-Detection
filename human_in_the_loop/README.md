# HITL (Human-in-the-Loop) Anomaly Filtering System

This package provides a complete system for managing anomalies with human feedback,
training autoencoder models, and making predictions on new data.

## Quick Start

### Using Make (Recommended)

```bash
# See all available commands
make help

# Run all tests
make test

# Run tests with coverage
make test-coverage

# Install dependencies
make install-dev

# Run code quality checks
make lint
make format
make type-check

# Quick pre-commit check
make check
```

### Development Scripts

For manual testing and debugging, see the `dev_scripts/` directory:

```bash
# Run all dev scripts
cd dev_scripts && ./run_all.sh

# Or run individual test scripts
python dev_scripts/test_utils.py
python dev_scripts/test_database.py
python dev_scripts/test_full_workflow.py
# ... etc

# See QUICKREF.md for details
cat dev_scripts/QUICKREF.md
```

The dev scripts provide functional testing for all implemented phases (1-6).

## Overview
<!--
This README should contain:
- High-level project description
- System purpose and use cases
- Key features list
- Installation instructions (uv-based)
- Quick start guide with example commands
- Architecture overview (link to ARCHITECTURE.md)
- CLI usage examples
- API usage examples
- Configuration options
- Development setup instructions
- Testing instructions
- Contributing guidelines
- License information
-->

## Installation

## Quick Start

## CLI Usage

## API Usage

## Development

## Testing

## License
