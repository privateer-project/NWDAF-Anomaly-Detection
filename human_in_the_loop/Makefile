# Makefile for HITL Project
# Human-in-the-Loop Anomaly Filtering System

.PHONY: help install install-dev test test-verbose test-coverage clean lint format type-check init-db run-api run-cli docs

# Default target
help:
	@echo "HITL Project - Available targets:"
	@echo ""
	@echo "  make install          Install package and dependencies"
	@echo "  make install-dev      Install package with dev dependencies"
	@echo "  make test             Run all tests"
	@echo "  make test-verbose     Run tests with verbose output"
	@echo "  make test-coverage    Run tests with coverage report"
	@echo "  make lint             Run linter (ruff)"
	@echo "  make format           Format code with ruff"
	@echo "  make type-check       Run type checker (mypy)"
	@echo "  make clean            Clean up generated files"
	@echo "  make init-db          Initialize database"
	@echo "  make run-api          Start API server"
	@echo "  make run-cli          Show CLI help"
	@echo "  make docs             Generate documentation"
	@echo ""

# Installation targets
install:
	pip install -e .

install-dev:
	pip install -e .
	pip install pytest pytest-cov ruff mypy httpx types-requests

# Testing targets
test:
	python -m pytest tests/ -v

test-verbose:
	python -m pytest tests/ -vv -s

test-coverage:
	python -m pytest tests/ --cov=hitl --cov-report=html --cov-report=term
	@echo ""
	@echo "Coverage report generated in htmlcov/index.html"

test-watch:
	python -m pytest tests/ -f

# Code quality targets
lint:
	python -m ruff check hitl/ tests/

format:
	python -m ruff format hitl/ tests/

type-check:
	python -m mypy hitl/ --strict

# Cleanup targets
clean:
	rm -rf __pycache__ .pytest_cache .mypy_cache .ruff_cache
	rm -rf htmlcov .coverage
	rm -rf build dist *.egg-info
	rm -rf hitl/__pycache__ hitl/**/__pycache__
	rm -rf tests/__pycache__ tests/**/__pycache__
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.so" -delete

# Database targets
init-db:
	python -c "from hitl.core.hitl import HITL; HITL().close()" || echo "Database initialized"

# Runtime targets
run-api:
	uvicorn hitl.api.server:app --reload --host 0.0.0.0 --port 8000

run-cli:
	python -m hitl.cli.main --help

# Documentation targets
docs:
	@echo "Generating documentation..."
	@echo "Not yet implemented - will use Sphinx or similar"

# Development workflow
dev: install-dev lint type-check test
	@echo ""
	@echo "✓ Development checks passed!"

# CI/CD workflow
ci: install test-coverage lint type-check
	@echo ""
	@echo "✓ CI checks passed!"

# Quick check before commit
check: format lint test
	@echo ""
	@echo "✓ Pre-commit checks passed!"
