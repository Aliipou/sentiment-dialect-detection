#!/bin/bash
# Script to run all tests with coverage

echo "Running tests with coverage..."

# Run unit tests
echo "=== Unit Tests ==="
pytest tests/unit/ -v --cov=src --cov-report=html --cov-report=term

# Run integration tests
echo "=== Integration Tests ==="
pytest tests/integration/ -v

# Display coverage report location
echo ""
echo "Coverage report generated at: htmlcov/index.html"
