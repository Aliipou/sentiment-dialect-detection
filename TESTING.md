# Testing Guide

## Test Structure

```
tests/
├── conftest.py          # Shared fixtures
├── unit/                # Fast tests, no external dependencies
│   ├── test_text_processor.py
│   ├── test_data_loader.py
│   └── test_api_schemas.py
└── integration/         # Full system tests
    └── test_api.py      # API endpoint tests (slow, requires models)
```

## Running Tests

### Quick Tests (Unit Only)

```bash
# Run only unit tests (fast, < 1 minute)
pytest tests/unit/ -v

# With coverage
pytest tests/unit/ --cov=src --cov-report=html
```

### Full Test Suite

```bash
# Run all tests including integration (slow, ~5-10 minutes first time)
pytest tests/ -v

# Skip slow tests
pytest tests/ -v -m "not slow"
```

### Specific Test Files

```bash
# Test specific module
pytest tests/unit/test_text_processor.py -v

# Test specific function
pytest tests/unit/test_text_processor.py::TestPersianTextPreprocessor::test_clean_text_removes_non_persian -v
```

## Test Coverage

```bash
# Generate HTML coverage report
pytest --cov=src --cov-report=html

# View report
# Open htmlcov/index.html in browser
```

## Integration Testing

Integration tests require models to be loaded and may take longer:

```bash
# Run integration tests only
pytest tests/integration/ -v

# Run integration tests with API server
# (API must be running on localhost:8000)
pytest tests/integration/test_api.py -v
```

## Writing Tests

### Unit Test Example

```python
# tests/unit/test_my_module.py
import pytest
from src.my_module import MyClass

class TestMyClass:
    def test_basic_functionality(self):
        obj = MyClass()
        result = obj.do_something("input")
        assert result == "expected_output"

    def test_error_handling(self):
        obj = MyClass()
        with pytest.raises(ValueError):
            obj.do_something(None)
```

### Integration Test Example

```python
# tests/integration/test_my_api.py
from fastapi.testclient import TestClient
from src.api.app import app

client = TestClient(app)

def test_api_endpoint():
    response = client.post("/endpoint", json={"key": "value"})
    assert response.status_code == 200
    assert "expected_key" in response.json()
```

## Continuous Integration

Tests run automatically on GitHub Actions:
- On every push to `main` or `develop`
- On every pull request
- Multiple Python versions (3.8, 3.9, 3.10, 3.11)

See `.github/workflows/ci.yml` for CI configuration.

## Test Markers

```python
# Mark slow tests
@pytest.mark.slow
def test_expensive_operation():
    pass

# Mark integration tests
@pytest.mark.integration
def test_full_system():
    pass

# Skip tests conditionally
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires GPU")
def test_gpu_operation():
    pass
```

## Debugging Tests

```bash
# Run with verbose output
pytest -vv

# Show print statements
pytest -s

# Drop into debugger on failure
pytest --pdb

# Run last failed tests only
pytest --lf
```

## Performance Testing

```bash
# Profile test execution time
pytest --durations=10

# Parallel test execution
pytest -n auto  # Requires pytest-xdist
```

## Best Practices

1. **Test Independence**: Each test should be independent and not rely on others
2. **Use Fixtures**: Share common setup code via fixtures in `conftest.py`
3. **Mock External Services**: Don't call external APIs in unit tests
4. **Clear Names**: Test names should describe what they test
5. **Fast Unit Tests**: Keep unit tests fast (< 1 second each)
6. **Integration Tags**: Mark slow integration tests with `@pytest.mark.slow`
