# Tests Documentation

This directory contains comprehensive tests for the librobot_vla project.

## Directory Structure

```
tests/
├── conftest.py                      # Shared pytest fixtures and configuration
├── pytest.ini                       # Pytest configuration
├── test_vlm_implementations.py      # Existing VLM implementation tests
│
├── unit/                           # Unit tests
│   ├── __init__.py
│   ├── test_registry.py            # Registry system tests
│   ├── test_config.py              # Configuration management tests
│   ├── test_models.py              # Model architecture tests
│   ├── test_action_heads.py        # Action head module tests
│   ├── test_encoders.py            # Encoder module tests
│   ├── test_frameworks.py          # Training/inference framework tests
│   ├── test_robots.py              # Robot interface tests
│   └── test_data.py                # Data loading and processing tests
│
├── integration/                    # Integration tests
│   ├── __init__.py
│   ├── test_training_pipeline.py   # End-to-end training tests
│   ├── test_inference_pipeline.py  # End-to-end inference tests
│   └── test_end_to_end.py         # Complete system integration tests
│
└── benchmarks/                     # Performance benchmarks
    ├── __init__.py
    ├── benchmark_inference.py      # Inference performance benchmarks
    └── benchmark_training.py       # Training performance benchmarks
```

## Running Tests

### Run All Tests

```bash
pytest
```

### Run Specific Test Categories

```bash
# Run only unit tests
pytest tests/unit/

# Run only integration tests
pytest tests/integration/

# Run only benchmarks
pytest tests/benchmarks/
```

### Run Specific Test Files

```bash
# Run a specific test file
pytest tests/unit/test_models.py

# Run a specific test class
pytest tests/unit/test_models.py::TestVLAModel

# Run a specific test method
pytest tests/unit/test_models.py::TestVLAModel::test_model_initialization
```

### Run Tests with Markers

```bash
# Run only fast tests (exclude slow tests)
pytest -m "not slow"

# Run only GPU tests
pytest -m gpu

# Run only integration tests
pytest -m integration

# Run only benchmark tests
pytest -m benchmark
```

### Run Tests with Coverage

```bash
# Generate coverage report
pytest --cov=librobot --cov-report=html --cov-report=term-missing

# View coverage report
open htmlcov/index.html
```

### Run Tests in Parallel

```bash
# Install pytest-xdist first
pip install pytest-xdist

# Run tests in parallel (auto-detect number of CPUs)
pytest -n auto

# Run tests with specific number of workers
pytest -n 4
```

### Verbose Output

```bash
# Show detailed output
pytest -v

# Show even more detailed output
pytest -vv

# Show print statements
pytest -s

# Show local variables in tracebacks
pytest -l
```

## Test Fixtures

Common fixtures are defined in `conftest.py` and are available to all tests:

### Device Fixtures
- `device`: PyTorch device (CUDA/CPU)
- `random_seed`: Set random seed for reproducibility

### Model Fixtures
- `model_config`: Standard model configuration
- `small_model_config`: Small model for faster tests

### Data Fixtures
- `sample_image`: Single image tensor
- `sample_batch_images`: Batch of images
- `sample_state`: Robot state
- `sample_action`: Robot action
- `sample_trajectory`: Complete trajectory
- `sample_batch`: Batch for training/inference

### Training Fixtures
- `training_config`: Training configuration
- `optimizer_config`: Optimizer configuration
- `scheduler_config`: Scheduler configuration

### Robot Fixtures
- `robot_config`: Robot configuration

### Dataset Fixtures
- `dataset_config`: Dataset configuration
- `episode_data`: Sample episode data

### Inference Fixtures
- `inference_config`: Inference configuration

## Writing New Tests

### Unit Test Example

```python
import pytest
import torch

class TestMyModule:
    """Test suite for MyModule."""
    
    def test_initialization(self):
        """Test module initialization."""
        # TODO: Implement test
        pass
    
    def test_forward_pass(self, sample_batch, device):
        """Test forward pass."""
        # TODO: Implement test
        pass
    
    @pytest.mark.parametrize("batch_size", [1, 4, 16])
    def test_various_batch_sizes(self, batch_size):
        """Test with various batch sizes."""
        # TODO: Implement test
        pass
```

### Integration Test Example

```python
import pytest

class TestPipeline:
    """Test suite for complete pipeline."""
    
    def test_end_to_end(self, training_config, output_dir):
        """Test end-to-end pipeline."""
        # TODO: Implement test
        pass
```

### Benchmark Example

```python
import pytest
import time

class BenchmarkPerformance:
    """Benchmark suite for performance."""
    
    def test_inference_latency(self, model, sample_batch):
        """Benchmark inference latency."""
        latencies = []
        for _ in range(100):
            start = time.perf_counter()
            output = model(sample_batch)
            latencies.append(time.perf_counter() - start)
        
        mean_latency = sum(latencies) / len(latencies)
        print(f"Mean latency: {mean_latency*1000:.2f} ms")
```

## Test Markers

Use markers to categorize tests:

```python
@pytest.mark.slow
def test_long_running_operation():
    """Test that takes a long time."""
    pass

@pytest.mark.gpu
def test_gpu_operation():
    """Test that requires GPU."""
    pass

@pytest.mark.integration
def test_integration():
    """Integration test."""
    pass

@pytest.mark.benchmark
def test_benchmark():
    """Benchmark test."""
    pass
```

## Parametrized Tests

Use parametrize for multiple test cases:

```python
@pytest.mark.parametrize("input_size,expected_output", [
    (10, 20),
    (20, 40),
    (30, 60),
])
def test_with_parameters(input_size, expected_output):
    """Test with different parameters."""
    result = function_to_test(input_size)
    assert result == expected_output
```

## Mocking

Use unittest.mock for testing without dependencies:

```python
from unittest.mock import Mock, patch

def test_with_mock():
    """Test using mock objects."""
    mock_object = Mock()
    mock_object.method.return_value = 42
    
    result = mock_object.method()
    assert result == 42
    mock_object.method.assert_called_once()
```

## Best Practices

1. **Write descriptive test names**: Test names should clearly describe what is being tested
2. **Use fixtures**: Leverage pytest fixtures for common setup
3. **Test one thing**: Each test should test a single behavior
4. **Use assertions**: Make clear assertions about expected behavior
5. **Add docstrings**: Explain what each test is verifying
6. **Mark slow tests**: Use `@pytest.mark.slow` for long-running tests
7. **Parametrize when possible**: Use parametrize for testing multiple cases
8. **Clean up**: Ensure tests clean up any resources they create
9. **Mock external dependencies**: Use mocks for external systems
10. **Keep tests independent**: Tests should not depend on each other

## Continuous Integration

These tests are designed to run in CI/CD pipelines. The typical CI workflow:

1. Install dependencies
2. Run linting (if configured)
3. Run unit tests
4. Run integration tests (if not slow)
5. Generate coverage report
6. Upload coverage to coverage service

## TODO Items

Many tests are marked with `# TODO: Implement` comments. These indicate:
- Tests that need implementation once features are developed
- Tests that are placeholders for comprehensive coverage
- Areas where additional test cases should be added

As you develop features, implement the corresponding tests to ensure quality and coverage.

## Contributing

When adding new features:
1. Add corresponding unit tests
2. Add integration tests if the feature involves multiple components
3. Add benchmarks if performance is critical
4. Update this README if you add new test categories
5. Ensure all tests pass before submitting PR

## Contact

For questions about tests, please refer to the main project documentation or contact the maintainers.
