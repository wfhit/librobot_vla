"""
Unit tests for the seed utilities module.

Tests random seed setting and state management for reproducibility.
"""

import random

import numpy as np
import pytest
import torch

from librobot.utils.seed import (
    set_seed,
    get_random_state,
    set_random_state,
    make_deterministic,
)


class TestSetSeed:
    """Test suite for set_seed function."""

    def test_set_seed_basic(self):
        """Test basic seed setting."""
        set_seed(42)

        # After setting seed, we should get consistent results
        value1 = random.random()
        np_value1 = np.random.random()
        torch_value1 = torch.rand(1).item()

        set_seed(42)

        value2 = random.random()
        np_value2 = np.random.random()
        torch_value2 = torch.rand(1).item()

        assert value1 == value2
        assert np_value1 == np_value2
        assert torch_value1 == torch_value2

    def test_set_seed_python_random(self):
        """Test that Python random module is seeded."""
        set_seed(123)
        values1 = [random.random() for _ in range(5)]

        set_seed(123)
        values2 = [random.random() for _ in range(5)]

        assert values1 == values2

    def test_set_seed_numpy(self):
        """Test that NumPy random is seeded."""
        set_seed(456)
        values1 = np.random.random(5).tolist()

        set_seed(456)
        values2 = np.random.random(5).tolist()

        assert values1 == values2

    def test_set_seed_torch(self):
        """Test that PyTorch random is seeded."""
        set_seed(789)
        values1 = torch.rand(5).tolist()

        set_seed(789)
        values2 = torch.rand(5).tolist()

        assert values1 == values2

    def test_different_seeds_produce_different_values(self):
        """Test that different seeds produce different random values."""
        set_seed(100)
        value1 = random.random()

        set_seed(200)
        value2 = random.random()

        assert value1 != value2

    @pytest.mark.parametrize("seed", [0, 1, 42, 12345, 2**31 - 1])
    def test_set_seed_various_values(self, seed):
        """Test set_seed with various seed values."""
        set_seed(seed)
        value1 = random.random()

        set_seed(seed)
        value2 = random.random()

        assert value1 == value2

    def test_set_seed_deterministic_mode(self):
        """Test set_seed with deterministic=True."""
        set_seed(42, deterministic=True, benchmark=False)

        # Should not raise
        value = torch.rand(1).item()
        assert isinstance(value, float)

    def test_set_seed_benchmark_mode(self):
        """Test set_seed with benchmark=True."""
        set_seed(42, deterministic=False, benchmark=True)

        # Should not raise
        value = torch.rand(1).item()
        assert isinstance(value, float)


class TestGetRandomState:
    """Test suite for get_random_state function."""

    def test_get_random_state_returns_dict(self):
        """Test that get_random_state returns a dictionary."""
        state = get_random_state()

        assert isinstance(state, dict)

    def test_get_random_state_has_required_keys(self):
        """Test that state dict has required keys."""
        state = get_random_state()

        assert "random" in state
        assert "numpy" in state
        assert "torch" in state
        assert "torch_cuda" in state

    def test_get_random_state_python(self):
        """Test that Python random state is captured."""
        state = get_random_state()

        assert state["random"] is not None
        assert isinstance(state["random"], tuple)

    def test_get_random_state_numpy(self):
        """Test that NumPy random state is captured."""
        state = get_random_state()

        assert state["numpy"] is not None

    def test_get_random_state_torch(self):
        """Test that PyTorch random state is captured."""
        state = get_random_state()

        assert state["torch"] is not None
        assert isinstance(state["torch"], torch.Tensor)


class TestSetRandomState:
    """Test suite for set_random_state function."""

    def test_set_random_state_restores_python(self):
        """Test that Python random state is restored."""
        set_seed(42)
        state = get_random_state()

        # Generate some random values to change state
        _ = [random.random() for _ in range(100)]

        set_random_state(state)
        value1 = random.random()

        set_seed(42)
        value2 = random.random()

        assert value1 == value2

    def test_set_random_state_restores_numpy(self):
        """Test that NumPy random state is restored."""
        set_seed(42)
        state = get_random_state()

        # Generate some random values to change state
        _ = np.random.random(100)

        set_random_state(state)
        value1 = np.random.random()

        set_seed(42)
        value2 = np.random.random()

        assert value1 == value2

    def test_set_random_state_restores_torch(self):
        """Test that PyTorch random state is restored."""
        set_seed(42)
        state = get_random_state()

        # Generate some random values to change state
        _ = torch.rand(100)

        set_random_state(state)
        value1 = torch.rand(1).item()

        set_seed(42)
        value2 = torch.rand(1).item()

        assert value1 == value2

    def test_get_set_random_state_roundtrip(self):
        """Test that get/set random state preserves full state."""
        set_seed(12345)

        # Generate some random values
        _ = [random.random() for _ in range(10)]
        _ = np.random.random(10)
        _ = torch.rand(10)

        # Capture state
        state = get_random_state()

        # Generate more values
        expected_python = random.random()
        expected_numpy = np.random.random()
        expected_torch = torch.rand(1).item()

        # Restore state
        set_random_state(state)

        # Values should match
        assert random.random() == expected_python
        assert np.random.random() == expected_numpy
        assert torch.rand(1).item() == expected_torch


class TestMakeDeterministic:
    """Test suite for make_deterministic function."""

    def test_make_deterministic_returns_seed(self):
        """Test that make_deterministic returns an integer seed."""
        seed = make_deterministic(42)

        assert isinstance(seed, int)
        assert seed == 42

    def test_make_deterministic_with_none(self):
        """Test make_deterministic with None generates a seed."""
        seed = make_deterministic(None)

        assert isinstance(seed, int)
        assert seed >= 0

    def test_make_deterministic_reproducibility(self):
        """Test that make_deterministic ensures reproducibility."""
        seed = make_deterministic(42)

        values1 = {
            "python": random.random(),
            "numpy": np.random.random(),
            "torch": torch.rand(1).item(),
        }

        make_deterministic(seed)

        values2 = {
            "python": random.random(),
            "numpy": np.random.random(),
            "torch": torch.rand(1).item(),
        }

        assert values1 == values2

    def test_make_deterministic_different_runs(self):
        """Test that different seeds produce different results."""
        make_deterministic(100)
        value1 = random.random()

        make_deterministic(200)
        value2 = random.random()

        assert value1 != value2

    @pytest.mark.parametrize("seed", [0, 1, 42, 9999])
    def test_make_deterministic_various_seeds(self, seed):
        """Test make_deterministic with various seed values."""
        returned_seed = make_deterministic(seed)

        assert returned_seed == seed


class TestReproducibility:
    """Test reproducibility across complex operations."""

    def test_model_weight_initialization(self):
        """Test reproducible model weight initialization."""
        set_seed(42)
        model1 = torch.nn.Linear(10, 5)
        weights1 = model1.weight.clone()

        set_seed(42)
        model2 = torch.nn.Linear(10, 5)
        weights2 = model2.weight.clone()

        assert torch.allclose(weights1, weights2)

    def test_numpy_array_operations(self):
        """Test reproducible NumPy operations."""
        set_seed(42)
        arr1 = np.random.randn(100, 100)
        result1 = np.mean(arr1)

        set_seed(42)
        arr2 = np.random.randn(100, 100)
        result2 = np.mean(arr2)

        assert result1 == result2

    def test_torch_tensor_operations(self):
        """Test reproducible PyTorch tensor operations."""
        set_seed(42)
        tensor1 = torch.randn(100, 100)
        result1 = tensor1.mean().item()

        set_seed(42)
        tensor2 = torch.randn(100, 100)
        result2 = tensor2.mean().item()

        assert result1 == result2

    def test_mixed_random_sources(self):
        """Test reproducibility with mixed random sources."""
        set_seed(42)

        # Use all random sources
        py_vals1 = [random.randint(0, 100) for _ in range(5)]
        np_vals1 = np.random.randint(0, 100, 5).tolist()
        torch_vals1 = torch.randint(0, 100, (5,)).tolist()

        set_seed(42)

        py_vals2 = [random.randint(0, 100) for _ in range(5)]
        np_vals2 = np.random.randint(0, 100, 5).tolist()
        torch_vals2 = torch.randint(0, 100, (5,)).tolist()

        assert py_vals1 == py_vals2
        assert np_vals1 == np_vals2
        assert torch_vals1 == torch_vals2
