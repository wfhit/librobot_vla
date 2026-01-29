"""
Pytest configuration and shared fixtures for librobot_vla tests.

This module provides common fixtures and configuration for all tests.
"""

import pytest
import torch
import numpy as np
import tempfile
from pathlib import Path


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU (deselect with '-m \"not gpu\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks integration tests"
    )
    config.addinivalue_line(
        "markers", "benchmark: marks benchmark tests"
    )


# ============================================================================
# Common Fixtures
# ============================================================================

@pytest.fixture(scope="session")
def device():
    """Get PyTorch device for testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="session")
def random_seed():
    """Set random seed for reproducibility."""
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed


@pytest.fixture
def tmp_dir():
    """Create temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# ============================================================================
# Model Fixtures
# ============================================================================

@pytest.fixture
def model_config():
    """Standard model configuration for testing."""
    return {
        "hidden_size": 256,
        "num_layers": 4,
        "num_heads": 8,
        "intermediate_size": 1024,
        "dropout": 0.1,
        "action_dim": 7,
        "state_dim": 14,
        "vocab_size": 10000
    }


@pytest.fixture
def small_model_config():
    """Small model configuration for faster tests."""
    return {
        "hidden_size": 64,
        "num_layers": 2,
        "num_heads": 4,
        "intermediate_size": 128,
        "dropout": 0.1,
        "action_dim": 7,
        "state_dim": 14,
        "vocab_size": 1000
    }


# ============================================================================
# Data Fixtures
# ============================================================================

@pytest.fixture
def sample_image(device):
    """Generate sample image tensor."""
    return torch.randn(3, 224, 224).to(device)


@pytest.fixture
def sample_batch_images(device):
    """Generate batch of sample images."""
    batch_size = 4
    return torch.randn(batch_size, 3, 224, 224).to(device)


@pytest.fixture
def sample_state(device):
    """Generate sample robot state."""
    return torch.randn(14).to(device)


@pytest.fixture
def sample_action(device):
    """Generate sample robot action."""
    return torch.randn(7).to(device)


@pytest.fixture
def sample_trajectory(device):
    """Generate sample trajectory."""
    seq_len = 32
    return {
        "observations": torch.randn(seq_len, 3, 224, 224).to(device),
        "states": torch.randn(seq_len, 14).to(device),
        "actions": torch.randn(seq_len, 7).to(device),
        "rewards": torch.randn(seq_len, 1).to(device),
        "dones": torch.zeros(seq_len, dtype=torch.bool).to(device)
    }


@pytest.fixture
def sample_batch(device):
    """Generate sample batch for training/inference."""
    batch_size = 4
    seq_len = 32
    return {
        "observations": torch.randn(batch_size, seq_len, 3, 224, 224).to(device),
        "states": torch.randn(batch_size, seq_len, 14).to(device),
        "actions": torch.randn(batch_size, seq_len, 7).to(device),
        "attention_mask": torch.ones(batch_size, seq_len).to(device)
    }


# ============================================================================
# Training Fixtures
# ============================================================================

@pytest.fixture
def training_config():
    """Standard training configuration."""
    return {
        "num_epochs": 10,
        "batch_size": 32,
        "learning_rate": 1e-4,
        "weight_decay": 0.01,
        "warmup_steps": 1000,
        "max_grad_norm": 1.0,
        "gradient_accumulation_steps": 1,
        "log_interval": 100,
        "eval_interval": 1000,
        "save_interval": 5000
    }


@pytest.fixture
def optimizer_config():
    """Standard optimizer configuration."""
    return {
        "type": "adam",
        "lr": 1e-4,
        "betas": (0.9, 0.999),
        "eps": 1e-8,
        "weight_decay": 0.01
    }


@pytest.fixture
def scheduler_config():
    """Standard scheduler configuration."""
    return {
        "type": "cosine",
        "warmup_steps": 1000,
        "num_training_steps": 100000,
        "num_cycles": 0.5
    }


# ============================================================================
# Robot Fixtures
# ============================================================================

@pytest.fixture
def robot_config():
    """Standard robot configuration."""
    return {
        "robot_type": "franka",
        "control_mode": "position",
        "control_frequency": 20,
        "joint_limits": {
            "lower": [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973],
            "upper": [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973]
        },
        "velocity_limits": [2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100],
        "home_position": [0, -0.785, 0, -2.356, 0, 1.571, 0.785]
    }


# ============================================================================
# Dataset Fixtures
# ============================================================================

@pytest.fixture
def dataset_config(tmp_dir):
    """Standard dataset configuration."""
    return {
        "dataset_path": str(tmp_dir / "dataset"),
        "train_split": 0.8,
        "val_split": 0.1,
        "test_split": 0.1,
        "num_workers": 4,
        "shuffle": True,
        "pin_memory": True
    }


@pytest.fixture
def episode_data():
    """Generate sample episode data."""
    num_steps = 50
    return {
        "observations": {
            "image": np.random.randn(num_steps, 3, 224, 224).astype(np.float32),
            "state": np.random.randn(num_steps, 14).astype(np.float32)
        },
        "actions": np.random.randn(num_steps, 7).astype(np.float32),
        "rewards": np.random.randn(num_steps, 1).astype(np.float32),
        "dones": np.zeros(num_steps, dtype=bool)
    }


# ============================================================================
# Inference Fixtures
# ============================================================================

@pytest.fixture
def inference_config():
    """Standard inference configuration."""
    return {
        "batch_size": 1,
        "max_length": 100,
        "temperature": 1.0,
        "top_k": 50,
        "top_p": 0.95,
        "num_beams": 1,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "precision": "fp32"
    }


# ============================================================================
# Utility Functions
# ============================================================================

def assert_tensor_close(tensor1, tensor2, rtol=1e-5, atol=1e-8):
    """Assert that two tensors are close."""
    assert torch.allclose(tensor1, tensor2, rtol=rtol, atol=atol)


def assert_shape_equal(tensor, expected_shape):
    """Assert that tensor has expected shape."""
    assert tensor.shape == expected_shape, \
        f"Expected shape {expected_shape}, got {tensor.shape}"


def count_parameters(model):
    """Count number of trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
