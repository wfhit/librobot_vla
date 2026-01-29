"""Test configuration system."""

import tempfile
from pathlib import Path

import pytest
from omegaconf import DictConfig

from librobot.utils.config import (
    config_to_dict,
    dict_to_config,
    load_config,
    merge_configs,
    save_config,
)


def test_load_config():
    """Test loading config from YAML."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("model:\n  lr: 0.001\n  batch_size: 32\n")
        f.flush()
        config_path = f.name

    try:
        config = load_config(config_path)
        assert isinstance(config, DictConfig)
        assert config.model.lr == 0.001
        assert config.model.batch_size == 32
    finally:
        Path(config_path).unlink()


def test_load_config_with_overrides():
    """Test config loading with CLI overrides."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("model:\n  lr: 0.001\n")
        f.flush()
        config_path = f.name

    try:
        config = load_config(config_path, overrides=["model.lr=0.01", "model.batch_size=64"])
        assert config.model.lr == 0.01
        assert config.model.batch_size == 64
    finally:
        Path(config_path).unlink()


def test_merge_configs():
    """Test merging multiple configs."""
    config1 = dict_to_config({"model": {"lr": 0.001, "batch_size": 32}})
    config2 = dict_to_config({"model": {"lr": 0.01}, "training": {"epochs": 100}})

    merged = merge_configs(config1, config2)
    assert merged.model.lr == 0.01  # Overridden
    assert merged.model.batch_size == 32  # Preserved
    assert merged.training.epochs == 100  # New key


def test_save_and_load_config():
    """Test saving and loading config."""
    config = dict_to_config({"model": {"lr": 0.001}})

    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "config.yaml"
        save_config(config, config_path)

        loaded = load_config(config_path)
        assert loaded.model.lr == 0.001


def test_config_to_dict():
    """Test converting config to dict."""
    config = dict_to_config({"model": {"lr": 0.001}})
    d = config_to_dict(config)

    assert isinstance(d, dict)
    assert d["model"]["lr"] == 0.001
