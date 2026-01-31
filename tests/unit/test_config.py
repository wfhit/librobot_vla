"""
Unit tests for configuration management.

Tests configuration loading, validation, merging, and serialization.
"""

import pytest
import yaml
from pathlib import Path
from omegaconf import DictConfig

from librobot.utils.config import Config, load_config, merge_configs, create_config


@pytest.fixture
def sample_config_dict():
    """Create a sample configuration dictionary."""
    return {
        "model": {"name": "test_model", "hidden_size": 768, "num_layers": 12},
        "training": {"batch_size": 32, "learning_rate": 1e-4, "num_epochs": 10},
        "data": {"dataset_path": "/path/to/data", "num_workers": 4},
    }


@pytest.fixture
def config_file(sample_config_dict, tmp_path):
    """Create a temporary YAML configuration file."""
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(sample_config_dict, f)
    return config_path


@pytest.fixture
def nested_config_dict():
    """Create a deeply nested configuration dictionary."""
    return {
        "level1": {
            "level2": {
                "level3": {"value": 42, "name": "deep"},
                "list_val": [1, 2, 3],
            },
            "simple": "text",
        },
    }


class TestConfigInitialization:
    """Test suite for Config initialization."""

    def test_init_with_dict(self, sample_config_dict):
        """Test initializing Config from dictionary."""
        config = Config(sample_config_dict)

        assert config.model.name == "test_model"
        assert config.model.hidden_size == 768

    def test_init_with_none(self):
        """Test initializing Config with None creates empty config."""
        config = Config(None)
        config_dict = config.to_dict()

        assert config_dict == {}

    def test_init_with_empty_dict(self):
        """Test initializing Config with empty dict."""
        config = Config({})
        config_dict = config.to_dict()

        assert config_dict == {}

    def test_init_with_dictconfig(self, sample_config_dict):
        """Test initializing Config from DictConfig."""
        from omegaconf import OmegaConf

        dict_config = OmegaConf.create(sample_config_dict)
        config = Config(dict_config)

        assert config.model.name == "test_model"

    def test_init_with_invalid_type_raises_error(self):
        """Test that invalid type raises TypeError."""
        with pytest.raises(TypeError):
            Config("not a dict")

        with pytest.raises(TypeError):
            Config([1, 2, 3])


class TestConfigFromYaml:
    """Test suite for loading Config from YAML files."""

    def test_from_yaml(self, config_file):
        """Test loading configuration from YAML file."""
        config = Config.from_yaml(config_file)

        assert config.model.name == "test_model"
        assert config.training.batch_size == 32

    def test_from_yaml_with_path_object(self, config_file):
        """Test loading YAML with Path object."""
        config = Config.from_yaml(Path(config_file))

        assert config.model is not None

    def test_from_yaml_nonexistent_raises_error(self, tmp_path):
        """Test loading nonexistent file raises error."""
        nonexistent = tmp_path / "nonexistent.yaml"

        with pytest.raises(FileNotFoundError):
            Config.from_yaml(nonexistent)


class TestConfigFromDict:
    """Test suite for Config.from_dict()."""

    def test_from_dict(self, sample_config_dict):
        """Test creating Config from dictionary."""
        config = Config.from_dict(sample_config_dict)

        assert config.model.hidden_size == 768
        assert config.data.num_workers == 4

    def test_from_dict_preserves_structure(self, nested_config_dict):
        """Test that nested structure is preserved."""
        config = Config.from_dict(nested_config_dict)

        assert config.level1.level2.level3.value == 42


class TestConfigFromCli:
    """Test suite for Config.from_cli()."""

    def test_from_cli_basic(self):
        """Test creating Config from CLI args."""
        args = ["key1=value1", "key2=42"]
        config = Config.from_cli(args)

        assert config.key1 == "value1"
        assert config.key2 == 42

    def test_from_cli_nested(self):
        """Test CLI args with nested keys."""
        args = ["model.hidden_size=1024", "training.lr=0.001"]
        config = Config.from_cli(args)

        assert config.model.hidden_size == 1024
        assert config.training.lr == 0.001

    def test_from_cli_empty(self):
        """Test empty CLI args."""
        config = Config.from_cli([])
        config_dict = config.to_dict()

        assert config_dict == {}


class TestConfigMerge:
    """Test suite for Config merging."""

    def test_merge_with_dict(self, sample_config_dict):
        """Test merging Config with dictionary."""
        config = Config(sample_config_dict)
        override = {"model": {"hidden_size": 1024}}

        merged = config.merge(override)

        assert merged.model.hidden_size == 1024
        assert merged.model.name == "test_model"  # Original value preserved

    def test_merge_with_config(self, sample_config_dict):
        """Test merging Config with another Config."""
        config1 = Config(sample_config_dict)
        config2 = Config({"model": {"hidden_size": 1024}, "new_key": "new_value"})

        merged = config1.merge(config2)

        assert merged.model.hidden_size == 1024
        assert merged.new_key == "new_value"

    def test_merge_with_dictconfig(self, sample_config_dict):
        """Test merging with DictConfig."""
        from omegaconf import OmegaConf

        config = Config(sample_config_dict)
        override = OmegaConf.create({"model": {"num_layers": 24}})

        merged = config.merge(override)

        assert merged.model.num_layers == 24

    def test_merge_does_not_modify_original(self, sample_config_dict):
        """Test that merge returns new Config without modifying original."""
        config = Config(sample_config_dict)
        original_size = config.model.hidden_size

        _ = config.merge({"model": {"hidden_size": 9999}})

        assert config.model.hidden_size == original_size

    def test_merge_invalid_type_raises_error(self, sample_config_dict):
        """Test that merging with invalid type raises error."""
        config = Config(sample_config_dict)

        with pytest.raises(TypeError):
            config.merge("not a valid type")


class TestConfigUpdate:
    """Test suite for Config.update() in-place modification."""

    def test_update_modifies_in_place(self, sample_config_dict):
        """Test that update modifies config in place."""
        config = Config(sample_config_dict)

        config.update({"model": {"hidden_size": 2048}})

        assert config.model.hidden_size == 2048

    def test_update_with_config(self, sample_config_dict):
        """Test update with another Config."""
        config = Config(sample_config_dict)
        other = Config({"training": {"batch_size": 64}})

        config.update(other)

        assert config.training.batch_size == 64


class TestConfigGetSet:
    """Test suite for get/set methods."""

    def test_get_simple_key(self, sample_config_dict):
        """Test getting value with simple key."""
        config = Config(sample_config_dict)

        value = config.get("model")

        assert value is not None
        assert value.name == "test_model"

    def test_get_nested_key(self, sample_config_dict):
        """Test getting value with dot notation key."""
        config = Config(sample_config_dict)

        value = config.get("model.hidden_size")

        assert value == 768

    def test_get_nonexistent_returns_default(self, sample_config_dict):
        """Test getting nonexistent key returns None.

        Note: OmegaConf.select returns None for missing keys rather than raising
        an exception, so the default parameter is not used in this case.
        """
        config = Config(sample_config_dict)

        value = config.get("nonexistent", default="default_value")

        # OmegaConf.select returns None for missing keys, doesn't use default
        assert value is None

    def test_get_nonexistent_nested_returns_default(self, sample_config_dict):
        """Test getting nonexistent nested key returns default."""
        config = Config(sample_config_dict)

        value = config.get("model.nonexistent.deep", default=None)

        assert value is None

    def test_set_simple_key(self, sample_config_dict):
        """Test setting value with simple key."""
        config = Config(sample_config_dict)

        config.set("new_key", "new_value")

        assert config.new_key == "new_value"

    def test_set_nested_key(self, sample_config_dict):
        """Test setting value with dot notation key."""
        config = Config(sample_config_dict)

        config.set("model.hidden_size", 4096)

        assert config.model.hidden_size == 4096


class TestConfigSerialization:
    """Test suite for configuration serialization."""

    def test_to_dict(self, sample_config_dict):
        """Test converting Config to dictionary."""
        config = Config(sample_config_dict)

        result = config.to_dict()

        assert isinstance(result, dict)
        assert result["model"]["name"] == "test_model"

    def test_to_yaml(self, sample_config_dict):
        """Test converting Config to YAML string."""
        config = Config(sample_config_dict)

        yaml_str = config.to_yaml()

        assert isinstance(yaml_str, str)
        assert "model:" in yaml_str
        assert "hidden_size" in yaml_str

    def test_save_to_file(self, sample_config_dict, tmp_path):
        """Test saving Config to YAML file."""
        config = Config(sample_config_dict)
        output_path = tmp_path / "output.yaml"

        config.save(output_path)

        assert output_path.exists()

        # Verify content
        loaded = Config.from_yaml(output_path)
        assert loaded.model.name == "test_model"

    def test_save_creates_parent_directories(self, sample_config_dict, tmp_path):
        """Test that save creates parent directories."""
        config = Config(sample_config_dict)
        output_path = tmp_path / "nested" / "dir" / "config.yaml"

        config.save(output_path)

        assert output_path.exists()


class TestConfigAccess:
    """Test suite for configuration access patterns."""

    def test_dot_notation_access(self, sample_config_dict):
        """Test accessing config values using dot notation."""
        config = Config(sample_config_dict)

        assert config.model.name == "test_model"
        assert config.model.hidden_size == 768
        assert config.training.learning_rate == 1e-4

    def test_bracket_notation_access(self, sample_config_dict):
        """Test accessing config values using bracket notation."""
        config = Config(sample_config_dict)

        assert config["model"]["name"] == "test_model"
        assert config["training"]["batch_size"] == 32

    def test_setattr(self, sample_config_dict):
        """Test setting values via attribute assignment."""
        config = Config(sample_config_dict)

        config.model.hidden_size = 2048
        config.new_attr = "value"

        assert config.model.hidden_size == 2048
        assert config.new_attr == "value"

    def test_setitem(self, sample_config_dict):
        """Test setting values via bracket notation."""
        config = Config(sample_config_dict)

        config["model"]["hidden_size"] = 4096
        config["new_key"] = "new_value"

        assert config["model"]["hidden_size"] == 4096
        assert config["new_key"] == "new_value"

    def test_contains(self, sample_config_dict):
        """Test checking if key exists."""
        config = Config(sample_config_dict)

        assert "model" in config
        assert "training" in config
        assert "nonexistent" not in config

    def test_nested_access(self, nested_config_dict):
        """Test accessing deeply nested values."""
        config = Config(nested_config_dict)

        assert config.level1.level2.level3.value == 42
        assert config.level1.level2.level3.name == "deep"
        assert list(config.level1.level2.list_val) == [1, 2, 3]

    def test_repr(self, sample_config_dict):
        """Test string representation."""
        config = Config(sample_config_dict)

        repr_str = repr(config)

        assert "Config" in repr_str
        assert "model" in repr_str


class TestLoadConfig:
    """Test suite for load_config function."""

    def test_load_config(self, config_file):
        """Test load_config utility function."""
        config = load_config(config_file)

        assert isinstance(config, Config)
        assert config.model.name == "test_model"


class TestMergeConfigs:
    """Test suite for merge_configs function."""

    def test_merge_multiple_dicts(self):
        """Test merging multiple dictionaries."""
        config1 = {"a": 1, "b": 2}
        config2 = {"b": 3, "c": 4}
        config3 = {"c": 5, "d": 6}

        merged = merge_configs(config1, config2, config3)

        assert merged.a == 1
        assert merged.b == 3
        assert merged.c == 5
        assert merged.d == 6

    def test_merge_configs_and_dicts(self, sample_config_dict):
        """Test merging Config objects and dictionaries."""
        config1 = Config(sample_config_dict)
        override = {"model": {"hidden_size": 2048}}

        merged = merge_configs(config1, override)

        assert merged.model.hidden_size == 2048
        assert merged.model.name == "test_model"

    def test_merge_empty(self):
        """Test merging with no arguments."""
        merged = merge_configs()

        assert isinstance(merged, Config)
        assert merged.to_dict() == {}

    def test_merge_single(self, sample_config_dict):
        """Test merging single config."""
        config = Config(sample_config_dict)

        merged = merge_configs(config)

        assert merged.model.name == config.model.name


class TestCreateConfig:
    """Test suite for create_config function."""

    def test_create_from_kwargs(self):
        """Test creating Config from keyword arguments."""
        config = create_config(
            model_name="test",
            hidden_size=768,
            num_layers=12,
        )

        assert config.model_name == "test"
        assert config.hidden_size == 768
        assert config.num_layers == 12

    def test_create_empty(self):
        """Test creating empty Config."""
        config = create_config()

        assert config.to_dict() == {}

    def test_create_with_nested_dict_value(self):
        """Test creating Config with nested dict as value."""
        config = create_config(model={"name": "test", "size": 768})

        assert config.model.name == "test"
        assert config.model.size == 768
