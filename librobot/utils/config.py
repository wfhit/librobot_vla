"""Configuration management using OmegaConf and Hydra."""

from pathlib import Path
from typing import Any, Optional, Union

from omegaconf import DictConfig, OmegaConf


class Config:
    """
    Configuration management wrapper around OmegaConf.

    Examples:
        >>> # Load from file
        >>> config = Config.from_yaml("config.yaml")
        >>>
        >>> # Create from dict
        >>> config = Config({"model": {"hidden_size": 768}})
        >>>
        >>> # Access values
        >>> hidden_size = config.model.hidden_size
        >>>
        >>> # Update values
        >>> config.model.hidden_size = 1024
        >>>
        >>> # Save to file
        >>> config.save("output.yaml")
    """

    def __init__(self, config: Optional[Union[dict, DictConfig]] = None):
        """
        Initialize configuration.

        Args:
            config: Initial configuration dictionary or DictConfig
        """
        if config is None:
            config = {}

        if isinstance(config, dict):
            self._config = OmegaConf.create(config)
        elif isinstance(config, DictConfig):
            self._config = config
        else:
            raise TypeError(f"Expected dict or DictConfig, got {type(config)}")

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "Config":
        """
        Load configuration from YAML file.

        Args:
            path: Path to YAML file

        Returns:
            Config: Configuration instance
        """
        config = OmegaConf.load(path)
        return cls(config)

    @classmethod
    def from_dict(cls, config_dict: dict) -> "Config":
        """
        Create configuration from dictionary.

        Args:
            config_dict: Configuration dictionary

        Returns:
            Config: Configuration instance
        """
        return cls(config_dict)

    @classmethod
    def from_cli(cls, args: Optional[list[str]] = None) -> "Config":
        """
        Create configuration from command line arguments.

        Args:
            args: Command line arguments. If None, uses sys.argv

        Returns:
            Config: Configuration instance
        """
        config = OmegaConf.from_cli(args)
        return cls(config)

    def merge(self, other: Union["Config", dict, DictConfig]) -> "Config":
        """
        Merge with another configuration.

        Args:
            other: Configuration to merge

        Returns:
            Config: New merged configuration
        """
        if isinstance(other, Config):
            other_config = other._config
        elif isinstance(other, dict):
            other_config = OmegaConf.create(other)
        elif isinstance(other, DictConfig):
            other_config = other
        else:
            raise TypeError(f"Expected Config, dict, or DictConfig, got {type(other)}")

        merged = OmegaConf.merge(self._config, other_config)
        return Config(merged)

    def update(self, other: Union["Config", dict, DictConfig]) -> None:
        """
        Update configuration in-place.

        Args:
            other: Configuration to merge
        """
        if isinstance(other, Config):
            other_config = other._config
        elif isinstance(other, dict):
            other_config = OmegaConf.create(other)
        elif isinstance(other, DictConfig):
            other_config = other
        else:
            raise TypeError(f"Expected Config, dict, or DictConfig, got {type(other)}")

        self._config = OmegaConf.merge(self._config, other_config)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value with optional default.

        Args:
            key: Configuration key (supports dot notation)
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        try:
            return OmegaConf.select(self._config, key)
        except Exception:
            return default

    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value.

        Args:
            key: Configuration key (supports dot notation)
            value: Value to set
        """
        OmegaConf.update(self._config, key, value)

    def to_dict(self) -> dict:
        """
        Convert configuration to dictionary.

        Returns:
            Dict: Configuration as dictionary
        """
        return OmegaConf.to_container(self._config, resolve=True)

    def to_yaml(self) -> str:
        """
        Convert configuration to YAML string.

        Returns:
            str: YAML representation
        """
        return OmegaConf.to_yaml(self._config)

    def save(self, path: Union[str, Path]) -> None:
        """
        Save configuration to YAML file.

        Args:
            path: Output file path
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            f.write(self.to_yaml())

    def __getattr__(self, name: str) -> Any:
        """Get attribute from configuration."""
        if name.startswith("_"):
            return object.__getattribute__(self, name)
        return getattr(self._config, name)

    def __setattr__(self, name: str, value: Any) -> None:
        """Set attribute in configuration."""
        if name.startswith("_"):
            object.__setattr__(self, name, value)
        else:
            setattr(self._config, name, value)

    def __getitem__(self, key: str) -> Any:
        """Get item from configuration."""
        return self._config[key]

    def __setitem__(self, key: str, value: Any) -> None:
        """Set item in configuration."""
        self._config[key] = value

    def __contains__(self, key: str) -> bool:
        """Check if key exists in configuration."""
        return key in self._config

    def __repr__(self) -> str:
        """String representation."""
        return f"Config({self.to_yaml()})"


def load_config(path: Union[str, Path]) -> Config:
    """
    Load configuration from file.

    Args:
        path: Path to configuration file

    Returns:
        Config: Configuration instance
    """
    return Config.from_yaml(path)


def merge_configs(*configs: Union[Config, dict, DictConfig]) -> Config:
    """
    Merge multiple configurations.

    Args:
        *configs: Configurations to merge

    Returns:
        Config: Merged configuration
    """
    if not configs:
        return Config()

    result = configs[0] if isinstance(configs[0], Config) else Config(configs[0])

    for config in configs[1:]:
        result = result.merge(config)

    return result


def create_config(**kwargs) -> Config:
    """
    Create configuration from keyword arguments.

    Args:
        **kwargs: Configuration parameters

    Returns:
        Config: Configuration instance
    """
    return Config(kwargs)
