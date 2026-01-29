"""Configuration system for LibroBot using OmegaConf."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from omegaconf import DictConfig, OmegaConf


def load_config(
    config_path: Union[str, Path],
    overrides: Optional[List[str]] = None,
    resolve: bool = True,
) -> DictConfig:
    """Load configuration from YAML file with optional CLI overrides.

    Args:
        config_path: Path to YAML config file
        overrides: Optional list of CLI overrides (e.g., ["model.lr=1e-4", "batch_size=32"])
        resolve: Whether to resolve interpolations

    Returns:
        OmegaConf DictConfig object

    Example:
        >>> config = load_config("configs/experiment/wheel_loader.yaml")
        >>> config = load_config("configs/defaults.yaml", overrides=["model.lr=1e-3"])
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Load base config
    config = OmegaConf.load(config_path)

    # Apply overrides
    if overrides:
        override_config = OmegaConf.from_dotlist(overrides)
        config = OmegaConf.merge(config, override_config)

    # Resolve interpolations
    if resolve:
        OmegaConf.resolve(config)

    return config


def merge_configs(*configs: Union[DictConfig, Dict[str, Any]]) -> DictConfig:
    """Merge multiple configs with later configs taking precedence.

    Args:
        *configs: Variable number of configs to merge

    Returns:
        Merged DictConfig

    Example:
        >>> base_config = load_config("configs/defaults.yaml")
        >>> model_config = load_config("configs/model/qwen.yaml")
        >>> config = merge_configs(base_config, model_config)
    """
    return OmegaConf.merge(*configs)


def save_config(config: DictConfig, save_path: Union[str, Path]) -> None:
    """Save configuration to YAML file.

    Args:
        config: Configuration to save
        save_path: Path to save file
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "w") as f:
        OmegaConf.save(config, f)


def config_to_dict(config: DictConfig) -> Dict[str, Any]:
    """Convert DictConfig to plain dict.

    Args:
        config: DictConfig to convert

    Returns:
        Plain dictionary
    """
    return OmegaConf.to_container(config, resolve=True)


def dict_to_config(d: Dict[str, Any]) -> DictConfig:
    """Convert plain dict to DictConfig.

    Args:
        d: Dictionary to convert

    Returns:
        DictConfig
    """
    return OmegaConf.create(d)
