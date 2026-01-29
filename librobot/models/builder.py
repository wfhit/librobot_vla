"""Model builder utilities for constructing VLA models from configs."""

from typing import Optional

import torch.nn as nn
from omegaconf import DictConfig

from librobot.models.encoders import MLPEncoder
from librobot.utils import REGISTRY
from librobot.utils.logging import get_logger

logger = get_logger(__name__)


def build_action_head(config: DictConfig) -> nn.Module:
    """Build action head from config.

    Args:
        config: Action head configuration

    Returns:
        Action head instance

    Example:
        >>> config = DictConfig({
        ...     "name": "mlp_oft",
        ...     "input_dim": 512,
        ...     "action_dim": 6,
        ...     "hidden_dim": 256,
        ... })
        >>> action_head = build_action_head(config)
    """
    name = config.name
    config_dict = dict(config)
    config_dict.pop("name")

    action_head_cls = REGISTRY.get("action_head", name)
    logger.info(f"Building action head: {name}")

    return action_head_cls(**config_dict)


def build_encoder(config: DictConfig) -> nn.Module:
    """Build encoder from config.

    Args:
        config: Encoder configuration

    Returns:
        Encoder instance
    """
    name = config.name
    config_dict = dict(config)
    config_dict.pop("name")

    encoder_cls = REGISTRY.get("encoder", name)
    logger.info(f"Building encoder: {name}")

    return encoder_cls(**config_dict)


def build_framework(
    config: DictConfig,
    vlm: nn.Module,
    action_head: nn.Module,
    state_encoder: Optional[nn.Module] = None,
    history_encoder: Optional[nn.Module] = None,
) -> nn.Module:
    """Build VLA framework from config.

    Args:
        config: Framework configuration
        vlm: Vision-Language Model
        action_head: Action prediction head
        state_encoder: Optional state encoder
        history_encoder: Optional history encoder

    Returns:
        VLA framework instance

    Example:
        >>> config = DictConfig({"name": "groot_style", "fusion_method": "concat"})
        >>> framework = build_framework(config, vlm, action_head, state_encoder)
    """
    name = config.name
    config_dict = dict(config)
    config_dict.pop("name")

    framework_cls = REGISTRY.get("framework", name)
    logger.info(f"Building framework: {name}")

    return framework_cls(
        vlm=vlm,
        action_head=action_head,
        state_encoder=state_encoder,
        history_encoder=history_encoder,
        **config_dict,
    )


def build_robot(config: DictConfig):
    """Build robot from config.

    Args:
        config: Robot configuration

    Returns:
        Robot instance

    Example:
        >>> config = DictConfig({"name": "wheel_loader"})
        >>> robot = build_robot(config)
    """
    name = config.name
    robot_cls = REGISTRY.get("robot", name)
    logger.info(f"Building robot: {name}")

    return robot_cls()


def build_model_from_config(config: DictConfig, vlm: nn.Module):
    """Build complete VLA model from config.

    This is a high-level builder that constructs all components.

    Args:
        config: Full model configuration
        vlm: Pretrained VLM to use

    Returns:
        Tuple of (framework, robot)

    Example:
        >>> from librobot.utils import load_config
        >>> config = load_config("configs/experiment/wheel_loader_groot.yaml")
        >>> framework, robot = build_model_from_config(config.model, vlm)
    """
    logger.info("Building VLA model from config")

    # Build robot
    robot = build_robot(config.robot if hasattr(config, "robot") else config)

    # Build state encoder if specified
    state_encoder = None
    if hasattr(config, "state_encoder") and config.state_encoder is not None:
        # Add robot state_dim if not in config
        if "input_dim" not in config.state_encoder:
            config.state_encoder.input_dim = robot.state_dim
        state_encoder = build_encoder(config.state_encoder)

    # Build history encoder if specified
    history_encoder = None
    if hasattr(config, "history_encoder") and config.history_encoder is not None:
        history_encoder = build_encoder(config.history_encoder)

    # Build action head
    action_head_config = config.action_head
    if "action_dim" not in action_head_config:
        action_head_config.action_dim = robot.action_dim

    # Calculate input dim for action head
    input_dim = vlm.hidden_dim
    if state_encoder is not None:
        if hasattr(config.framework, "fusion_method"):
            if config.framework.fusion_method == "concat":
                input_dim += state_encoder.output_dim
    if history_encoder is not None and hasattr(config.framework, "fusion_method"):
        if config.framework.fusion_method == "concat":
            input_dim += history_encoder.output_dim

    action_head_config.input_dim = input_dim
    action_head = build_action_head(action_head_config)

    # Build framework
    framework = build_framework(
        config.framework,
        vlm=vlm,
        action_head=action_head,
        state_encoder=state_encoder,
        history_encoder=history_encoder,
    )

    # Freeze VLM if specified
    if hasattr(config.vlm, "freeze") and config.vlm.freeze:
        framework.freeze_vlm()
        logger.info("VLM parameters frozen")

    logger.info("Model building complete")
    params = framework.get_trainable_parameters()
    logger.info(f"Trainable parameters: {params['total']:,}")

    return framework, robot
