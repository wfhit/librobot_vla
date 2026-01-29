"""Checkpoint utilities for LibroBot."""

import json
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from omegaconf import DictConfig, OmegaConf

from librobot.utils.logging import get_logger

logger = get_logger(__name__)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[Any],
    epoch: int,
    step: int,
    config: Optional[DictConfig],
    save_path: Path,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Save training checkpoint.

    Args:
        model: Model to save
        optimizer: Optimizer state
        scheduler: Scheduler state
        epoch: Current epoch
        step: Current step
        config: Training configuration
        save_path: Path to save checkpoint
        metadata: Optional additional metadata
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "step": step,
    }

    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    if config is not None:
        checkpoint["config"] = OmegaConf.to_container(config, resolve=True)

    if metadata is not None:
        checkpoint["metadata"] = metadata

    torch.save(checkpoint, save_path)
    logger.info(f"Checkpoint saved to {save_path}")


def load_checkpoint(
    checkpoint_path: Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    strict: bool = True,
) -> Dict[str, Any]:
    """Load training checkpoint.

    Args:
        checkpoint_path: Path to checkpoint
        model: Model to load state into
        optimizer: Optional optimizer to load state into
        scheduler: Optional scheduler to load state into
        strict: Whether to strictly enforce key matching

    Returns:
        Dictionary with epoch, step, and metadata
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Load model state
    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)

    # Load optimizer state
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Load scheduler state
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    logger.info(f"Checkpoint loaded from {checkpoint_path}")

    return {
        "epoch": checkpoint.get("epoch", 0),
        "step": checkpoint.get("step", 0),
        "metadata": checkpoint.get("metadata", {}),
        "config": checkpoint.get("config", None),
    }


def save_model(
    model: torch.nn.Module,
    save_path: Path,
    config: Optional[DictConfig] = None,
) -> None:
    """Save model only (no optimizer/scheduler state).

    Args:
        model: Model to save
        save_path: Path to save model
        config: Optional model configuration
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Save model state
    torch.save(model.state_dict(), save_path)

    # Save config separately
    if config is not None:
        config_path = save_path.parent / f"{save_path.stem}_config.yaml"
        OmegaConf.save(config, config_path)

    logger.info(f"Model saved to {save_path}")


def load_model(
    model: torch.nn.Module,
    checkpoint_path: Path,
    strict: bool = True,
) -> torch.nn.Module:
    """Load model from checkpoint.

    Args:
        model: Model to load state into
        checkpoint_path: Path to checkpoint
        strict: Whether to strictly enforce key matching

    Returns:
        Model with loaded state
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    state_dict = torch.load(checkpoint_path, map_location="cpu")

    # Handle case where checkpoint contains full checkpoint dict
    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]

    model.load_state_dict(state_dict, strict=strict)
    logger.info(f"Model loaded from {checkpoint_path}")

    return model
