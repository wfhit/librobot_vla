"""Training script for LibroBot VLA models."""

import argparse
from pathlib import Path

import torch

from librobot.utils import load_config, setup_logging, set_seed

logger = setup_logging()


def main():
    """Main training entry point."""
    parser = argparse.ArgumentParser(description="Train VLA model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/defaults.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--overrides",
        nargs="*",
        default=[],
        help="Config overrides (e.g., model.lr=1e-4)",
    )
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config, overrides=args.overrides)
    logger.info(f"Loaded config from {args.config}")

    # Set seed
    set_seed(config.get("seed", 42))
    logger.info(f"Set random seed to {config.seed}")

    # TODO: Implement training logic
    # 1. Build model from config
    # 2. Load dataset
    # 3. Create trainer
    # 4. Train model

    logger.info("Training logic not yet implemented")
    logger.info("This is a placeholder script for the VLA framework structure")


if __name__ == "__main__":
    main()
