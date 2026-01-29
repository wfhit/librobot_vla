"""Inference script for LibroBot VLA models."""

import argparse

from librobot.utils import load_config, setup_logging

logger = setup_logging()


def main():
    """Main inference entry point."""
    parser = argparse.ArgumentParser(description="Run VLA model inference")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to config file (optional)",
    )
    args = parser.parse_args()

    logger.info(f"Loading checkpoint from {args.checkpoint}")

    # TODO: Implement inference logic
    logger.info("Inference logic not yet implemented")


if __name__ == "__main__":
    main()
