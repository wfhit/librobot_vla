"""Evaluation script for LibroBot VLA models."""

import argparse

from librobot.utils import load_config, setup_logging

logger = setup_logging()


def main():
    """Main evaluation entry point."""
    parser = argparse.ArgumentParser(description="Evaluate VLA model")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    logger.info(f"Loaded config from {args.config}")

    # TODO: Implement evaluation logic
    logger.info("Evaluation logic not yet implemented")


if __name__ == "__main__":
    main()
