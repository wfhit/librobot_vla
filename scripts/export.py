"""Model export script for LibroBot VLA models."""

import argparse

from librobot.utils import setup_logging

logger = setup_logging()


def main():
    """Main export entry point."""
    parser = argparse.ArgumentParser(description="Export VLA model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for exported model",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="onnx",
        choices=["onnx", "torchscript"],
        help="Export format",
    )
    args = parser.parse_args()

    logger.info(f"Exporting model from {args.checkpoint} to {args.output}")

    # TODO: Implement export logic
    logger.info("Export logic not yet implemented")


if __name__ == "__main__":
    main()
