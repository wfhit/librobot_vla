#!/usr/bin/env python3
"""
Inference entry point for LibroBot VLA models.

This script provides a command-line interface for running inference with VLA models.
It supports two modes:
1. Single inference: Run inference on input data and return predictions
2. Server mode: Start REST or gRPC server for continuous inference

Features:
- Load models from checkpoints
- Single inference on images/text/state
- REST API server with FastAPI
- gRPC server for high-performance inference
- Batch inference on directories
- Save predictions to file

Example usage:
    # Single inference with image and text
    python scripts/inference.py --checkpoint checkpoints/best.pt \\
        --image path/to/image.jpg \\
        --text "Pick up the red block"

    # Start REST API server
    python scripts/inference.py --checkpoint checkpoints/best.pt \\
        --server rest --host 0.0.0.0 --port 8000

    # Start gRPC server
    python scripts/inference.py --checkpoint checkpoints/best.pt \\
        --server grpc --host 0.0.0.0 --port 50051

    # Batch inference on directory
    python scripts/inference.py --checkpoint checkpoints/best.pt \\
        --config configs/experiment/my_experiment.yaml \\
        --batch-dir path/to/images/ \\
        --output-file predictions.json

    # Server with custom config
    python scripts/inference.py --checkpoint checkpoints/best.pt \\
        --config configs/experiment/my_experiment.yaml \\
        --server rest --port 8000 --enable-cors
"""

import argparse
import json
import signal
import sys
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import torch
from PIL import Image

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from librobot.inference import VLAPolicy
from librobot.inference.server import GRPCServer, RESTServer
from librobot.models import create_vla as build_model
from librobot.utils.config import Config
from librobot.utils.logging import get_logger, setup_logging


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run inference with LibroBot VLA models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Required arguments
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")

    # Configuration
    parser.add_argument(
        "--config", type=str, default=None, help="Path to configuration YAML file (optional)"
    )

    # Inference mode
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--server", type=str, choices=["rest", "grpc"], help="Start inference server (rest or grpc)"
    )
    mode_group.add_argument(
        "--batch-dir", type=str, help="Directory containing images for batch inference"
    )

    # Single inference inputs
    parser.add_argument("--image", type=str, help="Path to input image for single inference")
    parser.add_argument("--text", type=str, help="Text instruction for single inference")
    parser.add_argument(
        "--state", type=str, help="Robot state as JSON string or file path for single inference"
    )

    # Server options
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Server host address (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Server port (default: 8000 for REST, 50051 for gRPC)",
    )
    parser.add_argument("--enable-cors", action="store_true", help="Enable CORS for REST server")
    parser.add_argument(
        "--workers", type=int, default=1, help="Number of worker processes for server"
    )

    # Output options
    parser.add_argument(
        "--output-file", type=str, default=None, help="Output file for predictions (JSON)"
    )
    parser.add_argument(
        "--save-visualization", action="store_true", help="Save visualization of predictions"
    )

    # Device options
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "cpu", "mps"],
        help="Device to run inference on",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="fp32",
        choices=["fp32", "fp16", "bf16"],
        help="Inference precision (default: fp32)",
    )

    # Performance options
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for batch inference")
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Compile model with torch.compile for faster inference",
    )

    # Logging
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level",
    )
    parser.add_argument("--log-file", type=str, default=None, help="Path to log file")

    return parser.parse_args()


def load_model_for_inference(
    checkpoint_path: str, config: Optional[Config], device: str, precision: str
) -> nn.Module:
    """
    Load model from checkpoint for inference.

    Args:
        checkpoint_path: Path to checkpoint file
        config: Configuration object (optional)
        device: Device to load model on
        precision: Inference precision (fp32/fp16/bf16)

    Returns:
        Loaded model ready for inference
    """
    logger = get_logger(__name__)

    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    logger.info(f"Loading checkpoint: {checkpoint_path}")

    # Build model
    if config:
        model = build_model(config.get("model", {}))
    else:
        # Try to infer model from checkpoint
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if "config" in checkpoint:
            model = build_model(checkpoint["config"].get("model", {}))
        else:
            raise ValueError("No config provided and checkpoint doesn't contain config")

    # Load weights
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint)

    # Set precision
    dtype_map = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    dtype = dtype_map[precision]

    # Move to device and set precision
    model.to(device=device, dtype=dtype)
    model.eval()

    logger.info(f"Model loaded on {device} with {precision} precision")
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    return model


def load_image(image_path: Union[str, Path]) -> np.ndarray:
    """
    Load and preprocess image.

    Args:
        image_path: Path to image file

    Returns:
        Preprocessed image as numpy array
    """
    image = Image.open(image_path).convert("RGB")
    return np.array(image)


def run_single_inference(
    model: nn.Module,
    image: Optional[np.ndarray],
    text: Optional[str],
    state: Optional[np.ndarray],
    device: str,
) -> dict[str, Any]:
    """
    Run single inference.

    Args:
        model: Model for inference
        image: Input image (optional)
        text: Input text (optional)
        state: Robot state (optional)
        device: Device to run on

    Returns:
        Prediction dictionary
    """
    logger = get_logger(__name__)

    # Prepare inputs
    inputs = {}

    if image is not None:
        # Convert to tensor and add batch dimension
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        image_tensor = image_tensor.unsqueeze(0).to(device)
        inputs["image"] = image_tensor

    if text is not None:
        inputs["text"] = [text]  # Batch of 1

    if state is not None:
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
        inputs["state"] = state_tensor

    # Run inference
    with torch.no_grad():
        outputs = model(inputs)

    # Process outputs
    if isinstance(outputs, dict):
        actions = outputs.get("actions", outputs.get("logits", None))
    else:
        actions = outputs

    if actions is not None:
        actions = actions.cpu().numpy()[0]  # Remove batch dimension

    result = {
        "actions": actions.tolist() if actions is not None else None,
        "inputs": {
            "has_image": image is not None,
            "has_text": text is not None,
            "has_state": state is not None,
        },
    }

    logger.debug(f"Inference result: {result}")

    return result


def run_batch_inference(
    model: nn.Module, batch_dir: Path, batch_size: int, device: str, output_file: Optional[str]
) -> list[dict[str, Any]]:
    """
    Run batch inference on directory of images.

    Args:
        model: Model for inference
        batch_dir: Directory containing images
        batch_size: Batch size for inference
        device: Device to run on
        output_file: Output file for predictions (optional)

    Returns:
        List of predictions
    """
    logger = get_logger(__name__)

    # Find all images
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    image_paths = [p for p in batch_dir.iterdir() if p.suffix.lower() in image_extensions]

    logger.info(f"Found {len(image_paths)} images in {batch_dir}")

    predictions = []

    # Process in batches
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i : i + batch_size]
        logger.info(
            f"Processing batch {i // batch_size + 1}/{(len(image_paths) + batch_size - 1) // batch_size}"
        )

        # Load images
        images = []
        for path in batch_paths:
            try:
                image = load_image(path)
                images.append(image)
            except Exception as e:
                logger.error(f"Failed to load {path}: {e}")
                continue

        if not images:
            continue

        # Run inference on batch
        with torch.no_grad():
            # Stack images and convert to tensor
            images_array = np.stack(images)
            images_tensor = torch.from_numpy(images_array).permute(0, 3, 1, 2).float() / 255.0
            images_tensor = images_tensor.to(device)

            outputs = model({"image": images_tensor})

            if isinstance(outputs, dict):
                actions = outputs.get("actions", outputs.get("logits", None))
            else:
                actions = outputs

            if actions is not None:
                actions = actions.cpu().numpy()

                for j, path in enumerate(batch_paths[: len(actions)]):
                    predictions.append({"image_path": str(path), "actions": actions[j].tolist()})

    logger.info(f"Completed batch inference on {len(predictions)} images")

    # Save predictions if output file specified
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(predictions, f, indent=2)

        logger.info(f"Saved predictions to: {output_path}")

    return predictions


def start_server(
    server_type: str,
    checkpoint_path: str,
    config: Optional[Config],
    host: str,
    port: int,
    device: str,
    precision: str,
    enable_cors: bool,
    workers: int,
):
    """
    Start inference server.

    Args:
        server_type: Type of server (rest or grpc)
        checkpoint_path: Path to model checkpoint
        config: Configuration object
        host: Server host
        port: Server port
        device: Device to run on
        precision: Inference precision
        enable_cors: Enable CORS (REST only)
        workers: Number of worker processes
    """
    logger = get_logger(__name__)

    # Load model
    model = load_model_for_inference(checkpoint_path, config, device, precision)

    # Create policy
    policy = VLAPolicy(model=model, device=device)

    if server_type == "rest":
        # Default port for REST
        if port is None:
            port = 8000

        logger.info(f"Starting REST server on {host}:{port}")
        server = RESTServer(
            host=host,
            port=port,
            model_path=checkpoint_path,
            device=device,
            enable_cors=enable_cors,
        )

        # Setup graceful shutdown
        def signal_handler(sig, frame):
            logger.info("Shutting down server...")
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        server.start()

    elif server_type == "grpc":
        # Default port for gRPC
        if port is None:
            port = 50051

        logger.info(f"Starting gRPC server on {host}:{port}")
        server = GRPCServer(
            host=host,
            port=port,
            model_path=checkpoint_path,
            device=device,
        )

        # Setup graceful shutdown
        def signal_handler(sig, frame):
            logger.info("Shutting down server...")
            server.stop()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        server.start()
        server.wait_for_termination()


def main():
    """Main inference entry point."""
    args = parse_args()

    # Setup logging
    setup_logging(
        level=args.log_level,
        log_file=args.log_file,
        force_colors=True,
    )
    logger = get_logger(__name__)

    try:
        logger.info("=" * 80)
        logger.info("LibroBot VLA Inference")
        logger.info("=" * 80)

        # Load config if provided
        config = None
        if args.config:
            if not Path(args.config).exists():
                raise FileNotFoundError(f"Config file not found: {args.config}")
            config = Config.from_yaml(args.config)
            logger.info(f"Loaded config from: {args.config}")

        # Determine device
        if args.device:
            device = args.device
        elif config and config.get("device") == "cuda" and torch.cuda.is_available():
            device = "cuda"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        logger.info(f"Using device: {device}")

        # Server mode
        if args.server:
            start_server(
                server_type=args.server,
                checkpoint_path=args.checkpoint,
                config=config,
                host=args.host,
                port=args.port,
                device=device,
                precision=args.precision,
                enable_cors=args.enable_cors,
                workers=args.workers,
            )
            return

        # Load model for single or batch inference
        model = load_model_for_inference(
            checkpoint_path=args.checkpoint, config=config, device=device, precision=args.precision
        )

        # Compile model if requested
        if args.compile:
            logger.info("Compiling model with torch.compile...")
            model = torch.compile(model)

        # Batch inference mode
        if args.batch_dir:
            batch_dir = Path(args.batch_dir)
            if not batch_dir.exists():
                raise FileNotFoundError(f"Batch directory not found: {batch_dir}")

            predictions = run_batch_inference(
                model=model,
                batch_dir=batch_dir,
                batch_size=args.batch_size,
                device=device,
                output_file=args.output_file,
            )

            logger.info(f"Processed {len(predictions)} images")
            return

        # Single inference mode
        if not args.image and not args.text and not args.state:
            logger.error("For single inference, provide at least one of: --image, --text, --state")
            sys.exit(1)

        # Load inputs
        image = None
        if args.image:
            image = load_image(args.image)
            logger.info(f"Loaded image: {args.image}")

        text = args.text
        if text:
            logger.info(f"Text instruction: {text}")

        state = None
        if args.state:
            # Try to load as JSON string or file
            try:
                state = json.loads(args.state)
                state = np.array(state, dtype=np.float32)
            except json.JSONDecodeError:
                # Try as file path
                state_path = Path(args.state)
                if state_path.exists():
                    state = np.load(state_path)
                else:
                    raise ValueError(f"Invalid state: {args.state}")
            logger.info(f"Robot state shape: {state.shape}")

        # Run inference
        logger.info("Running inference...")
        result = run_single_inference(
            model=model, image=image, text=text, state=state, device=device
        )

        # Print result
        logger.info("=" * 80)
        logger.info("Inference Result:")
        logger.info("=" * 80)
        if result["actions"] is not None:
            logger.info(f"Predicted actions: {result['actions']}")
        else:
            logger.info("No actions predicted")

        # Save result if output file specified
        if args.output_file:
            output_path = Path(args.output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, "w") as f:
                json.dump(result, f, indent=2)

            logger.info(f"Saved result to: {output_path}")

    except KeyboardInterrupt:
        logger.warning("Inference interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Inference failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
