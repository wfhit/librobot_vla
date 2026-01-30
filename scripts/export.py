#!/usr/bin/env python3
"""
Model export entry point for LibroBot VLA models.

This script provides a command-line interface for exporting trained VLA models
to various deployment formats:
- ONNX: For deployment with ONNX Runtime
- TorchScript: For deployment in C++ or mobile
- TensorRT: For optimized NVIDIA GPU inference
- CoreML: For Apple devices (iOS, macOS)
- OpenVINO: For Intel hardware

Features:
- Load models from checkpoints
- Export to multiple formats
- Optimize exported models
- Validate exported models
- Generate deployment metadata

Example usage:
    # Export to ONNX
    python scripts/export.py --checkpoint checkpoints/best.pt \\
        --format onnx --output models/model.onnx

    # Export to TorchScript
    python scripts/export.py --checkpoint checkpoints/best.pt \\
        --format torchscript --output models/model.pt

    # Export with optimization
    python scripts/export.py --checkpoint checkpoints/best.pt \\
        --format onnx --output models/model.onnx \\
        --optimize --opset-version 14

    # Export to multiple formats
    python scripts/export.py --checkpoint checkpoints/best.pt \\
        --format onnx torchscript \\
        --output-dir models/exports

    # Export with custom input shapes
    python scripts/export.py --checkpoint checkpoints/best.pt \\
        --config configs/experiment/my_experiment.yaml \\
        --format onnx --output models/model.onnx \\
        --input-shape 1,3,224,224
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import warnings

import torch
import torch.nn as nn
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from librobot.utils.config import Config
from librobot.utils.logging import setup_logging, get_logger
from librobot.models import create_vla as build_model


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Export LibroBot VLA models to deployment formats",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Required arguments
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--format",
        type=str,
        nargs="+",
        required=True,
        choices=["onnx", "torchscript", "tensorrt", "coreml", "openvino"],
        help="Export format(s)"
    )

    # Configuration
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration YAML file (optional if checkpoint contains config)"
    )

    # Output
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (for single format export)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/exports",
        help="Output directory (for multiple format export)"
    )

    # Input specification
    parser.add_argument(
        "--input-shape",
        type=str,
        default=None,
        help="Input shape as comma-separated values (e.g., '1,3,224,224')"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for exported model"
    )
    parser.add_argument(
        "--dynamic-axes",
        nargs="+",
        default=None,
        help="Dynamic axes for ONNX export (e.g., 'batch' 'sequence')"
    )

    # ONNX-specific options
    parser.add_argument(
        "--opset-version",
        type=int,
        default=14,
        help="ONNX opset version (default: 14)"
    )
    parser.add_argument(
        "--optimize-onnx",
        action="store_true",
        help="Optimize ONNX model with onnxoptimizer"
    )

    # TorchScript-specific options
    parser.add_argument(
        "--torchscript-method",
        type=str,
        default="trace",
        choices=["trace", "script"],
        help="TorchScript export method (trace or script)"
    )

    # Optimization
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Apply optimization to exported model"
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Apply quantization to exported model"
    )
    parser.add_argument(
        "--half-precision",
        action="store_true",
        help="Export model in half precision (FP16)"
    )

    # Validation
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate exported model against original"
    )
    parser.add_argument(
        "--validation-samples",
        type=int,
        default=10,
        help="Number of samples to use for validation"
    )

    # Device
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cuda", "cpu"],
        help="Device to use for export (default: cpu)"
    )

    # Logging
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level"
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Path to log file"
    )

    return parser.parse_args()


def load_model_for_export(
    checkpoint_path: str,
    config: Optional[Config],
    device: str
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Load model from checkpoint for export.

    Args:
        checkpoint_path: Path to checkpoint file
        config: Configuration object (optional)
        device: Device to load model on

    Returns:
        Tuple of (model, metadata)
    """
    logger = get_logger(__name__)

    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    logger.info(f"Loading checkpoint: {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Build model
    if config:
        model = build_model(config.get("model", {}))
    elif "config" in checkpoint:
        model = build_model(checkpoint["config"].get("model", {}))
    else:
        raise ValueError("No config provided and checkpoint doesn't contain config")

    # Load weights
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    # Extract metadata
    metadata = {
        "checkpoint_path": checkpoint_path,
        "epoch": checkpoint.get("epoch", None),
        "step": checkpoint.get("step", None),
        "model_class": model.__class__.__name__,
        "num_parameters": sum(p.numel() for p in model.parameters()),
    }

    logger.info(f"Model loaded: {metadata['model_class']}")
    logger.info(f"Parameters: {metadata['num_parameters']:,}")

    return model, metadata


def get_dummy_input(
    model: nn.Module,
    batch_size: int,
    input_shape: Optional[str],
    device: str
) -> Dict[str, torch.Tensor]:
    """
    Generate dummy input for model export.

    Args:
        model: Model to export
        batch_size: Batch size
        input_shape: Input shape specification
        device: Device

    Returns:
        Dictionary of dummy inputs
    """
    logger = get_logger(__name__)

    dummy_input = {}

    # Parse input shape if provided
    if input_shape:
        shape = tuple(int(x) for x in input_shape.split(","))
        dummy_input["image"] = torch.randn(*shape).to(device)
        logger.info(f"Using provided input shape: {shape}")
    else:
        # Default shapes
        dummy_input["image"] = torch.randn(batch_size, 3, 224, 224).to(device)
        logger.info(f"Using default image shape: {dummy_input['image'].shape}")

    # Add other modalities with default shapes
    dummy_input["state"] = torch.randn(batch_size, 7).to(device)  # Default robot state dim

    logger.info(f"Generated dummy inputs: {list(dummy_input.keys())}")

    return dummy_input


def export_to_onnx(
    model: nn.Module,
    dummy_input: Dict[str, torch.Tensor],
    output_path: Path,
    opset_version: int,
    dynamic_axes: Optional[List[str]],
    optimize: bool
) -> Dict[str, Any]:
    """
    Export model to ONNX format.

    Args:
        model: Model to export
        dummy_input: Dummy input for tracing
        output_path: Output file path
        opset_version: ONNX opset version
        dynamic_axes: Dynamic axes specification
        optimize: Whether to optimize ONNX model

    Returns:
        Export metadata
    """
    logger = get_logger(__name__)

    try:
        import onnx
        import onnxruntime as ort
    except ImportError:
        raise ImportError("ONNX export requires 'onnx' and 'onnxruntime'. Install with: pip install onnx onnxruntime")

    logger.info(f"Exporting to ONNX (opset version {opset_version})...")

    # Prepare dynamic axes
    dynamic_axes_dict = None
    if dynamic_axes:
        dynamic_axes_dict = {}
        for input_name in dummy_input.keys():
            dynamic_axes_dict[input_name] = {0: dynamic_axes[0]} if dynamic_axes else {0: "batch"}
        dynamic_axes_dict["output"] = {0: dynamic_axes[0]} if dynamic_axes else {0: "batch"}

    # Export to ONNX
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        (dummy_input,),
        str(output_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=list(dummy_input.keys()),
        output_names=["output"],
        dynamic_axes=dynamic_axes_dict,
    )

    logger.info(f"ONNX model exported to: {output_path}")

    # Optimize if requested
    if optimize:
        try:
            from onnxoptimizer import optimize as onnx_optimize

            logger.info("Optimizing ONNX model...")
            onnx_model = onnx.load(str(output_path))
            optimized_model = onnx_optimize(onnx_model)
            onnx.save(optimized_model, str(output_path))
            logger.info("ONNX model optimized")
        except ImportError:
            logger.warning("onnxoptimizer not available, skipping optimization")

    # Verify ONNX model
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    logger.info("ONNX model verified successfully")

    # Get model size
    model_size = output_path.stat().st_size / (1024 * 1024)  # MB

    metadata = {
        "format": "onnx",
        "opset_version": opset_version,
        "file_size_mb": round(model_size, 2),
        "optimized": optimize,
    }

    return metadata


def export_to_torchscript(
    model: nn.Module,
    dummy_input: Dict[str, torch.Tensor],
    output_path: Path,
    method: str,
    optimize: bool
) -> Dict[str, Any]:
    """
    Export model to TorchScript format.

    Args:
        model: Model to export
        dummy_input: Dummy input for tracing
        output_path: Output file path
        method: Export method (trace or script)
        optimize: Whether to optimize TorchScript

    Returns:
        Export metadata
    """
    logger = get_logger(__name__)

    logger.info(f"Exporting to TorchScript (method: {method})...")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if method == "trace":
        # Trace model
        with torch.no_grad():
            traced_model = torch.jit.trace(model, (dummy_input,))

        # Optimize if requested
        if optimize:
            traced_model = torch.jit.optimize_for_inference(traced_model)

        # Save
        traced_model.save(str(output_path))
        logger.info(f"TorchScript model (traced) exported to: {output_path}")

    elif method == "script":
        # Script model
        scripted_model = torch.jit.script(model)

        # Optimize if requested
        if optimize:
            scripted_model = torch.jit.optimize_for_inference(scripted_model)

        # Save
        scripted_model.save(str(output_path))
        logger.info(f"TorchScript model (scripted) exported to: {output_path}")

    # Get model size
    model_size = output_path.stat().st_size / (1024 * 1024)  # MB

    metadata = {
        "format": "torchscript",
        "method": method,
        "file_size_mb": round(model_size, 2),
        "optimized": optimize,
    }

    return metadata


def export_to_tensorrt(
    model: nn.Module,
    dummy_input: Dict[str, torch.Tensor],
    output_path: Path
) -> Dict[str, Any]:
    """
    Export model to TensorRT format.

    Args:
        model: Model to export
        dummy_input: Dummy input for tracing
        output_path: Output file path

    Returns:
        Export metadata
    """
    logger = get_logger(__name__)

    try:
        import torch_tensorrt
    except ImportError:
        raise ImportError("TensorRT export requires 'torch_tensorrt'. Install from: https://github.com/pytorch/TensorRT")

    logger.info("Exporting to TensorRT...")
    logger.warning("TensorRT export is experimental and may not work for all models")

    # Convert to TensorRT
    trt_model = torch_tensorrt.compile(
        model,
        inputs=[torch_tensorrt.Input(shape=v.shape) for v in dummy_input.values()],
        enabled_precisions={torch.float32},
    )

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.jit.save(trt_model, str(output_path))

    logger.info(f"TensorRT model exported to: {output_path}")

    model_size = output_path.stat().st_size / (1024 * 1024)  # MB

    metadata = {
        "format": "tensorrt",
        "file_size_mb": round(model_size, 2),
    }

    return metadata


def validate_export(
    original_model: nn.Module,
    exported_path: Path,
    export_format: str,
    dummy_input: Dict[str, torch.Tensor],
    num_samples: int,
    device: str
) -> Dict[str, float]:
    """
    Validate exported model against original.

    Args:
        original_model: Original PyTorch model
        exported_path: Path to exported model
        export_format: Export format
        dummy_input: Dummy input for testing
        num_samples: Number of samples to test
        device: Device to run on

    Returns:
        Validation metrics
    """
    logger = get_logger(__name__)

    logger.info(f"Validating {export_format} export...")

    errors = []

    with torch.no_grad():
        for i in range(num_samples):
            # Generate random input
            test_input = {
                k: torch.randn_like(v) for k, v in dummy_input.items()
            }

            # Get original output
            original_output = original_model(test_input)
            if isinstance(original_output, dict):
                original_output = original_output.get("actions", original_output.get("logits"))

            # Get exported output
            if export_format == "onnx":
                import onnxruntime as ort
                session = ort.InferenceSession(str(exported_path))
                input_feed = {k: v.cpu().numpy() for k, v in test_input.items()}
                exported_output = session.run(None, input_feed)[0]
                exported_output = torch.from_numpy(exported_output).to(device)

            elif export_format == "torchscript":
                loaded_model = torch.jit.load(str(exported_path))
                loaded_model.to(device)
                exported_output = loaded_model(test_input)
                if isinstance(exported_output, dict):
                    exported_output = exported_output.get("actions", exported_output.get("logits"))

            else:
                logger.warning(f"Validation not implemented for {export_format}")
                return {}

            # Compute error
            error = torch.abs(original_output - exported_output).mean().item()
            errors.append(error)

    metrics = {
        "mean_absolute_error": float(np.mean(errors)),
        "max_absolute_error": float(np.max(errors)),
        "validation_samples": num_samples,
    }

    logger.info(f"Validation results:")
    logger.info(f"  Mean absolute error: {metrics['mean_absolute_error']:.6f}")
    logger.info(f"  Max absolute error: {metrics['max_absolute_error']:.6f}")

    return metrics


def save_export_metadata(
    output_dir: Path,
    model_metadata: Dict[str, Any],
    export_metadata: List[Dict[str, Any]],
    config: Optional[Config]
):
    """
    Save export metadata to file.

    Args:
        output_dir: Output directory
        model_metadata: Model metadata
        export_metadata: Export metadata for each format
        config: Configuration object
    """
    logger = get_logger(__name__)

    metadata = {
        "timestamp": datetime.now().isoformat(),
        "model": model_metadata,
        "exports": export_metadata,
    }

    if config:
        metadata["config"] = config.to_dict()

    metadata_file = output_dir / "export_metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Saved export metadata to: {metadata_file}")


def main():
    """Main export entry point."""
    args = parse_args()

    # Setup logging
    log_file = args.log_file
    if log_file is None:
        output_dir = Path(args.output_dir if not args.output else Path(args.output).parent)
        log_file = output_dir / "export.log"

    setup_logging(
        level=args.log_level,
        log_file=log_file,
        force_colors=True,
    )
    logger = get_logger(__name__)

    try:
        logger.info("=" * 80)
        logger.info("LibroBot VLA Model Export")
        logger.info("=" * 80)

        # Load config if provided
        config = None
        if args.config:
            if not Path(args.config).exists():
                raise FileNotFoundError(f"Config file not found: {args.config}")
            config = Config.from_yaml(args.config)
            logger.info(f"Loaded config from: {args.config}")

        # Load model
        model, model_metadata = load_model_for_export(
            checkpoint_path=args.checkpoint,
            config=config,
            device=args.device
        )

        # Apply half precision if requested
        if args.half_precision:
            logger.info("Converting model to half precision (FP16)...")
            model.half()

        # Generate dummy input
        dummy_input = get_dummy_input(
            model=model,
            batch_size=args.batch_size,
            input_shape=args.input_shape,
            device=args.device
        )

        # Export to each format
        export_metadata_list = []

        for export_format in args.format:
            logger.info("=" * 80)
            logger.info(f"Exporting to {export_format.upper()}")
            logger.info("=" * 80)

            # Determine output path
            if args.output and len(args.format) == 1:
                output_path = Path(args.output)
            else:
                output_dir = Path(args.output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)

                # Default extensions
                extensions = {
                    "onnx": ".onnx",
                    "torchscript": ".pt",
                    "tensorrt": ".trt",
                    "coreml": ".mlmodel",
                    "openvino": ".xml",
                }
                output_path = output_dir / f"model{extensions[export_format]}"

            try:
                if export_format == "onnx":
                    metadata = export_to_onnx(
                        model=model,
                        dummy_input=dummy_input,
                        output_path=output_path,
                        opset_version=args.opset_version,
                        dynamic_axes=args.dynamic_axes,
                        optimize=args.optimize or args.optimize_onnx
                    )

                elif export_format == "torchscript":
                    metadata = export_to_torchscript(
                        model=model,
                        dummy_input=dummy_input,
                        output_path=output_path,
                        method=args.torchscript_method,
                        optimize=args.optimize
                    )

                elif export_format == "tensorrt":
                    metadata = export_to_tensorrt(
                        model=model,
                        dummy_input=dummy_input,
                        output_path=output_path
                    )

                else:
                    logger.warning(f"Export format '{export_format}' not yet implemented")
                    continue

                metadata["output_path"] = str(output_path)
                export_metadata_list.append(metadata)

                # Validate if requested
                if args.validate:
                    validation_metrics = validate_export(
                        original_model=model,
                        exported_path=output_path,
                        export_format=export_format,
                        dummy_input=dummy_input,
                        num_samples=args.validation_samples,
                        device=args.device
                    )
                    metadata["validation"] = validation_metrics

            except Exception as e:
                logger.error(f"Failed to export to {export_format}: {e}")
                continue

        # Save export metadata
        if export_metadata_list:
            output_dir = Path(args.output_dir if not args.output else Path(args.output).parent)
            save_export_metadata(
                output_dir=output_dir,
                model_metadata=model_metadata,
                export_metadata=export_metadata_list,
                config=config
            )

        logger.info("=" * 80)
        logger.info(f"Export completed! Exported to {len(export_metadata_list)} format(s)")
        logger.info("=" * 80)

        for metadata in export_metadata_list:
            logger.info(f"{metadata['format']}: {metadata['output_path']} ({metadata['file_size_mb']} MB)")

    except KeyboardInterrupt:
        logger.warning("Export interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Export failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
