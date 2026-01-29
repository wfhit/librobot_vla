#!/usr/bin/env python3
"""
Evaluation entry point for LibroBot VLA models.

This script provides a command-line interface for evaluating trained VLA models.
It supports:
- Loading models from checkpoints
- Evaluating on test datasets
- Computing comprehensive metrics
- Generating evaluation reports
- Saving predictions and visualizations

Example usage:
    # Evaluate on test set
    python scripts/evaluate.py --checkpoint checkpoints/best.pt \\
        --config configs/experiment/my_experiment.yaml

    # Evaluate with custom test data
    python scripts/evaluate.py --checkpoint checkpoints/best.pt \\
        --config configs/experiment/my_experiment.yaml \\
        --test-data path/to/test_data

    # Evaluate and save predictions
    python scripts/evaluate.py --checkpoint checkpoints/best.pt \\
        --config configs/experiment/my_experiment.yaml \\
        --save-predictions --output-dir results/evaluation

    # Evaluate multiple checkpoints
    python scripts/evaluate.py --checkpoint checkpoints/*.pt \\
        --config configs/experiment/my_experiment.yaml \\
        --output-dir results/multi_checkpoint_eval
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import glob

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from librobot.utils.config import Config
from librobot.utils.logging import setup_logging, get_logger
from librobot.utils.checkpoint import Checkpoint
from librobot.utils.seed import set_seed
from librobot.models import create_vla as build_model
from librobot.data.datasets import create_dataset as build_dataset
from librobot.training.losses.base import AbstractLoss


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate LibroBot VLA models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Required arguments
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint file(s). Supports glob patterns (e.g., checkpoints/*.pt)"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration YAML file"
    )
    
    # Data
    parser.add_argument(
        "--test-data",
        type=str,
        default=None,
        help="Path to test dataset. If not specified, uses config value."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size for evaluation. If not specified, uses config value."
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of data loading workers"
    )
    
    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/evaluation",
        help="Directory for evaluation outputs"
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Save model predictions to file"
    )
    parser.add_argument(
        "--save-visualizations",
        action="store_true",
        help="Save visualizations of predictions"
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
    
    # Device
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "cpu", "mps"],
        help="Device to use for evaluation"
    )
    
    # Reproducibility
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    # Metrics
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=None,
        help="Specific metrics to compute (default: all available metrics)"
    )
    
    return parser.parse_args()


def load_checkpoint_for_eval(checkpoint_path: str, model: nn.Module, device: str) -> Dict[str, Any]:
    """
    Load checkpoint and prepare model for evaluation.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load weights into
        device: Device to load model on
        
    Returns:
        Checkpoint metadata dictionary
    """
    logger = get_logger(__name__)
    
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model state
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
        "metrics": checkpoint.get("metrics", {}),
    }
    
    logger.info(f"Loaded checkpoint from epoch {metadata['epoch']}, step {metadata['step']}")
    if metadata["metrics"]:
        logger.info(f"Training metrics: {metadata['metrics']}")
    
    return metadata


def compute_metrics(
    predictions: List[Dict[str, Any]],
    targets: List[Dict[str, Any]],
    metric_names: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Compute evaluation metrics.
    
    Args:
        predictions: List of prediction dictionaries
        targets: List of target dictionaries
        metric_names: Specific metrics to compute (None = all)
        
    Returns:
        Dictionary of metric names to values
    """
    logger = get_logger(__name__)
    metrics = {}
    
    # Extract predictions and targets
    pred_actions = np.array([p["actions"] for p in predictions])
    target_actions = np.array([t["actions"] for t in targets])
    
    # Mean squared error
    if metric_names is None or "mse" in metric_names:
        mse = np.mean((pred_actions - target_actions) ** 2)
        metrics["mse"] = float(mse)
    
    # Mean absolute error
    if metric_names is None or "mae" in metric_names:
        mae = np.mean(np.abs(pred_actions - target_actions))
        metrics["mae"] = float(mae)
    
    # Root mean squared error
    if metric_names is None or "rmse" in metric_names:
        rmse = np.sqrt(np.mean((pred_actions - target_actions) ** 2))
        metrics["rmse"] = float(rmse)
    
    # Per-dimension metrics
    if metric_names is None or "per_dim_mse" in metric_names:
        per_dim_mse = np.mean((pred_actions - target_actions) ** 2, axis=0)
        for i, mse_val in enumerate(per_dim_mse):
            metrics[f"mse_dim_{i}"] = float(mse_val)
    
    # Success rate (within threshold)
    if metric_names is None or "success_rate" in metric_names:
        threshold = 0.1  # Configurable threshold
        per_sample_error = np.mean((pred_actions - target_actions) ** 2, axis=1)
        success_rate = np.mean(per_sample_error < threshold)
        metrics["success_rate"] = float(success_rate)
    
    logger.info("Computed metrics:")
    for name, value in metrics.items():
        logger.info(f"  {name}: {value:.6f}")
    
    return metrics


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: Optional[nn.Module],
    device: str,
    save_predictions: bool = False
) -> tuple[Dict[str, float], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Run evaluation on dataset.
    
    Args:
        model: Model to evaluate
        dataloader: DataLoader for test data
        loss_fn: Loss function (optional)
        device: Device to run evaluation on
        save_predictions: Whether to save predictions
        
    Returns:
        Tuple of (metrics, predictions, targets)
    """
    logger = get_logger(__name__)
    
    model.eval()
    all_losses = []
    predictions = []
    targets = []
    
    logger.info(f"Evaluating on {len(dataloader)} batches...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            # Move batch to device
            if isinstance(batch, dict):
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
            elif isinstance(batch, (list, tuple)):
                batch = [x.to(device) if isinstance(x, torch.Tensor) else x for x in batch]
            
            # Forward pass
            try:
                outputs = model(batch)
                
                # Compute loss if loss function provided
                if loss_fn is not None:
                    if isinstance(outputs, dict):
                        loss = loss_fn(outputs, batch)
                    else:
                        loss = loss_fn(outputs, batch.get("actions", batch[-1]))
                    all_losses.append(loss.item())
                
                # Save predictions if requested
                if save_predictions:
                    # Extract actions from outputs
                    if isinstance(outputs, dict):
                        pred_actions = outputs.get("actions", outputs.get("logits", None))
                    else:
                        pred_actions = outputs
                    
                    if pred_actions is not None:
                        predictions.extend([
                            {"actions": pred_actions[i].cpu().numpy()}
                            for i in range(pred_actions.shape[0])
                        ])
                    
                    # Extract target actions
                    target_actions = batch.get("actions", batch[-1]) if isinstance(batch, dict) else batch[-1]
                    targets.extend([
                        {"actions": target_actions[i].cpu().numpy()}
                        for i in range(target_actions.shape[0])
                    ])
            
            except Exception as e:
                logger.error(f"Error processing batch {batch_idx}: {e}")
                continue
    
    # Compute summary metrics
    metrics = {}
    if all_losses:
        metrics["loss"] = float(np.mean(all_losses))
        metrics["loss_std"] = float(np.std(all_losses))
    
    return metrics, predictions, targets


def save_evaluation_results(
    output_dir: Path,
    metrics: Dict[str, float],
    predictions: Optional[List[Dict[str, Any]]],
    checkpoint_metadata: Dict[str, Any],
    config: Config
):
    """
    Save evaluation results to disk.
    
    Args:
        output_dir: Directory to save results
        metrics: Computed metrics
        predictions: Model predictions (optional)
        checkpoint_metadata: Checkpoint metadata
        config: Configuration object
    """
    logger = get_logger(__name__)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metrics
    metrics_file = output_dir / "metrics.json"
    results = {
        "timestamp": datetime.now().isoformat(),
        "checkpoint": checkpoint_metadata,
        "metrics": metrics,
    }
    
    with open(metrics_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved metrics to: {metrics_file}")
    
    # Save predictions if provided
    if predictions:
        predictions_file = output_dir / "predictions.npz"
        pred_dict = {}
        
        # Group predictions by key
        for key in predictions[0].keys():
            pred_dict[key] = np.array([p[key] for p in predictions])
        
        np.savez(predictions_file, **pred_dict)
        logger.info(f"Saved predictions to: {predictions_file}")
    
    # Save config
    config_file = output_dir / "config.yaml"
    config.save(config_file)
    logger.info(f"Saved config to: {config_file}")
    
    # Generate summary report
    report_file = output_dir / "evaluation_report.txt"
    with open(report_file, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("LibroBot VLA Evaluation Report\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Timestamp: {results['timestamp']}\n")
        f.write(f"Checkpoint: {checkpoint_metadata['checkpoint_path']}\n")
        f.write(f"Epoch: {checkpoint_metadata.get('epoch', 'N/A')}\n")
        f.write(f"Step: {checkpoint_metadata.get('step', 'N/A')}\n\n")
        
        f.write("Metrics:\n")
        f.write("-" * 80 + "\n")
        for name, value in sorted(metrics.items()):
            f.write(f"{name:30s}: {value:.6f}\n")
        
        if predictions:
            f.write(f"\nTotal predictions: {len(predictions)}\n")
    
    logger.info(f"Saved report to: {report_file}")


def main():
    """Main evaluation entry point."""
    args = parse_args()
    
    # Setup logging
    log_file = args.log_file
    if log_file is None:
        log_file = Path(args.output_dir) / "evaluate.log"
    
    setup_logging(
        level=args.log_level,
        log_file=log_file,
        force_colors=True,
    )
    logger = get_logger(__name__)
    
    try:
        logger.info("=" * 80)
        logger.info("LibroBot VLA Evaluation")
        logger.info("=" * 80)
        
        # Set seed
        set_seed(args.seed)
        logger.info(f"Set random seed: {args.seed}")
        
        # Load configuration
        if not Path(args.config).exists():
            raise FileNotFoundError(f"Config file not found: {args.config}")
        
        config = Config.from_yaml(args.config)
        logger.info(f"Loaded config from: {args.config}")
        
        # Determine device
        if args.device:
            device = args.device
        elif config.get("device") == "cuda" and torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
        logger.info(f"Using device: {device}")
        
        # Find checkpoint files
        checkpoint_paths = glob.glob(args.checkpoint)
        if not checkpoint_paths:
            raise FileNotFoundError(f"No checkpoints found matching: {args.checkpoint}")
        
        checkpoint_paths = sorted(checkpoint_paths)
        logger.info(f"Found {len(checkpoint_paths)} checkpoint(s) to evaluate")
        
        # Build model
        logger.info("Building model...")
        model = build_model(config.get("model", {}))
        logger.info(f"Model created: {model.__class__.__name__}")
        
        # Build test dataset
        logger.info("Loading test dataset...")
        if args.test_data:
            test_config = {"path": args.test_data}
        else:
            test_config = config.get("dataset.test", config.get("dataset.val", {}))
        
        test_dataset = build_dataset(test_config)
        logger.info(f"Test samples: {len(test_dataset)}")
        
        # Build dataloader
        batch_size = args.batch_size or config.get("training.batch_size", 32)
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True if device == "cuda" else False,
        )
        
        # Build loss function (optional for evaluation)
        loss_fn = None
        # Note: Loss function can be added if available in config
        # if "loss" in config:
        #     loss_fn = build_loss_from_config(config.get("loss", {}))
        #     loss_fn.to(device)
        
        # Evaluate each checkpoint
        for checkpoint_path in checkpoint_paths:
            logger.info("=" * 80)
            logger.info(f"Evaluating checkpoint: {checkpoint_path}")
            logger.info("=" * 80)
            
            # Load checkpoint
            checkpoint_metadata = load_checkpoint_for_eval(checkpoint_path, model, device)
            
            # Run evaluation
            eval_metrics, predictions, targets = evaluate_model(
                model=model,
                dataloader=test_dataloader,
                loss_fn=loss_fn,
                device=device,
                save_predictions=args.save_predictions
            )
            
            # Compute additional metrics from predictions
            if predictions and targets:
                additional_metrics = compute_metrics(
                    predictions=predictions,
                    targets=targets,
                    metric_names=args.metrics
                )
                eval_metrics.update(additional_metrics)
            
            # Prepare output directory
            if len(checkpoint_paths) == 1:
                output_dir = Path(args.output_dir)
            else:
                checkpoint_name = Path(checkpoint_path).stem
                output_dir = Path(args.output_dir) / checkpoint_name
            
            # Save results
            save_evaluation_results(
                output_dir=output_dir,
                metrics=eval_metrics,
                predictions=predictions if args.save_predictions else None,
                checkpoint_metadata=checkpoint_metadata,
                config=config
            )
            
            logger.info(f"Results saved to: {output_dir}")
        
        logger.info("=" * 80)
        logger.info("Evaluation completed successfully!")
        logger.info("=" * 80)
    
    except KeyboardInterrupt:
        logger.warning("Evaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Evaluation failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
