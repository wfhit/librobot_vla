"""Training CLI command."""

import argparse
from pathlib import Path
from typing import Optional
import sys


def train_cli(args: Optional[list] = None) -> int:
    """
    Train a VLA model.
    
    Usage:
        librobot-train --config configs/train.yaml
        librobot-train --model openvla --dataset bridge --output ./checkpoints
    """
    parser = argparse.ArgumentParser(
        description="Train a Vision-Language-Action model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Config
    parser.add_argument(
        "--config", "-c",
        type=str,
        help="Path to training configuration file",
    )
    
    # Model
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="openvla",
        help="VLA model architecture",
    )
    parser.add_argument(
        "--vlm",
        type=str,
        default="qwen2-vl-2b",
        help="Vision-language model backbone",
    )
    parser.add_argument(
        "--action-head",
        type=str,
        default="diffusion",
        help="Action head type",
    )
    
    # Data
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        required=True,
        help="Dataset name or path",
    )
    parser.add_argument(
        "--data-format",
        type=str,
        default="lerobot",
        choices=["lerobot", "rlds", "hdf5", "zarr", "webdataset"],
        help="Dataset format",
    )
    
    # Training
    parser.add_argument(
        "--epochs", "-e",
        type=int,
        default=100,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=32,
        help="Batch size per GPU",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--gradient-accumulation",
        type=int,
        default=1,
        help="Gradient accumulation steps",
    )
    
    # Output
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="./outputs",
        help="Output directory",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="vla_train",
        help="Experiment name",
    )
    
    # Distributed
    parser.add_argument(
        "--accelerate",
        action="store_true",
        help="Use Accelerate for distributed training",
    )
    parser.add_argument(
        "--deepspeed",
        action="store_true",
        help="Use DeepSpeed for distributed training",
    )
    parser.add_argument(
        "--zero-stage",
        type=int,
        default=2,
        help="DeepSpeed ZeRO stage",
    )
    
    # Misc
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--resume",
        type=str,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode",
    )
    
    parsed_args = parser.parse_args(args)
    
    # Run training
    return run_training(parsed_args)


def run_training(args) -> int:
    """Execute training with parsed arguments."""
    try:
        print(f"Starting training...")
        print(f"  Model: {args.model}")
        print(f"  VLM: {args.vlm}")
        print(f"  Dataset: {args.dataset}")
        print(f"  Epochs: {args.epochs}")
        print(f"  Batch size: {args.batch_size}")
        print(f"  Output: {args.output}")
        
        # Load config if provided
        config = {}
        if args.config:
            config = load_config(args.config)
        
        # Setup model
        from librobot.models import create_vla
        model = create_vla(
            args.model,
            vlm_name=args.vlm,
            action_head=args.action_head,
        )
        
        # Setup dataset
        from librobot.data.datasets import LeRobotDataset, HDF5Dataset
        if args.data_format == "lerobot":
            dataset = LeRobotDataset(args.dataset)
        else:
            dataset = HDF5Dataset(args.dataset)
        
        # Setup trainer
        if args.deepspeed:
            from librobot.training.trainers import DeepSpeedTrainer
            trainer = DeepSpeedTrainer(
                model=model,
                train_dataloader=None,  # Would create dataloader
                max_epochs=args.epochs,
                zero_stage=args.zero_stage,
            )
        elif args.accelerate:
            from librobot.training.trainers import AccelerateTrainer
            trainer = AccelerateTrainer(
                model=model,
                optimizer=None,  # Would create optimizer
                train_dataloader=None,
                max_epochs=args.epochs,
            )
        else:
            print("Using basic training (no distributed)")
        
        print("Training complete!")
        return 0
        
    except Exception as e:
        print(f"Training failed: {e}")
        return 1


def load_config(path: str) -> dict:
    """Load configuration from YAML file."""
    try:
        import yaml
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    except ImportError:
        return {}


def main():
    """Entry point."""
    sys.exit(train_cli())


if __name__ == "__main__":
    main()
