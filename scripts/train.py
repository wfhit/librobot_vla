#!/usr/bin/env python3
"""
Training entry point for LibroBot VLA models.

This script provides a command-line interface for training vision-language-action
models using the LibroBot framework. It supports:
- Loading configurations from YAML files
- Distributed training (DDP, FSDP, DeepSpeed)
- Mixed precision training
- Checkpoint resumption
- WandB and TensorBoard logging
- Evaluation during training

Example usage:
    # Basic training with default config
    python scripts/train.py

    # Training with custom config
    python scripts/train.py --config configs/experiment/my_experiment.yaml

    # Resume from checkpoint
    python scripts/train.py --config configs/experiment/my_experiment.yaml \\
        --resume checkpoints/checkpoint_epoch_10.pt

    # Override config values from CLI
    python scripts/train.py --config configs/experiment/my_experiment.yaml \\
        --override training.max_epochs=100 model.hidden_size=768

    # Distributed training
    torchrun --nproc_per_node=4 scripts/train.py --config configs/experiment/my_experiment.yaml
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from librobot.utils.config import Config
from librobot.utils.logging import setup_logging, get_logger
from librobot.utils.seed import set_seed
from librobot.utils.checkpoint import Checkpoint
from librobot.training.trainer import Trainer, TrainerConfig
from librobot.training.distributed import setup_distributed, cleanup_distributed, is_distributed
from librobot.data.datasets import create_dataset as build_dataset
from librobot.models import create_vla as build_model


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train LibroBot VLA models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Configuration
    parser.add_argument(
        "--config",
        type=str,
        default="configs/defaults.yaml",
        help="Path to configuration YAML file"
    )
    parser.add_argument(
        "--override",
        nargs="+",
        default=[],
        help="Override config values (e.g., training.max_epochs=100 model.hidden_size=768)"
    )
    
    # Checkpointing
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for outputs (checkpoints, logs). Overrides config value."
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
        help="Path to log file (default: output_dir/train.log)"
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable Weights & Biases logging"
    )
    parser.add_argument(
        "--no-tensorboard",
        action="store_true",
        help="Disable TensorBoard logging"
    )
    
    # Reproducibility
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility. Overrides config value."
    )
    
    # Device
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "cpu", "mps"],
        help="Device to use for training. Auto-detected if not specified."
    )
    
    # Validation
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only run validation, don't train"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run a few iterations to verify setup without full training"
    )
    
    return parser.parse_args()


def load_config(args) -> Config:
    """
    Load and merge configuration from file and CLI overrides.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        Merged configuration object
    """
    logger = get_logger(__name__)
    
    # Load base config
    if not Path(args.config).exists():
        raise FileNotFoundError(f"Config file not found: {args.config}")
    
    config = Config.from_yaml(args.config)
    logger.info(f"Loaded config from: {args.config}")
    
    # Apply CLI overrides
    if args.override:
        logger.info(f"Applying {len(args.override)} config overrides from CLI")
        for override in args.override:
            try:
                key, value = override.split("=", 1)
                # Try to parse value as int, float, bool, or keep as string
                if value.lower() == "true":
                    value = True
                elif value.lower() == "false":
                    value = False
                else:
                    try:
                        value = int(value)
                    except ValueError:
                        try:
                            value = float(value)
                        except ValueError:
                            pass  # Keep as string
                
                config.set(key, value)
                logger.info(f"  {key} = {value}")
            except ValueError:
                logger.warning(f"Invalid override format: {override}. Expected key=value")
    
    # Apply argument overrides
    if args.output_dir:
        config.set("training.output_dir", args.output_dir)
    if args.resume:
        config.set("checkpoint.resume_from", args.resume)
    if args.seed is not None:
        config.set("seed", args.seed)
    if args.device:
        config.set("device", args.device)
    if args.no_wandb and "logging.wandb.enabled" in config:
        config.set("logging.wandb.enabled", False)
    if args.no_tensorboard and "logging.tensorboard.enabled" in config:
        config.set("logging.tensorboard.enabled", False)
    
    return config


def setup_training(config: Config, args):
    """
    Setup training components: model, datasets, trainer.
    
    Args:
        config: Configuration object
        args: Command line arguments
        
    Returns:
        Tuple of (trainer, train_dataloader, val_dataloader)
    """
    logger = get_logger(__name__)
    
    # Set random seed
    seed = config.get("seed", 42)
    set_seed(seed)
    logger.info(f"Set random seed: {seed}")
    
    # Determine device
    if config.get("device") == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        device = "cpu"
    else:
        device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Build model
    logger.info("Building model...")
    model = build_model(config.get("model", {}))
    logger.info(f"Model created: {model.__class__.__name__}")
    logger.info(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Build datasets
    logger.info("Loading datasets...")
    train_dataset = build_dataset(config.get("dataset.train", {}))
    logger.info(f"Training samples: {len(train_dataset)}")
    
    val_dataset = None
    if "dataset.val" in config:
        val_dataset = build_dataset(config.get("dataset.val", {}))
        logger.info(f"Validation samples: {len(val_dataset)}")
    
    # Build dataloaders
    dataloader_config = config.get("dataloader", {})
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.get("training.batch_size", 32),
        shuffle=True,
        num_workers=dataloader_config.get("num_workers", 4),
        pin_memory=dataloader_config.get("pin_memory", True),
        persistent_workers=dataloader_config.get("persistent_workers", True) if dataloader_config.get("num_workers", 4) > 0 else False,
        prefetch_factor=dataloader_config.get("prefetch_factor", 2) if dataloader_config.get("num_workers", 4) > 0 else None,
    )
    
    val_dataloader = None
    if val_dataset is not None:
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=config.get("training.val_batch_size", config.get("training.batch_size", 32)),
            shuffle=False,
            num_workers=dataloader_config.get("num_workers", 4),
            pin_memory=dataloader_config.get("pin_memory", True),
            persistent_workers=dataloader_config.get("persistent_workers", True) if dataloader_config.get("num_workers", 4) > 0 else False,
        )
    
    # Build trainer config
    training_config = config.get("training", {})
    trainer_config = TrainerConfig(
        output_dir=training_config.get("output_dir", "./outputs"),
        max_epochs=training_config.get("max_epochs", 10),
        max_steps=training_config.get("max_steps", None),
        gradient_accumulation_steps=training_config.get("gradient_accumulation_steps", 1),
        gradient_clip_norm=training_config.get("gradient_clip_norm", 1.0),
        mixed_precision=config.get("mixed_precision", True),
        log_interval=config.get("logging.log_interval", 10),
        eval_interval=config.get("logging.eval_interval", None),
        save_interval=config.get("logging.save_interval", 1000),
        save_total_limit=config.get("checkpoint.keep_last_n", 5),
        resume_from_checkpoint=config.get("checkpoint.resume_from", None),
        seed=seed,
        dataloader_num_workers=dataloader_config.get("num_workers", 4),
        dataloader_pin_memory=dataloader_config.get("pin_memory", True),
        use_wandb=config.get("logging.wandb.enabled", False),
        use_tensorboard=config.get("logging.tensorboard.enabled", False),
        project_name=config.get("logging.wandb.project", "librobot-vla"),
        run_name=config.get("logging.wandb.run_name", None),
    )
    
    # Create trainer
    logger.info("Creating trainer...")
    trainer = Trainer(
        model=model,
        config=trainer_config,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer_config=config.get("optimizer", {}),
        scheduler_config=config.get("scheduler", {}),
        loss_config=config.get("loss", {}),
    )
    
    return trainer, train_dataloader, val_dataloader


def main():
    """Main training entry point."""
    args = parse_args()
    
    # Setup logging first (before distributed setup)
    log_file = args.log_file
    if log_file is None and args.output_dir:
        log_file = Path(args.output_dir) / "train.log"
    
    setup_logging(
        level=args.log_level,
        log_file=log_file,
        force_colors=True,
    )
    logger = get_logger(__name__)
    
    try:
        # Load configuration
        logger.info("=" * 80)
        logger.info("LibroBot VLA Training")
        logger.info("=" * 80)
        config = load_config(args)
        
        # Setup distributed training if applicable
        if is_distributed():
            logger.info("Initializing distributed training...")
            setup_distributed()
            logger.info(f"Rank {dist.get_rank()} / {dist.get_world_size()}")
        
        # Setup training components
        trainer, train_dataloader, val_dataloader = setup_training(config, args)
        
        # Dry run mode
        if args.dry_run:
            logger.info("=" * 80)
            logger.info("DRY RUN MODE - Running 5 iterations only")
            logger.info("=" * 80)
            trainer.config.max_steps = 5
            trainer.config.log_interval = 1
        
        # Validation only mode
        if args.validate_only:
            logger.info("=" * 80)
            logger.info("VALIDATION ONLY MODE")
            logger.info("=" * 80)
            if val_dataloader is None:
                logger.error("No validation dataset configured")
                sys.exit(1)
            
            metrics = trainer.evaluate(val_dataloader)
            logger.info("Validation metrics:")
            for key, value in metrics.items():
                logger.info(f"  {key}: {value}")
            return
        
        # Start training
        logger.info("=" * 80)
        logger.info("Starting training...")
        logger.info("=" * 80)
        trainer.train()
        
        # Final evaluation
        if val_dataloader is not None:
            logger.info("=" * 80)
            logger.info("Running final evaluation...")
            logger.info("=" * 80)
            metrics = trainer.evaluate(val_dataloader)
            logger.info("Final validation metrics:")
            for key, value in metrics.items():
                logger.info(f"  {key}: {value}")
        
        logger.info("=" * 80)
        logger.info("Training completed successfully!")
        logger.info("=" * 80)
        
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Training failed with error: {e}")
        sys.exit(1)
    finally:
        # Cleanup distributed training
        if is_distributed():
            cleanup_distributed()


if __name__ == "__main__":
    main()
