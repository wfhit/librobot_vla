"""Checkpoint saving and loading with metadata management."""

import json
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Union, List
from datetime import datetime
import torch
import torch.nn as nn


class Checkpoint:
    """
    Checkpoint manager for saving and loading model states with metadata.
    
    Examples:
        >>> checkpoint = Checkpoint(save_dir="./checkpoints")
        >>> checkpoint.save(model, optimizer, epoch=10, metrics={"loss": 0.5})
        >>> state = checkpoint.load("checkpoint_epoch_10.pt")
    """
    
    def __init__(
        self,
        save_dir: Union[str, Path],
        keep_last_n: Optional[int] = None,
        save_best: bool = True,
        metric_name: str = "loss",
        mode: str = "min",
    ):
        """
        Initialize checkpoint manager.
        
        Args:
            save_dir: Directory to save checkpoints
            keep_last_n: Number of recent checkpoints to keep (None = keep all)
            save_best: If True, keeps the best checkpoint separately
            metric_name: Metric name for determining best checkpoint
            mode: 'min' or 'max' for best checkpoint selection
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.keep_last_n = keep_last_n
        self.save_best = save_best
        self.metric_name = metric_name
        self.mode = mode
        
        self.best_metric = float('inf') if mode == 'min' else float('-inf')
        self.checkpoints: List[Path] = []
        
        # Load existing checkpoint list
        self._load_checkpoint_list()
    
    def save(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        epoch: Optional[int] = None,
        step: Optional[int] = None,
        metrics: Optional[Dict[str, float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        filename: Optional[str] = None,
    ) -> Path:
        """
        Save checkpoint with model state and metadata.
        
        Args:
            model: Model to save
            optimizer: Optimizer state to save
            scheduler: Learning rate scheduler state to save
            epoch: Current epoch number
            step: Current training step
            metrics: Dictionary of metric values
            metadata: Additional metadata to save
            filename: Custom filename (auto-generated if None)
            
        Returns:
            Path: Path to saved checkpoint
        """
        # Prepare checkpoint data
        checkpoint_data = {
            'model_state_dict': model.state_dict(),
            'epoch': epoch,
            'step': step,
            'timestamp': datetime.now().isoformat(),
        }
        
        if optimizer is not None:
            checkpoint_data['optimizer_state_dict'] = optimizer.state_dict()
        
        if scheduler is not None:
            checkpoint_data['scheduler_state_dict'] = scheduler.state_dict()
        
        if metrics is not None:
            checkpoint_data['metrics'] = metrics
        
        if metadata is not None:
            checkpoint_data['metadata'] = metadata
        
        # Generate filename
        if filename is None:
            if epoch is not None:
                filename = f"checkpoint_epoch_{epoch}.pt"
            elif step is not None:
                filename = f"checkpoint_step_{step}.pt"
            else:
                filename = f"checkpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
        
        filepath = self.save_dir / filename
        
        # Save checkpoint
        torch.save(checkpoint_data, filepath)
        self.checkpoints.append(filepath)
        
        # Save metadata separately for easy inspection
        metadata_path = filepath.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump({
                'epoch': epoch,
                'step': step,
                'metrics': metrics,
                'timestamp': checkpoint_data['timestamp'],
            }, f, indent=2)
        
        # Check if this is the best checkpoint
        if self.save_best and metrics and self.metric_name in metrics:
            metric_value = metrics[self.metric_name]
            is_best = (
                (self.mode == 'min' and metric_value < self.best_metric) or
                (self.mode == 'max' and metric_value > self.best_metric)
            )
            
            if is_best:
                self.best_metric = metric_value
                best_path = self.save_dir / "best.pt"
                shutil.copy2(filepath, best_path)
                
                # Save best metadata
                best_metadata_path = self.save_dir / "best.json"
                shutil.copy2(metadata_path, best_metadata_path)
        
        # Clean up old checkpoints
        self._cleanup_checkpoints()
        self._save_checkpoint_list()
        
        return filepath
    
    def load(
        self,
        filename: Union[str, Path],
        model: Optional[nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        map_location: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Load checkpoint from file.
        
        Args:
            filename: Checkpoint filename or path
            model: Model to load state into
            optimizer: Optimizer to load state into
            scheduler: Scheduler to load state into
            map_location: Device to map tensors to
            
        Returns:
            Dict: Checkpoint data
        """
        filepath = Path(filename)
        if not filepath.is_absolute():
            filepath = self.save_dir / filepath
        
        if not filepath.exists():
            raise FileNotFoundError(f"Checkpoint not found: {filepath}")
        
        checkpoint_data = torch.load(filepath, map_location=map_location)
        
        # Load states into provided objects
        if model is not None and 'model_state_dict' in checkpoint_data:
            model.load_state_dict(checkpoint_data['model_state_dict'])
        
        if optimizer is not None and 'optimizer_state_dict' in checkpoint_data:
            optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
        
        if scheduler is not None and 'scheduler_state_dict' in checkpoint_data:
            scheduler.load_state_dict(checkpoint_data['scheduler_state_dict'])
        
        return checkpoint_data
    
    def load_best(
        self,
        model: Optional[nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        map_location: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Load the best checkpoint.
        
        Args:
            model: Model to load state into
            optimizer: Optimizer to load state into
            scheduler: Scheduler to load state into
            map_location: Device to map tensors to
            
        Returns:
            Dict: Checkpoint data
        """
        return self.load("best.pt", model, optimizer, scheduler, map_location)
    
    def load_latest(
        self,
        model: Optional[nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        map_location: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Load the most recent checkpoint.
        
        Args:
            model: Model to load state into
            optimizer: Optimizer to load state into
            scheduler: Scheduler to load state into
            map_location: Device to map tensors to
            
        Returns:
            Dict: Checkpoint data
        """
        if not self.checkpoints:
            raise FileNotFoundError("No checkpoints found")
        
        latest = self.checkpoints[-1]
        return self.load(latest, model, optimizer, scheduler, map_location)
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """
        List all available checkpoints with metadata.
        
        Returns:
            List of checkpoint information dictionaries
        """
        checkpoints_info = []
        
        for checkpoint_path in self.checkpoints:
            if checkpoint_path.exists():
                metadata_path = checkpoint_path.with_suffix('.json')
                
                info = {'path': str(checkpoint_path)}
                
                if metadata_path.exists():
                    with open(metadata_path, 'r') as f:
                        info.update(json.load(f))
                
                checkpoints_info.append(info)
        
        return checkpoints_info
    
    def _cleanup_checkpoints(self) -> None:
        """Remove old checkpoints based on keep_last_n."""
        if self.keep_last_n is None:
            return
        
        # Keep only the last N checkpoints
        while len(self.checkpoints) > self.keep_last_n:
            old_checkpoint = self.checkpoints.pop(0)
            
            if old_checkpoint.exists():
                old_checkpoint.unlink()
            
            # Remove metadata file
            metadata_path = old_checkpoint.with_suffix('.json')
            if metadata_path.exists():
                metadata_path.unlink()
    
    def _save_checkpoint_list(self) -> None:
        """Save list of checkpoints to file."""
        list_path = self.save_dir / ".checkpoint_list.json"
        with open(list_path, 'w') as f:
            json.dump([str(p) for p in self.checkpoints], f, indent=2)
    
    def _load_checkpoint_list(self) -> None:
        """Load list of checkpoints from file."""
        list_path = self.save_dir / ".checkpoint_list.json"
        if list_path.exists():
            with open(list_path, 'r') as f:
                paths = json.load(f)
                self.checkpoints = [Path(p) for p in paths if Path(p).exists()]


def save_checkpoint(
    path: Union[str, Path],
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    **kwargs
) -> None:
    """
    Quick function to save a checkpoint.
    
    Args:
        path: Output path for checkpoint
        model: Model to save
        optimizer: Optimizer to save
        **kwargs: Additional data to save
    """
    checkpoint_data = {
        'model_state_dict': model.state_dict(),
        **kwargs
    }
    
    if optimizer is not None:
        checkpoint_data['optimizer_state_dict'] = optimizer.state_dict()
    
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint_data, path)


def load_checkpoint(
    path: Union[str, Path],
    model: Optional[nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    map_location: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Quick function to load a checkpoint.
    
    Args:
        path: Path to checkpoint
        model: Model to load state into
        optimizer: Optimizer to load state into
        map_location: Device to map tensors to
        
    Returns:
        Dict: Checkpoint data
    """
    checkpoint_data = torch.load(path, map_location=map_location)
    
    if model is not None and 'model_state_dict' in checkpoint_data:
        model.load_state_dict(checkpoint_data['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint_data:
        optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
    
    return checkpoint_data
