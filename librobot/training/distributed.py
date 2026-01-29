"""Distributed training utilities for DDP, DeepSpeed, and FSDP."""

import os
from typing import Any, Dict, Optional, Tuple, Union
from datetime import timedelta
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from librobot.utils.logging import get_logger


logger = get_logger(__name__)


class DistributedConfig:
    """
    Configuration for distributed training.
    
    Args:
        backend: Distributed backend ('nccl', 'gloo', 'mpi')
        world_size: Total number of processes
        rank: Rank of current process
        local_rank: Local rank on current node
        master_addr: Address of master node
        master_port: Port of master node
    """
    
    def __init__(
        self,
        backend: str = "nccl",
        world_size: Optional[int] = None,
        rank: Optional[int] = None,
        local_rank: Optional[int] = None,
        master_addr: Optional[str] = None,
        master_port: Optional[str] = None,
    ):
        self.backend = backend
        self.world_size = world_size or int(os.environ.get("WORLD_SIZE", 1))
        self.rank = rank if rank is not None else int(os.environ.get("RANK", 0))
        self.local_rank = local_rank if local_rank is not None else int(
            os.environ.get("LOCAL_RANK", 0)
        )
        self.master_addr = master_addr or os.environ.get("MASTER_ADDR", "localhost")
        self.master_port = master_port or os.environ.get("MASTER_PORT", "12355")
    
    @property
    def is_distributed(self) -> bool:
        """Check if running in distributed mode."""
        return self.world_size > 1
    
    @property
    def is_main_process(self) -> bool:
        """Check if this is the main process."""
        return self.rank == 0
    
    @property
    def is_local_main_process(self) -> bool:
        """Check if this is the main process on this node."""
        return self.local_rank == 0


def setup_distributed(
    backend: str = "nccl",
    timeout_minutes: int = 30,
) -> DistributedConfig:
    """
    Initialize distributed training environment.
    
    Args:
        backend: Distributed backend ('nccl', 'gloo', 'mpi')
        timeout_minutes: Timeout for distributed operations
        
    Returns:
        DistributedConfig: Configuration object
        
    Examples:
        >>> config = setup_distributed()
        >>> if config.is_distributed:
        ...     model = DDP(model, device_ids=[config.local_rank])
    """
    config = DistributedConfig(backend=backend)
    
    if not config.is_distributed:
        logger.info("Running in single-process mode")
        return config
    
    # Set environment variables
    os.environ["MASTER_ADDR"] = config.master_addr
    os.environ["MASTER_PORT"] = config.master_port
    
    # Initialize process group
    if not dist.is_initialized():
        timeout = timedelta(minutes=timeout_minutes)
        dist.init_process_group(
            backend=backend,
            init_method="env://",
            world_size=config.world_size,
            rank=config.rank,
            timeout=timeout,
        )
    
    logger.info(
        f"Distributed training initialized: "
        f"world_size={config.world_size}, "
        f"rank={config.rank}, "
        f"local_rank={config.local_rank}, "
        f"backend={backend}"
    )
    
    return config


def cleanup_distributed() -> None:
    """
    Clean up distributed training environment.
    
    Should be called at the end of training when using distributed mode.
    """
    if dist.is_initialized():
        dist.destroy_process_group()
        logger.info("Distributed training environment cleaned up")


def is_distributed() -> bool:
    """
    Check if running in distributed mode.
    
    Returns:
        bool: True if distributed training is active
    """
    return dist.is_available() and dist.is_initialized()


def get_world_size() -> int:
    """
    Get the number of processes in distributed training.
    
    Returns:
        int: World size (1 if not distributed)
    """
    if is_distributed():
        return dist.get_world_size()
    return 1


def get_rank() -> int:
    """
    Get the rank of current process.
    
    Returns:
        int: Rank (0 if not distributed)
    """
    if is_distributed():
        return dist.get_rank()
    return 0


def is_main_process() -> bool:
    """
    Check if this is the main process.
    
    Returns:
        bool: True if main process or not distributed
    """
    return get_rank() == 0


def barrier() -> None:
    """
    Synchronize all processes.
    
    Blocks until all processes reach this point.
    """
    if is_distributed():
        dist.barrier()


def all_reduce(tensor: torch.Tensor, op: str = "sum") -> torch.Tensor:
    """
    All-reduce a tensor across all processes.
    
    Args:
        tensor: Tensor to reduce
        op: Reduction operation ('sum', 'mean', 'max', 'min')
        
    Returns:
        Reduced tensor
    """
    if not is_distributed():
        return tensor
    
    # Map operation string to dist.ReduceOp
    op_map = {
        "sum": dist.ReduceOp.SUM,
        "mean": dist.ReduceOp.SUM,
        "max": dist.ReduceOp.MAX,
        "min": dist.ReduceOp.MIN,
    }
    
    if op not in op_map:
        raise ValueError(f"Unknown operation: {op}. Available: {list(op_map.keys())}")
    
    # Perform all-reduce
    dist.all_reduce(tensor, op=op_map[op])
    
    # Average if needed
    if op == "mean":
        tensor /= get_world_size()
    
    return tensor


def all_gather(tensor: torch.Tensor) -> torch.Tensor:
    """
    Gather tensors from all processes.
    
    Args:
        tensor: Tensor to gather
        
    Returns:
        Concatenated tensor from all processes
    """
    if not is_distributed():
        return tensor
    
    # Prepare output tensor list
    world_size = get_world_size()
    tensor_list = [torch.zeros_like(tensor) for _ in range(world_size)]
    
    # Gather tensors
    dist.all_gather(tensor_list, tensor)
    
    # Concatenate
    return torch.cat(tensor_list, dim=0)


def broadcast(tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
    """
    Broadcast tensor from source to all processes.
    
    Args:
        tensor: Tensor to broadcast
        src: Source rank
        
    Returns:
        Broadcasted tensor
    """
    if is_distributed():
        dist.broadcast(tensor, src=src)
    return tensor


class DDPWrapper:
    """
    Wrapper for PyTorch DistributedDataParallel.
    
    Provides simplified interface for DDP training with automatic device placement
    and gradient synchronization control.
    
    Args:
        model: Model to wrap
        config: Distributed configuration
        device_ids: GPU IDs for this process
        output_device: Output device
        find_unused_parameters: Whether to find unused parameters
        gradient_as_bucket_view: Use gradient as bucket view for efficiency
        static_graph: Whether computational graph is static
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: DistributedConfig,
        device_ids: Optional[list] = None,
        output_device: Optional[int] = None,
        find_unused_parameters: bool = False,
        gradient_as_bucket_view: bool = True,
        static_graph: bool = False,
    ):
        self.config = config
        
        if not config.is_distributed:
            self.model = model
            logger.info("DDP not used (single process)")
            return
        
        # Set device IDs
        if device_ids is None:
            device_ids = [config.local_rank]
        if output_device is None:
            output_device = config.local_rank
        
        # Wrap model with DDP
        self.model = DDP(
            model,
            device_ids=device_ids,
            output_device=output_device,
            find_unused_parameters=find_unused_parameters,
            gradient_as_bucket_view=gradient_as_bucket_view,
            static_graph=static_graph,
        )
        
        logger.info(
            f"Model wrapped with DDP: "
            f"device_ids={device_ids}, "
            f"output_device={output_device}"
        )
    
    def __getattr__(self, name: str) -> Any:
        """Forward attribute access to wrapped model."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)
    
    def unwrap(self) -> nn.Module:
        """
        Get the underlying model without DDP wrapper.
        
        Returns:
            Unwrapped model
        """
        if isinstance(self.model, DDP):
            return self.model.module
        return self.model


class DeepSpeedWrapper:
    """
    Wrapper for DeepSpeed training.
    
    Note: This is a placeholder for DeepSpeed integration.
    Actual implementation requires deepspeed package.
    
    TODO: Implement DeepSpeed integration with:
        - ZeRO optimization stages
        - Mixed precision training
        - Gradient accumulation
        - Checkpoint loading/saving
        - Pipeline parallelism support
    
    Args:
        model: Model to wrap
        config: DeepSpeed configuration dict
        model_parameters: Model parameters
        training_data: Training dataloader
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        model_parameters: Optional[Any] = None,
        training_data: Optional[Any] = None,
    ):
        self.model = model
        self.config = config
        
        try:
            import deepspeed
            # TODO: Implement DeepSpeed initialization
            # self.model, self.optimizer, _, self.scheduler = deepspeed.initialize(
            #     model=model,
            #     model_parameters=model_parameters,
            #     config=config,
            #     training_data=training_data,
            # )
            logger.warning("DeepSpeed wrapper is a placeholder - full implementation pending")
        except ImportError:
            logger.warning(
                "DeepSpeed not installed. Install with: pip install deepspeed"
            )
            raise
    
    def unwrap(self) -> nn.Module:
        """Get the underlying model."""
        # TODO: Implement proper unwrapping for DeepSpeed
        return self.model


class FSDPWrapper:
    """
    Wrapper for Fully Sharded Data Parallel (FSDP).
    
    Note: This is a placeholder for FSDP integration.
    Actual implementation requires PyTorch >= 1.12.
    
    TODO: Implement FSDP integration with:
        - Automatic mixed precision
        - Gradient accumulation
        - CPU offloading
        - Activation checkpointing
        - Transformer auto-wrap policy
    
    Args:
        model: Model to wrap
        config: Distributed configuration
        mixed_precision: Whether to use mixed precision
        cpu_offload: Whether to offload to CPU
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: DistributedConfig,
        mixed_precision: bool = False,
        cpu_offload: bool = False,
    ):
        self.model = model
        self.config = config
        
        if not config.is_distributed:
            logger.info("FSDP not used (single process)")
            return
        
        try:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            # TODO: Implement FSDP initialization
            # self.model = FSDP(
            #     model,
            #     mixed_precision=...,
            #     cpu_offload=...,
            # )
            logger.warning("FSDP wrapper is a placeholder - full implementation pending")
        except ImportError:
            logger.warning(
                "FSDP not available. Requires PyTorch >= 1.12"
            )
            raise
    
    def unwrap(self) -> nn.Module:
        """Get the underlying model."""
        # TODO: Implement proper unwrapping for FSDP
        return self.model


def wrap_model_distributed(
    model: nn.Module,
    strategy: str = "ddp",
    config: Optional[DistributedConfig] = None,
    **kwargs
) -> Union[nn.Module, DDPWrapper, DeepSpeedWrapper, FSDPWrapper]:
    """
    Wrap model for distributed training.
    
    Args:
        model: Model to wrap
        strategy: Distributed strategy ('ddp', 'deepspeed', 'fsdp')
        config: Distributed configuration
        **kwargs: Additional strategy-specific arguments
        
    Returns:
        Wrapped model
        
    Examples:
        >>> config = setup_distributed()
        >>> model = wrap_model_distributed(model, strategy="ddp", config=config)
    """
    if config is None:
        config = setup_distributed()
    
    if not config.is_distributed:
        logger.info("Single process mode - no distributed wrapper needed")
        return model
    
    if strategy == "ddp":
        return DDPWrapper(model, config, **kwargs)
    elif strategy == "deepspeed":
        return DeepSpeedWrapper(model, **kwargs)
    elif strategy == "fsdp":
        return FSDPWrapper(model, config, **kwargs)
    else:
        raise ValueError(
            f"Unknown distributed strategy: {strategy}. "
            f"Available: ddp, deepspeed, fsdp"
        )


def save_checkpoint_distributed(
    state_dict: Dict[str, Any],
    filepath: str,
    config: Optional[DistributedConfig] = None,
) -> None:
    """
    Save checkpoint in distributed training.
    
    Only saves on main process to avoid conflicts.
    
    Args:
        state_dict: State dictionary to save
        filepath: Path to save checkpoint
        config: Distributed configuration
    """
    if config is None:
        should_save = is_main_process()
    else:
        should_save = config.is_main_process
    
    if should_save:
        torch.save(state_dict, filepath)
        logger.info(f"Checkpoint saved: {filepath}")
    
    # Synchronize all processes
    barrier()


def load_checkpoint_distributed(
    filepath: str,
    map_location: Optional[Union[str, torch.device]] = None,
) -> Dict[str, Any]:
    """
    Load checkpoint in distributed training.
    
    Args:
        filepath: Path to checkpoint
        map_location: Device to map tensors to
        
    Returns:
        Loaded state dictionary
        
    Warning:
        Uses weights_only=False to support loading optimizer states and other objects.
        Only load checkpoints from trusted sources to avoid security risks.
    """
    # Synchronize before loading
    barrier()
    
    # Note: weights_only=False allows loading optimizer states and other objects
    # SECURITY WARNING: Only load checkpoints from trusted sources
    state_dict = torch.load(filepath, map_location=map_location, weights_only=False)
    
    logger.info(f"Checkpoint loaded: {filepath}")
    return state_dict
