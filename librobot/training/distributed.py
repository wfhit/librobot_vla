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
    
    Provides efficient distributed training with ZeRO optimization,
    mixed precision, gradient accumulation, and pipeline parallelism.
    
    Features:
        - ZeRO optimization stages (1, 2, 3)
        - Mixed precision training (FP16, BF16)
        - Gradient accumulation
        - CPU/NVMe offloading
        - Pipeline parallelism support
        - Automatic loss scaling
    
    Args:
        model: Model to wrap
        config: DeepSpeed configuration dict
        model_parameters: Model parameters (optional, uses model.parameters())
        training_data: Training dataloader (optional)
        optimizer: Optional optimizer (DeepSpeed can create one)
        lr_scheduler: Optional LR scheduler
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        model_parameters: Optional[Any] = None,
        training_data: Optional[Any] = None,
        optimizer: Optional[Any] = None,
        lr_scheduler: Optional[Any] = None,
    ):
        self._original_model = model
        self.ds_config = config
        self.optimizer = optimizer
        self.scheduler = lr_scheduler
        self.engine = None
        
        try:
            import deepspeed
            
            # Get model parameters
            if model_parameters is None:
                model_parameters = filter(lambda p: p.requires_grad, model.parameters())
            
            # Initialize DeepSpeed
            self.engine, self.optimizer, self.data_loader, self.scheduler = deepspeed.initialize(
                model=model,
                model_parameters=list(model_parameters),
                config=config,
                training_data=training_data,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
            )
            
            self.model = self.engine
            
            # Log configuration
            zero_stage = config.get("zero_optimization", {}).get("stage", 0)
            fp16_enabled = config.get("fp16", {}).get("enabled", False)
            bf16_enabled = config.get("bf16", {}).get("enabled", False)
            
            logger.info(
                f"DeepSpeed initialized: "
                f"ZeRO stage={zero_stage}, "
                f"FP16={fp16_enabled}, "
                f"BF16={bf16_enabled}"
            )
            
        except ImportError:
            logger.warning(
                "DeepSpeed not installed. Install with: pip install deepspeed"
            )
            self.model = model
            raise
    
    def forward(self, *args, **kwargs) -> Any:
        """Forward pass through the model."""
        return self.model(*args, **kwargs)
    
    def backward(self, loss: torch.Tensor) -> None:
        """Backward pass with DeepSpeed loss scaling."""
        self.engine.backward(loss)
    
    def step(self) -> None:
        """Optimizer step with DeepSpeed."""
        self.engine.step()
    
    def __getattr__(self, name: str) -> Any:
        """Forward attribute access to wrapped model."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            if self.engine is not None:
                return getattr(self.engine, name)
            return getattr(self.model, name)
    
    def unwrap(self) -> nn.Module:
        """
        Get the underlying model without DeepSpeed wrapper.
        
        Returns:
            Unwrapped model
        """
        if self.engine is not None:
            return self.engine.module
        return self._original_model
    
    def save_checkpoint(
        self,
        save_dir: str,
        tag: Optional[str] = None,
        client_state: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Save DeepSpeed checkpoint.
        
        Args:
            save_dir: Directory to save checkpoint
            tag: Checkpoint tag (version)
            client_state: Additional state to save
        """
        if self.engine is not None:
            self.engine.save_checkpoint(
                save_dir=save_dir,
                tag=tag,
                client_state=client_state or {},
            )
            logger.info(f"DeepSpeed checkpoint saved to {save_dir}")
    
    def load_checkpoint(
        self,
        load_dir: str,
        tag: Optional[str] = None,
        load_optimizer_states: bool = True,
        load_lr_scheduler_states: bool = True,
    ) -> Dict[str, Any]:
        """
        Load DeepSpeed checkpoint.
        
        Args:
            load_dir: Directory to load checkpoint from
            tag: Checkpoint tag (version)
            load_optimizer_states: Whether to load optimizer states
            load_lr_scheduler_states: Whether to load LR scheduler states
            
        Returns:
            Client state dictionary
        """
        if self.engine is not None:
            _, client_state = self.engine.load_checkpoint(
                load_dir=load_dir,
                tag=tag,
                load_optimizer_states=load_optimizer_states,
                load_lr_scheduler_states=load_lr_scheduler_states,
            )
            logger.info(f"DeepSpeed checkpoint loaded from {load_dir}")
            return client_state or {}
        return {}
    
    @staticmethod
    def get_default_config(
        zero_stage: int = 2,
        fp16: bool = True,
        bf16: bool = False,
        gradient_accumulation_steps: int = 1,
        train_batch_size: int = 32,
        train_micro_batch_size_per_gpu: int = 4,
        offload_optimizer: bool = False,
        offload_params: bool = False,
    ) -> Dict[str, Any]:
        """
        Get default DeepSpeed configuration.
        
        Args:
            zero_stage: ZeRO optimization stage (0, 1, 2, or 3)
            fp16: Enable FP16 training
            bf16: Enable BF16 training
            gradient_accumulation_steps: Number of gradient accumulation steps
            train_batch_size: Total training batch size
            train_micro_batch_size_per_gpu: Micro batch size per GPU
            offload_optimizer: Offload optimizer to CPU (ZeRO-3)
            offload_params: Offload parameters to CPU (ZeRO-3)
            
        Returns:
            DeepSpeed configuration dictionary
        """
        config = {
            "train_batch_size": train_batch_size,
            "train_micro_batch_size_per_gpu": train_micro_batch_size_per_gpu,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "gradient_clipping": 1.0,
            "steps_per_print": 100,
        }
        
        # FP16/BF16 configuration
        if fp16:
            config["fp16"] = {
                "enabled": True,
                "loss_scale": 0,
                "loss_scale_window": 1000,
                "initial_scale_power": 16,
                "hysteresis": 2,
                "min_loss_scale": 1,
            }
        elif bf16:
            config["bf16"] = {
                "enabled": True,
            }
        
        # ZeRO optimization configuration
        zero_config = {
            "stage": zero_stage,
            "allgather_partitions": True,
            "allgather_bucket_size": 2e8,
            "reduce_scatter": True,
            "reduce_bucket_size": 2e8,
            "overlap_comm": True,
            "contiguous_gradients": True,
        }
        
        # ZeRO-3 specific options
        if zero_stage == 3:
            zero_config["stage3_max_live_parameters"] = 1e9
            zero_config["stage3_max_reuse_distance"] = 1e9
            zero_config["stage3_prefetch_bucket_size"] = 5e7
            zero_config["stage3_param_persistence_threshold"] = 1e5
            
            if offload_optimizer:
                zero_config["offload_optimizer"] = {
                    "device": "cpu",
                    "pin_memory": True,
                }
            
            if offload_params:
                zero_config["offload_param"] = {
                    "device": "cpu",
                    "pin_memory": True,
                }
        
        config["zero_optimization"] = zero_config
        
        return config


class FSDPWrapper:
    """
    Wrapper for Fully Sharded Data Parallel (FSDP).
    
    Provides memory-efficient distributed training by sharding model parameters,
    gradients, and optimizer states across multiple GPUs.
    
    Features:
        - Automatic mixed precision training
        - CPU offloading for memory reduction
        - Activation checkpointing
        - Transformer auto-wrap policy
        - Gradient accumulation support
    
    Args:
        model: Model to wrap
        config: Distributed configuration
        mixed_precision: Whether to use mixed precision ("fp16", "bf16", or None)
        cpu_offload: Whether to offload parameters to CPU
        activation_checkpointing: Whether to use activation checkpointing
        sharding_strategy: FSDP sharding strategy ("full", "shard_grad_op", "no_shard")
        auto_wrap_policy: Auto wrap policy for transformers ("transformer", "size_based", None)
        min_num_params: Minimum parameters for size-based wrapping
        backward_prefetch: Backward prefetch strategy ("backward_pre", "backward_post", None)
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: DistributedConfig,
        mixed_precision: Optional[str] = None,
        cpu_offload: bool = False,
        activation_checkpointing: bool = False,
        sharding_strategy: str = "full",
        auto_wrap_policy: Optional[str] = None,
        min_num_params: int = 100_000,
        backward_prefetch: Optional[str] = "backward_pre",
    ):
        self.config = config
        self._mixed_precision = mixed_precision
        self._cpu_offload = cpu_offload
        self._activation_checkpointing = activation_checkpointing
        self._original_model = model
        
        if not config.is_distributed:
            self.model = model
            logger.info("FSDP not used (single process)")
            return
        
        try:
            from torch.distributed.fsdp import (
                FullyShardedDataParallel as FSDP,
                ShardingStrategy,
                MixedPrecision,
                CPUOffload,
                BackwardPrefetch,
            )
            from torch.distributed.fsdp.wrap import (
                transformer_auto_wrap_policy,
                size_based_auto_wrap_policy,
            )
            from functools import partial
            
            # Configure sharding strategy
            sharding_map = {
                "full": ShardingStrategy.FULL_SHARD,
                "shard_grad_op": ShardingStrategy.SHARD_GRAD_OP,
                "no_shard": ShardingStrategy.NO_SHARD,
            }
            shard_strategy = sharding_map.get(sharding_strategy, ShardingStrategy.FULL_SHARD)
            
            # Configure mixed precision
            mp_policy = None
            if mixed_precision == "fp16":
                mp_policy = MixedPrecision(
                    param_dtype=torch.float16,
                    reduce_dtype=torch.float16,
                    buffer_dtype=torch.float16,
                )
            elif mixed_precision == "bf16":
                mp_policy = MixedPrecision(
                    param_dtype=torch.bfloat16,
                    reduce_dtype=torch.bfloat16,
                    buffer_dtype=torch.bfloat16,
                )
            
            # Configure CPU offloading
            cpu_offload_policy = CPUOffload(offload_params=cpu_offload) if cpu_offload else None
            
            # Configure auto wrap policy
            wrap_policy = None
            if auto_wrap_policy == "transformer":
                # Get transformer layer classes from model
                transformer_layer_cls = self._get_transformer_layers(model)
                if transformer_layer_cls:
                    wrap_policy = partial(
                        transformer_auto_wrap_policy,
                        transformer_layer_cls=transformer_layer_cls,
                    )
            elif auto_wrap_policy == "size_based":
                wrap_policy = partial(
                    size_based_auto_wrap_policy,
                    min_num_params=min_num_params,
                )
            
            # Configure backward prefetch
            prefetch_map = {
                "backward_pre": BackwardPrefetch.BACKWARD_PRE,
                "backward_post": BackwardPrefetch.BACKWARD_POST,
            }
            backward_prefetch_policy = prefetch_map.get(backward_prefetch) if backward_prefetch else None
            
            # Wrap model with FSDP
            self.model = FSDP(
                model,
                sharding_strategy=shard_strategy,
                mixed_precision=mp_policy,
                cpu_offload=cpu_offload_policy,
                auto_wrap_policy=wrap_policy,
                backward_prefetch=backward_prefetch_policy,
                device_id=torch.cuda.current_device() if torch.cuda.is_available() else None,
            )
            
            # Apply activation checkpointing
            if activation_checkpointing:
                self._apply_activation_checkpointing()
            
            logger.info(
                f"Model wrapped with FSDP: "
                f"sharding={sharding_strategy}, "
                f"mixed_precision={mixed_precision}, "
                f"cpu_offload={cpu_offload}, "
                f"activation_checkpointing={activation_checkpointing}"
            )
            
        except ImportError as e:
            logger.warning(f"FSDP not available. Requires PyTorch >= 2.0: {e}")
            self.model = model
            raise
    
    def _get_transformer_layers(self, model: nn.Module) -> set:
        """Get transformer layer classes from model for auto-wrapping."""
        layer_classes = set()
        
        # Common transformer layer class names
        transformer_names = [
            "TransformerEncoderLayer",
            "TransformerDecoderLayer",
            "BertLayer",
            "GPT2Block",
            "LlamaDecoderLayer",
            "Qwen2DecoderLayer",
            "DecoderLayer",
            "EncoderLayer",
        ]
        
        for name, module in model.named_modules():
            class_name = module.__class__.__name__
            if class_name in transformer_names:
                layer_classes.add(type(module))
        
        return layer_classes
    
    def _apply_activation_checkpointing(self) -> None:
        """Apply activation checkpointing to transformer layers."""
        try:
            from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
                checkpoint_wrapper,
                CheckpointImpl,
                apply_activation_checkpointing,
            )
            
            def check_fn(submodule) -> bool:
                """Check if module is a transformer layer that should be checkpointed."""
                try:
                    class_name = submodule.__class__.__name__
                    transformer_layer_names = [
                        "TransformerEncoderLayer",
                        "TransformerDecoderLayer",
                        "DecoderLayer",
                    ]
                    return any(name in class_name for name in transformer_layer_names)
                except AttributeError:
                    return False
            
            apply_activation_checkpointing(
                self.model,
                checkpoint_wrapper_fn=checkpoint_wrapper,
                check_fn=check_fn,
            )
            logger.info("Activation checkpointing applied to transformer layers")
            
        except ImportError:
            logger.warning("Activation checkpointing not available")
    
    def __getattr__(self, name: str) -> Any:
        """Forward attribute access to wrapped model."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)
    
    def unwrap(self) -> nn.Module:
        """
        Get the underlying model without FSDP wrapper.
        
        Returns:
            Unwrapped model
        """
        try:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            if isinstance(self.model, FSDP):
                return self.model.module
        except ImportError:
            pass
        return self._original_model
    
    def get_state_dict(self) -> Dict[str, Any]:
        """
        Get model state dict with FSDP-specific handling.
        
        Returns:
            State dictionary
        """
        try:
            from torch.distributed.fsdp import (
                FullyShardedDataParallel as FSDP,
                StateDictType,
                FullStateDictConfig,
            )
            
            # Use full state dict for saving
            save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT, save_policy):
                state_dict = self.model.state_dict()
            return state_dict
            
        except ImportError:
            return self.model.state_dict()
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        Load state dict with FSDP-specific handling.
        
        Args:
            state_dict: State dictionary to load
        """
        try:
            from torch.distributed.fsdp import (
                FullyShardedDataParallel as FSDP,
                StateDictType,
                FullStateDictConfig,
            )
            
            load_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT, load_policy):
                self.model.load_state_dict(state_dict)
                
        except ImportError:
            self.model.load_state_dict(state_dict)


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
