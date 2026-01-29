"""Optimizer builders with registry support."""

from typing import Any, Callable, Dict, Iterable, List, Optional, Union
import torch
import torch.nn as nn
from torch.optim import Optimizer

from librobot.utils.registry import Registry


# Global optimizer registry
OPTIMIZER_REGISTRY = Registry("optimizers")


def register_optimizer(name: str, **kwargs):
    """
    Decorator to register an optimizer builder.
    
    Args:
        name: Name to register the optimizer under
        **kwargs: Additional metadata
        
    Examples:
        >>> @register_optimizer("adam")
        >>> def build_adam(params, lr=1e-3, **kwargs):
        ...     return torch.optim.Adam(params, lr=lr, **kwargs)
    """
    return OPTIMIZER_REGISTRY.register(name, **kwargs)


class OptimizerBuilder:
    """
    Builder class for creating optimizers with parameter groups.
    
    Supports:
    - Layerwise learning rate scaling
    - Weight decay exclusions (bias, norm layers)
    - Parameter group customization
    - Mixed precision optimization
    
    Examples:
        >>> builder = OptimizerBuilder("adamw", lr=1e-3, weight_decay=0.01)
        >>> optimizer = builder.build(model)
        >>> 
        >>> # With custom parameter groups
        >>> builder = OptimizerBuilder("adamw", lr=1e-3)
        >>> builder.add_param_group(model.encoder.parameters(), lr=1e-4)
        >>> builder.add_param_group(model.decoder.parameters(), lr=1e-3)
        >>> optimizer = builder.build()
    """
    
    def __init__(
        self,
        optimizer_type: str,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        exclude_bias_and_norm: bool = True,
        **optimizer_kwargs
    ):
        """
        Initialize optimizer builder.
        
        Args:
            optimizer_type: Type of optimizer (registered name)
            lr: Base learning rate
            weight_decay: Weight decay coefficient
            exclude_bias_and_norm: If True, excludes bias and norm params from weight decay
            **optimizer_kwargs: Additional optimizer-specific arguments
        """
        self.optimizer_type = optimizer_type
        self.lr = lr
        self.weight_decay = weight_decay
        self.exclude_bias_and_norm = exclude_bias_and_norm
        self.optimizer_kwargs = optimizer_kwargs
        self._param_groups: List[Dict[str, Any]] = []
    
    def add_param_group(
        self,
        params: Iterable[torch.Tensor],
        lr: Optional[float] = None,
        weight_decay: Optional[float] = None,
        **kwargs
    ) -> None:
        """
        Add a custom parameter group.
        
        Args:
            params: Parameters for this group
            lr: Learning rate override for this group
            weight_decay: Weight decay override for this group
            **kwargs: Additional optimizer arguments for this group
        """
        group = {'params': list(params)}
        
        if lr is not None:
            group['lr'] = lr
        if weight_decay is not None:
            group['weight_decay'] = weight_decay
        
        group.update(kwargs)
        self._param_groups.append(group)
    
    def build(self, model: Optional[nn.Module] = None) -> Optimizer:
        """
        Build optimizer instance.
        
        Args:
            model: Model to optimize. Required if no custom param groups added.
            
        Returns:
            Optimizer: Configured optimizer instance
        """
        # Use custom param groups if specified
        if self._param_groups:
            param_groups = self._param_groups
        elif model is not None:
            param_groups = self._get_param_groups(model)
        else:
            raise ValueError("Either model or custom param groups must be provided")
        
        # Get optimizer builder from registry
        optimizer_fn = OPTIMIZER_REGISTRY.get(self.optimizer_type)
        if optimizer_fn is None:
            raise ValueError(
                f"Optimizer '{self.optimizer_type}' not found in registry. "
                f"Available: {OPTIMIZER_REGISTRY.list()}"
            )
        
        return optimizer_fn(
            param_groups,
            lr=self.lr,
            **self.optimizer_kwargs
        )
    
    def _get_param_groups(self, model: nn.Module) -> List[Dict[str, Any]]:
        """
        Create parameter groups with weight decay exclusions.
        
        Args:
            model: Model to create param groups from
            
        Returns:
            List of parameter group dictionaries
        """
        if not self.exclude_bias_and_norm:
            # Simple case: all parameters in one group
            return [{
                'params': model.parameters(),
                'lr': self.lr,
                'weight_decay': self.weight_decay,
            }]
        
        # Separate parameters with and without weight decay
        decay_params = []
        no_decay_params = []
        
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            
            # Exclude bias and normalization parameters from weight decay
            if self._should_exclude_from_decay(name, param):
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        
        param_groups = [
            {
                'params': decay_params,
                'lr': self.lr,
                'weight_decay': self.weight_decay,
            },
            {
                'params': no_decay_params,
                'lr': self.lr,
                'weight_decay': 0.0,
            }
        ]
        
        return param_groups
    
    @staticmethod
    def _should_exclude_from_decay(name: str, param: torch.Tensor) -> bool:
        """
        Check if parameter should be excluded from weight decay.
        
        Args:
            name: Parameter name
            param: Parameter tensor
            
        Returns:
            bool: True if should exclude from decay
        """
        # Bias parameters
        if name.endswith('.bias'):
            return True
        
        # 1D parameters (usually norm layer scales/biases)
        if param.ndim == 1:
            return True
        
        # Normalization layers by name
        if any(norm in name for norm in ['norm', 'ln', 'bn', 'gn']):
            return True
        
        # Embedding parameters
        if 'embedding' in name or 'emb' in name:
            return True
        
        return False


# Register built-in PyTorch optimizers
@register_optimizer("adam")
def build_adam(
    params: Union[Iterable[torch.Tensor], Iterable[Dict[str, Any]]],
    lr: float = 1e-3,
    betas: tuple = (0.9, 0.999),
    eps: float = 1e-8,
    weight_decay: float = 0.0,
    **kwargs
) -> Optimizer:
    """
    Build Adam optimizer.
    
    Args:
        params: Model parameters or parameter groups
        lr: Learning rate
        betas: Coefficients for computing running averages
        eps: Term for numerical stability
        weight_decay: Weight decay coefficient
        **kwargs: Additional optimizer arguments
        
    Returns:
        Optimizer: Adam optimizer instance
    """
    return torch.optim.Adam(
        params,
        lr=lr,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
        **kwargs
    )


@register_optimizer("adamw")
def build_adamw(
    params: Union[Iterable[torch.Tensor], Iterable[Dict[str, Any]]],
    lr: float = 1e-3,
    betas: tuple = (0.9, 0.999),
    eps: float = 1e-8,
    weight_decay: float = 0.01,
    **kwargs
) -> Optimizer:
    """
    Build AdamW optimizer with decoupled weight decay.
    
    Recommended for transformer models and most modern architectures.
    
    Args:
        params: Model parameters or parameter groups
        lr: Learning rate
        betas: Coefficients for computing running averages
        eps: Term for numerical stability
        weight_decay: Weight decay coefficient (decoupled)
        **kwargs: Additional optimizer arguments
        
    Returns:
        Optimizer: AdamW optimizer instance
    """
    return torch.optim.AdamW(
        params,
        lr=lr,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
        **kwargs
    )


@register_optimizer("sgd")
def build_sgd(
    params: Union[Iterable[torch.Tensor], Iterable[Dict[str, Any]]],
    lr: float = 1e-2,
    momentum: float = 0.0,
    dampening: float = 0.0,
    weight_decay: float = 0.0,
    nesterov: bool = False,
    **kwargs
) -> Optimizer:
    """
    Build SGD optimizer with optional momentum.
    
    Args:
        params: Model parameters or parameter groups
        lr: Learning rate
        momentum: Momentum factor
        dampening: Dampening for momentum
        weight_decay: Weight decay coefficient
        nesterov: Whether to use Nesterov momentum
        **kwargs: Additional optimizer arguments
        
    Returns:
        Optimizer: SGD optimizer instance
    """
    return torch.optim.SGD(
        params,
        lr=lr,
        momentum=momentum,
        dampening=dampening,
        weight_decay=weight_decay,
        nesterov=nesterov,
        **kwargs
    )


@register_optimizer("rmsprop")
def build_rmsprop(
    params: Union[Iterable[torch.Tensor], Iterable[Dict[str, Any]]],
    lr: float = 1e-2,
    alpha: float = 0.99,
    eps: float = 1e-8,
    weight_decay: float = 0.0,
    momentum: float = 0.0,
    **kwargs
) -> Optimizer:
    """
    Build RMSprop optimizer.
    
    Args:
        params: Model parameters or parameter groups
        lr: Learning rate
        alpha: Smoothing constant
        eps: Term for numerical stability
        weight_decay: Weight decay coefficient
        momentum: Momentum factor
        **kwargs: Additional optimizer arguments
        
    Returns:
        Optimizer: RMSprop optimizer instance
    """
    return torch.optim.RMSprop(
        params,
        lr=lr,
        alpha=alpha,
        eps=eps,
        weight_decay=weight_decay,
        momentum=momentum,
        **kwargs
    )


def build_optimizer(
    optimizer_type: str,
    model: nn.Module,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    exclude_bias_and_norm: bool = True,
    **optimizer_kwargs
) -> Optimizer:
    """
    Convenience function to build an optimizer.
    
    Args:
        optimizer_type: Type of optimizer
        model: Model to optimize
        lr: Learning rate
        weight_decay: Weight decay coefficient
        exclude_bias_and_norm: If True, excludes bias and norm params from weight decay
        **optimizer_kwargs: Additional optimizer arguments
        
    Returns:
        Optimizer: Configured optimizer instance
        
    Examples:
        >>> optimizer = build_optimizer("adamw", model, lr=1e-3, weight_decay=0.01)
    """
    builder = OptimizerBuilder(
        optimizer_type=optimizer_type,
        lr=lr,
        weight_decay=weight_decay,
        exclude_bias_and_norm=exclude_bias_and_norm,
        **optimizer_kwargs
    )
    return builder.build(model)


def get_optimizer_names() -> List[str]:
    """
    Get list of registered optimizer names.
    
    Returns:
        List of optimizer names
    """
    return OPTIMIZER_REGISTRY.list()
