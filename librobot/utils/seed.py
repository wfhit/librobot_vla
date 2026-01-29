"""Random seed utilities for reproducible experiments."""

import os
import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = False, benchmark: bool = True) -> None:
    """
    Set random seeds for reproducibility across all libraries.
    
    Args:
        seed: Random seed value
        deterministic: If True, makes PyTorch operations deterministic (slower)
        benchmark: If True, enables cuDNN benchmarking (faster but less reproducible)
        
    Examples:
        >>> set_seed(42)
        >>> set_seed(42, deterministic=True, benchmark=False)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Note: PYTHONHASHSEED must be set before Python starts to take effect
    # Setting it here only affects subprocesses
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.benchmark = benchmark


def get_random_state() -> dict:
    """
    Get the current random state for all RNG libraries.
    
    Returns:
        dict: Dictionary containing random states for random, numpy, and torch
    """
    return {
        'random': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
        'torch_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def set_random_state(state: dict) -> None:
    """
    Restore random state for all RNG libraries.
    
    Args:
        state: Dictionary containing random states (from get_random_state)
    """
    random.setstate(state['random'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    
    if state.get('torch_cuda') is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['torch_cuda'])


def make_deterministic(seed: Optional[int] = None) -> int:
    """
    Make everything deterministic for perfect reproducibility.
    
    Args:
        seed: Random seed. If None, generates one from system entropy
        
    Returns:
        int: The seed that was used
    """
    if seed is None:
        seed = int.from_bytes(os.urandom(4), byteorder='big')
    
    set_seed(seed, deterministic=True, benchmark=False)
    return seed
