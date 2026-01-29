"""Memory management utilities for efficient resource usage."""

import gc
import os
from typing import Optional, Dict, Any
import psutil
import torch


def get_memory_info() -> Dict[str, Any]:
    """
    Get current memory usage information.
    
    Returns:
        Dictionary containing memory statistics for CPU and GPU
    """
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    
    info = {
        'cpu': {
            'rss_mb': mem_info.rss / 1024 / 1024,  # Resident Set Size
            'vms_mb': mem_info.vms / 1024 / 1024,  # Virtual Memory Size
            'percent': process.memory_percent(),
        },
        'system': {
            'available_mb': psutil.virtual_memory().available / 1024 / 1024,
            'total_mb': psutil.virtual_memory().total / 1024 / 1024,
            'percent': psutil.virtual_memory().percent,
        }
    }
    
    if torch.cuda.is_available():
        info['gpu'] = {}
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024 / 1024
            reserved = torch.cuda.memory_reserved(i) / 1024 / 1024
            total = torch.cuda.get_device_properties(i).total_memory / 1024 / 1024
            
            info['gpu'][f'cuda:{i}'] = {
                'allocated_mb': allocated,
                'reserved_mb': reserved,
                'total_mb': total,
                'free_mb': total - allocated,
            }
    
    return info


def print_memory_info() -> None:
    """Print formatted memory information."""
    info = get_memory_info()
    
    print("\n" + "="*60)
    print("Memory Usage")
    print("="*60)
    
    print(f"\nCPU:")
    print(f"  RSS: {info['cpu']['rss_mb']:.2f} MB")
    print(f"  VMS: {info['cpu']['vms_mb']:.2f} MB")
    print(f"  Percent: {info['cpu']['percent']:.2f}%")
    
    print(f"\nSystem:")
    print(f"  Total: {info['system']['total_mb']:.2f} MB")
    print(f"  Available: {info['system']['available_mb']:.2f} MB")
    print(f"  Percent: {info['system']['percent']:.2f}%")
    
    if 'gpu' in info:
        print(f"\nGPU:")
        for device, stats in info['gpu'].items():
            print(f"  {device}:")
            print(f"    Allocated: {stats['allocated_mb']:.2f} MB")
            print(f"    Reserved: {stats['reserved_mb']:.2f} MB")
            print(f"    Total: {stats['total_mb']:.2f} MB")
            print(f"    Free: {stats['free_mb']:.2f} MB")
    
    print("="*60 + "\n")


def clear_memory(device: Optional[str] = None) -> None:
    """
    Clear memory by running garbage collection and emptying CUDA cache.
    
    Args:
        device: Specific CUDA device to clear (e.g., 'cuda:0'). If None, clears all devices
    """
    gc.collect()
    
    if torch.cuda.is_available():
        if device is not None:
            with torch.cuda.device(device):
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        else:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


def optimize_memory() -> None:
    """
    Optimize memory usage with aggressive garbage collection and cache clearing.
    """
    for _ in range(3):
        gc.collect()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                torch.cuda.empty_cache()


class MemoryTracker:
    """
    Context manager for tracking memory usage.
    
    Examples:
        >>> with MemoryTracker("my_operation") as tracker:
        ...     expensive_function()
        >>> print(f"Memory delta: {tracker.delta_mb:.2f} MB")
    """
    
    def __init__(self, name: str = "", verbose: bool = True):
        """
        Initialize memory tracker.
        
        Args:
            name: Name for this tracking session
            verbose: If True, prints memory information
        """
        self.name = name
        self.verbose = verbose
        self.start_info: Optional[Dict] = None
        self.end_info: Optional[Dict] = None
        self.delta_mb: float = 0.0
    
    def __enter__(self):
        """Enter context and record starting memory."""
        clear_memory()
        self.start_info = get_memory_info()
        return self
    
    def __exit__(self, *args):
        """Exit context and record ending memory."""
        clear_memory()
        self.end_info = get_memory_info()
        
        start_cpu = self.start_info['cpu']['rss_mb']
        end_cpu = self.end_info['cpu']['rss_mb']
        self.delta_mb = end_cpu - start_cpu
        
        if self.verbose:
            name_str = f"[{self.name}] " if self.name else ""
            print(f"{name_str}Memory delta: {self.delta_mb:+.2f} MB "
                  f"(Start: {start_cpu:.2f} MB, End: {end_cpu:.2f} MB)")
            
            if 'gpu' in self.end_info:
                for device in self.end_info['gpu']:
                    start_alloc = self.start_info['gpu'][device]['allocated_mb']
                    end_alloc = self.end_info['gpu'][device]['allocated_mb']
                    delta_alloc = end_alloc - start_alloc
                    print(f"{name_str}GPU {device} delta: {delta_alloc:+.2f} MB "
                          f"(Start: {start_alloc:.2f} MB, End: {end_alloc:.2f} MB)")


def set_memory_growth(enable: bool = True) -> None:
    """
    Enable/disable CUDA memory growth to prevent OOM errors.
    
    Args:
        enable: If True, allows PyTorch to allocate memory dynamically
    """
    if torch.cuda.is_available():
        if enable:
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        else:
            os.environ.pop('PYTORCH_CUDA_ALLOC_CONF', None)


def get_optimal_batch_size(
    test_fn,
    start_batch_size: int = 1,
    max_batch_size: int = 1024,
    device: str = 'cuda',
) -> int:
    """
    Find optimal batch size that doesn't cause OOM.
    
    Args:
        test_fn: Function that takes batch_size as argument and runs inference
        start_batch_size: Initial batch size to test
        max_batch_size: Maximum batch size to test
        device: Device to test on
        
    Returns:
        int: Optimal batch size
    """
    batch_size = start_batch_size
    optimal_size = batch_size
    
    while batch_size <= max_batch_size:
        try:
            clear_memory(device)
            test_fn(batch_size)
            optimal_size = batch_size
            batch_size *= 2
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                break
            raise
        finally:
            clear_memory(device)
    
    return optimal_size
