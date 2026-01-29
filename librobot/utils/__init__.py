"""Utilities package for LibroBot VLA framework."""

from .registry import Registry, GlobalRegistry, RegistryError, build_from_config
from .config import Config, load_config, merge_configs, create_config
from .logging import (
    Logger,
    get_logger,
    get_default_logger,
    setup_logging,
    debug,
    info,
    warning,
    error,
    critical,
    exception,
    DEBUG,
    INFO,
    WARNING,
    ERROR,
    CRITICAL,
)
from .checkpoint import Checkpoint, save_checkpoint, load_checkpoint
from .profiler import Profiler, TorchProfiler, profile_function, benchmark_function
from .memory import (
    get_memory_info,
    print_memory_info,
    clear_memory,
    optimize_memory,
    MemoryTracker,
    set_memory_growth,
    get_optimal_batch_size,
)
from .seed import set_seed, get_random_state, set_random_state, make_deterministic
from .timer import Timer, TimerRegistry, get_global_timer_registry
from .io import (
    save_json,
    load_json,
    save_yaml,
    load_yaml,
    save_pickle,
    load_pickle,
    save_torch,
    load_torch,
    ensure_dir,
    read_text,
    write_text,
    read_lines,
    write_lines,
)

__all__ = [
    # Registry
    "Registry",
    "GlobalRegistry",
    "RegistryError",
    "build_from_config",
    # Config
    "Config",
    "load_config",
    "merge_configs",
    "create_config",
    # Logging
    "Logger",
    "get_logger",
    "get_default_logger",
    "setup_logging",
    "debug",
    "info",
    "warning",
    "error",
    "critical",
    "exception",
    "DEBUG",
    "INFO",
    "WARNING",
    "ERROR",
    "CRITICAL",
    # Checkpoint
    "Checkpoint",
    "save_checkpoint",
    "load_checkpoint",
    # Profiler
    "Profiler",
    "TorchProfiler",
    "profile_function",
    "benchmark_function",
    # Memory
    "get_memory_info",
    "print_memory_info",
    "clear_memory",
    "optimize_memory",
    "MemoryTracker",
    "set_memory_growth",
    "get_optimal_batch_size",
    # Seed
    "set_seed",
    "get_random_state",
    "set_random_state",
    "make_deterministic",
    # Timer
    "Timer",
    "TimerRegistry",
    "get_global_timer_registry",
    # I/O
    "save_json",
    "load_json",
    "save_yaml",
    "load_yaml",
    "save_pickle",
    "load_pickle",
    "save_torch",
    "load_torch",
    "ensure_dir",
    "read_text",
    "write_text",
    "read_lines",
    "write_lines",
]
