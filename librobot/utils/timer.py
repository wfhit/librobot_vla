"""Timing utilities for performance measurement."""

import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Optional


class Timer:
    """
    Context manager and decorator for timing code execution.

    Examples:
        >>> # As context manager
        >>> with Timer("my_operation"):
        ...     expensive_function()

        >>> # As decorator
        >>> @Timer.decorator("function_name")
        ... def my_function():
        ...     pass

        >>> # Manual timing
        >>> timer = Timer()
        >>> timer.start()
        >>> # ... do work ...
        >>> elapsed = timer.stop()
    """

    def __init__(self, name: Optional[str] = None, verbose: bool = True):
        """
        Initialize timer.

        Args:
            name: Name for this timer
            verbose: If True, prints timing information
        """
        self.name = name
        self.verbose = verbose
        self.start_time: Optional[float] = None
        self.elapsed: Optional[float] = None

    def start(self) -> None:
        """Start the timer."""
        self.start_time = time.perf_counter()

    def stop(self) -> float:
        """
        Stop the timer and return elapsed time.

        Returns:
            float: Elapsed time in seconds
        """
        if self.start_time is None:
            raise RuntimeError("Timer was not started")

        self.elapsed = time.perf_counter() - self.start_time

        if self.verbose and self.name:
            print(f"[{self.name}] Elapsed time: {self.elapsed:.4f}s")

        return self.elapsed

    def __enter__(self):
        """Enter context manager."""
        self.start()
        return self

    def __exit__(self, *args):
        """Exit context manager."""
        self.stop()

    @staticmethod
    def decorator(name: Optional[str] = None, verbose: bool = True):
        """
        Create a decorator that times function execution.

        Args:
            name: Name for the timer. If None, uses function name
            verbose: If True, prints timing information

        Returns:
            Decorated function
        """
        from functools import wraps

        def decorator_wrapper(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                timer_name = name or func.__name__
                with Timer(timer_name, verbose):
                    return func(*args, **kwargs)
            return wrapper
        return decorator_wrapper


class TimerRegistry:
    """
    Thread-safe registry for collecting multiple timing measurements.

    Examples:
        >>> registry = TimerRegistry()
        >>> with registry.timer("operation_1"):
        ...     pass
        >>> with registry.timer("operation_2"):
        ...     pass
        >>> stats = registry.get_stats()
    """

    def __init__(self):
        """Initialize timer registry."""
        self._timings: dict[str, list[float]] = defaultdict(list)
        self._lock = threading.Lock()

    @contextmanager
    def timer(self, name: str, verbose: bool = False):
        """
        Create a named timer context.

        Args:
            name: Name for this timing
            verbose: If True, prints timing information

        Yields:
            Timer instance
        """
        timer = Timer(name=name if verbose else None, verbose=verbose)
        timer.start()

        try:
            yield timer
        finally:
            elapsed = timer.stop()
            with self._lock:
                self._timings[name].append(elapsed)

    def record(self, name: str, elapsed: float) -> None:
        """
        Manually record a timing.

        Args:
            name: Name for this timing
            elapsed: Elapsed time in seconds
        """
        with self._lock:
            self._timings[name].append(elapsed)

    def get_timings(self, name: str) -> list[float]:
        """
        Get all timings for a specific name.

        Args:
            name: Timer name

        Returns:
            List of timing measurements
        """
        with self._lock:
            return self._timings[name].copy()

    def get_stats(self) -> dict[str, dict[str, float]]:
        """
        Get statistics for all timers.

        Returns:
            Dictionary mapping timer names to statistics (mean, std, min, max, total, count)
        """
        import numpy as np

        stats = {}
        with self._lock:
            for name, timings in self._timings.items():
                if timings:
                    stats[name] = {
                        'mean': float(np.mean(timings)),
                        'std': float(np.std(timings)),
                        'min': float(np.min(timings)),
                        'max': float(np.max(timings)),
                        'total': float(np.sum(timings)),
                        'count': len(timings),
                    }

        return stats

    def reset(self) -> None:
        """Clear all recorded timings."""
        with self._lock:
            self._timings.clear()

    def print_stats(self) -> None:
        """Print formatted statistics for all timers."""
        stats = self.get_stats()

        if not stats:
            print("No timings recorded")
            return

        print("\n" + "="*80)
        print("Timing Statistics")
        print("="*80)
        print(f"{'Name':<30} {'Count':>8} {'Mean':>12} {'Std':>12} {'Min':>12} {'Max':>12}")
        print("-"*80)

        for name in sorted(stats.keys()):
            timing_stat = stats[name]
            print(f"{name:<30} {timing_stat['count']:>8} {timing_stat['mean']:>12.4f}s {timing_stat['std']:>12.4f}s "
                  f"{timing_stat['min']:>12.4f}s {timing_stat['max']:>12.4f}s")

        print("="*80 + "\n")


_global_registry = TimerRegistry()


def get_global_timer_registry() -> TimerRegistry:
    """
    Get the global timer registry.

    Returns:
        TimerRegistry: Global registry instance
    """
    return _global_registry
