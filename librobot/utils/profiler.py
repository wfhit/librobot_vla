"""Performance profiling utilities for debugging and optimization."""

import time
import cProfile
import pstats
from io import StringIO
from pathlib import Path
from typing import Optional, Union, Dict, Any, Callable
from contextlib import contextmanager
import torch
from torch.profiler import profile, ProfilerActivity, schedule, tensorboard_trace_handler


class Profiler:
    """
    Python profiler wrapper for performance analysis.

    Examples:
        >>> profiler = Profiler()
        >>> profiler.start()
        >>> expensive_function()
        >>> profiler.stop()
        >>> profiler.print_stats(top=20)
    """

    def __init__(self):
        """Initialize profiler."""
        self.profiler: Optional[cProfile.Profile] = None
        self.stats: Optional[pstats.Stats] = None

    def start(self) -> None:
        """Start profiling."""
        self.profiler = cProfile.Profile()
        self.profiler.enable()

    def stop(self) -> None:
        """Stop profiling."""
        if self.profiler is None:
            raise RuntimeError("Profiler was not started")

        self.profiler.disable()

        stream = StringIO()
        self.stats = pstats.Stats(self.profiler, stream=stream)

    def print_stats(self, top: int = 20, sort_by: str = 'cumulative') -> None:
        """
        Print profiling statistics.

        Args:
            top: Number of top functions to display
            sort_by: Sort key ('cumulative', 'time', 'calls', etc.)
        """
        if self.stats is None:
            raise RuntimeError("No stats available. Run start() and stop() first")

        print("\n" + "="*80)
        print(f"Profiling Results (Top {top} by {sort_by})")
        print("="*80)

        self.stats.sort_stats(sort_by)
        self.stats.print_stats(top)

    def save_stats(self, path: Union[str, Path]) -> None:
        """
        Save profiling statistics to file.

        Args:
            path: Output file path
        """
        if self.profiler is None:
            raise RuntimeError("Profiler was not started")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.profiler.dump_stats(str(path))

    @contextmanager
    def profile(self, name: str = "", print_stats: bool = True, top: int = 20):
        """
        Context manager for profiling a code block.

        Args:
            name: Name for this profiling session
            print_stats: If True, prints statistics after profiling
            top: Number of top functions to display

        Yields:
            Profiler instance
        """
        if name:
            print(f"\nProfiling: {name}")

        self.start()
        try:
            yield self
        finally:
            self.stop()
            if print_stats:
                self.print_stats(top=top)


class TorchProfiler:
    """
    PyTorch profiler wrapper for GPU/CPU performance analysis.

    Examples:
        >>> profiler = TorchProfiler(log_dir="./logs")
        >>> with profiler.profile():
        ...     model(input)
    """

    def __init__(
        self,
        log_dir: Optional[Union[str, Path]] = None,
        activities: Optional[list] = None,
        record_shapes: bool = True,
        profile_memory: bool = True,
        with_stack: bool = False,
    ):
        """
        Initialize PyTorch profiler.

        Args:
            log_dir: Directory to save profiling results
            activities: List of activities to profile (CPU, CUDA)
            record_shapes: If True, records tensor shapes
            profile_memory: If True, profiles memory usage
            with_stack: If True, records Python stack traces
        """
        self.log_dir = Path(log_dir) if log_dir else None

        if activities is None:
            activities = [ProfilerActivity.CPU]
            if torch.cuda.is_available():
                activities.append(ProfilerActivity.CUDA)

        self.activities = activities
        self.record_shapes = record_shapes
        self.profile_memory = profile_memory
        self.with_stack = with_stack
        self.profiler: Optional[profile] = None

    @contextmanager
    def profile(
        self,
        wait: int = 1,
        warmup: int = 1,
        active: int = 3,
        repeat: int = 1,
    ):
        """
        Context manager for profiling with scheduling.

        Args:
            wait: Number of steps to skip at the start
            warmup: Number of warmup steps
            active: Number of active profiling steps
            repeat: Number of times to repeat the cycle

        Yields:
            torch.profiler.profile instance
        """
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            on_trace_ready = tensorboard_trace_handler(str(self.log_dir))
        else:
            on_trace_ready = None

        with profile(
            activities=self.activities,
            schedule=schedule(wait=wait, warmup=warmup, active=active, repeat=repeat),
            on_trace_ready=on_trace_ready,
            record_shapes=self.record_shapes,
            profile_memory=self.profile_memory,
            with_stack=self.with_stack,
        ) as prof:
            self.profiler = prof
            yield prof

    @contextmanager
    def simple_profile(self):
        """
        Simple context manager for basic profiling without scheduling.

        Yields:
            torch.profiler.profile instance
        """
        with profile(
            activities=self.activities,
            record_shapes=self.record_shapes,
            profile_memory=self.profile_memory,
            with_stack=self.with_stack,
        ) as prof:
            self.profiler = prof
            yield prof

    def print_stats(
        self,
        sort_by: str = "cpu_time_total",
        row_limit: int = 20,
    ) -> None:
        """
        Print profiling statistics.

        Args:
            sort_by: Sort key for results
            row_limit: Maximum number of rows to display
        """
        if self.profiler is None:
            raise RuntimeError("No profiling data available")

        print("\n" + "="*80)
        print("PyTorch Profiling Results")
        print("="*80)
        print(self.profiler.key_averages().table(sort_by=sort_by, row_limit=row_limit))
        print("="*80 + "\n")

    def export_chrome_trace(self, path: Union[str, Path]) -> None:
        """
        Export profiling results to Chrome trace format.

        Args:
            path: Output file path
        """
        if self.profiler is None:
            raise RuntimeError("No profiling data available")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.profiler.export_chrome_trace(str(path))


def profile_function(func: Callable, *args, **kwargs) -> Dict[str, Any]:
    """
    Profile a function call and return timing information.

    Args:
        func: Function to profile
        *args: Function arguments
        **kwargs: Function keyword arguments

    Returns:
        Dictionary with timing and result information
    """
    start_time = time.perf_counter()
    start_cpu = time.process_time()

    result = func(*args, **kwargs)

    wall_time = time.perf_counter() - start_time
    cpu_time = time.process_time() - start_cpu

    return {
        'result': result,
        'wall_time': wall_time,
        'cpu_time': cpu_time,
        'function': func.__name__,
    }


def benchmark_function(
    func: Callable,
    *args,
    num_iterations: int = 100,
    warmup: int = 10,
    **kwargs
) -> Dict[str, float]:
    """
    Benchmark a function with multiple iterations.

    Args:
        func: Function to benchmark
        *args: Function arguments
        num_iterations: Number of iterations to run
        warmup: Number of warmup iterations
        **kwargs: Function keyword arguments

    Returns:
        Dictionary with benchmark statistics
    """
    import numpy as np

    # Warmup
    for _ in range(warmup):
        func(*args, **kwargs)

    # Benchmark
    times = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        func(*args, **kwargs)
        times.append(time.perf_counter() - start)

    times = np.array(times)

    return {
        'mean': float(np.mean(times)),
        'std': float(np.std(times)),
        'min': float(np.min(times)),
        'max': float(np.max(times)),
        'median': float(np.median(times)),
        'iterations': num_iterations,
    }
