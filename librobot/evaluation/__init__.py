"""Evaluation package for LibroBot VLA."""

from .benchmarks import (
    AbstractBenchmark,
    BenchmarkConfig,
    BenchmarkSuite,
    BridgeBenchmark,
    EpisodeResult,
    LIBEROBenchmark,
    RealWorldBenchmark,
    SimplerBenchmark,
    SimulationBenchmark,
    create_benchmark,
    create_standard_suite,
)
from .metrics import (
    BENCHMARK_CONFIGS,
    MAE,
    MSE,
    EpisodeReturn,
    MetricBase,
    MetricCollection,
    PositionError,
    RotationError,
    Smoothness,
    SuccessRate,
    TrajectoryLength,
    create_metrics,
)

__all__ = [
    # Metrics
    "MetricBase",
    "MSE",
    "MAE",
    "SuccessRate",
    "PositionError",
    "RotationError",
    "TrajectoryLength",
    "Smoothness",
    "EpisodeReturn",
    "MetricCollection",
    "create_metrics",
    "BENCHMARK_CONFIGS",
    # Benchmarks
    "BenchmarkConfig",
    "EpisodeResult",
    "AbstractBenchmark",
    "SimulationBenchmark",
    "BridgeBenchmark",
    "LIBEROBenchmark",
    "SimplerBenchmark",
    "RealWorldBenchmark",
    "BenchmarkSuite",
    "create_benchmark",
    "create_standard_suite",
]
