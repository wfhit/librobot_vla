"""Evaluation package for LibroBot VLA."""

from .metrics import (
    MetricBase,
    MSE,
    MAE,
    SuccessRate,
    PositionError,
    RotationError,
    TrajectoryLength,
    Smoothness,
    EpisodeReturn,
    MetricCollection,
    create_metrics,
    BENCHMARK_CONFIGS,
)

from .benchmarks import (
    BenchmarkConfig,
    EpisodeResult,
    AbstractBenchmark,
    SimulationBenchmark,
    BridgeBenchmark,
    LIBEROBenchmark,
    SimplerBenchmark,
    RealWorldBenchmark,
    BenchmarkSuite,
    create_benchmark,
    create_standard_suite,
)

__all__ = [
    # Metrics
    'MetricBase',
    'MSE',
    'MAE',
    'SuccessRate',
    'PositionError',
    'RotationError',
    'TrajectoryLength',
    'Smoothness',
    'EpisodeReturn',
    'MetricCollection',
    'create_metrics',
    'BENCHMARK_CONFIGS',
    # Benchmarks
    'BenchmarkConfig',
    'EpisodeResult',
    'AbstractBenchmark',
    'SimulationBenchmark',
    'BridgeBenchmark',
    'LIBEROBenchmark',
    'SimplerBenchmark',
    'RealWorldBenchmark',
    'BenchmarkSuite',
    'create_benchmark',
    'create_standard_suite',
]