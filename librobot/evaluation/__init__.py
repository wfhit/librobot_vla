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

__all__ = [
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
]