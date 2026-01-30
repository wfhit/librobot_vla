"""Evaluation metrics and benchmarks for VLA models."""

from typing import Optional

import numpy as np


class MetricBase:
    """Base class for evaluation metrics."""

    def __init__(self, name: str):
        self.name = name
        self._values: list[float] = []

    def reset(self) -> None:
        """Reset accumulated values."""
        self._values = []

    def update(self, *args, **kwargs) -> None:
        """Update metric with new data."""
        raise NotImplementedError

    def compute(self) -> float:
        """Compute final metric value."""
        if not self._values:
            return 0.0
        return np.mean(self._values)

    def __repr__(self) -> str:
        return f"{self.name}: {self.compute():.4f}"


class MSE(MetricBase):
    """Mean Squared Error metric."""

    def __init__(self):
        super().__init__("MSE")

    def update(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
    ) -> None:
        mse = np.mean((predictions - targets) ** 2)
        self._values.append(mse)


class MAE(MetricBase):
    """Mean Absolute Error metric."""

    def __init__(self):
        super().__init__("MAE")

    def update(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
    ) -> None:
        mae = np.mean(np.abs(predictions - targets))
        self._values.append(mae)


class SuccessRate(MetricBase):
    """Task success rate metric."""

    def __init__(self, threshold: float = 0.05):
        super().__init__("Success Rate")
        self.threshold = threshold

    def update(
        self,
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray,
    ) -> None:
        distance = np.linalg.norm(achieved_goal - desired_goal)
        success = float(distance < self.threshold)
        self._values.append(success)


class PositionError(MetricBase):
    """End-effector position error metric."""

    def __init__(self):
        super().__init__("Position Error")

    def update(
        self,
        predicted_pos: np.ndarray,
        target_pos: np.ndarray,
    ) -> None:
        error = np.linalg.norm(predicted_pos - target_pos)
        self._values.append(error)


class RotationError(MetricBase):
    """Rotation error metric (in degrees)."""

    def __init__(self):
        super().__init__("Rotation Error")

    def update(
        self,
        predicted_rot: np.ndarray,  # quaternion
        target_rot: np.ndarray,
    ) -> None:
        # Compute angular difference
        dot = np.abs(np.dot(predicted_rot, target_rot))
        dot = np.clip(dot, -1.0, 1.0)
        angle = 2 * np.arccos(dot) * 180 / np.pi
        self._values.append(angle)


class TrajectoryLength(MetricBase):
    """Trajectory length metric."""

    def __init__(self):
        super().__init__("Trajectory Length")

    def update(self, trajectory: np.ndarray) -> None:
        """
        Args:
            trajectory: [T, D] trajectory
        """
        if len(trajectory) < 2:
            return

        length = np.sum(np.linalg.norm(np.diff(trajectory, axis=0), axis=1))
        self._values.append(length)


class Smoothness(MetricBase):
    """Action smoothness metric (lower is smoother)."""

    def __init__(self):
        super().__init__("Smoothness")

    def update(self, actions: np.ndarray) -> None:
        """
        Args:
            actions: [T, D] action sequence
        """
        if len(actions) < 3:
            return

        # Compute second derivative (acceleration)
        acc = np.diff(actions, n=2, axis=0)
        smoothness = np.mean(np.abs(acc))
        self._values.append(smoothness)


class EpisodeReturn(MetricBase):
    """Cumulative episode return."""

    def __init__(self, gamma: float = 0.99):
        super().__init__("Episode Return")
        self.gamma = gamma

    def update(self, rewards: np.ndarray) -> None:
        """
        Args:
            rewards: [T] reward sequence
        """
        T = len(rewards)
        discounts = self.gamma ** np.arange(T)
        ret = np.sum(rewards * discounts)
        self._values.append(ret)


class MetricCollection:
    """Collection of metrics for comprehensive evaluation."""

    def __init__(self, metrics: Optional[list[MetricBase]] = None):
        """
        Args:
            metrics: List of metrics to track
        """
        self.metrics = metrics or [
            MSE(),
            MAE(),
            SuccessRate(),
            Smoothness(),
        ]
        self._metric_dict = {m.name: m for m in self.metrics}

    def reset(self) -> None:
        """Reset all metrics."""
        for metric in self.metrics:
            metric.reset()

    def update(self, **kwargs) -> None:
        """Update all applicable metrics."""
        for metric in self.metrics:
            try:
                metric.update(**kwargs)
            except (TypeError, KeyError):
                pass  # Metric doesn't accept these arguments

    def compute(self) -> dict[str, float]:
        """Compute all metrics."""
        return {m.name: m.compute() for m in self.metrics}

    def __getitem__(self, name: str) -> MetricBase:
        return self._metric_dict[name]


# Benchmark configurations
BENCHMARK_CONFIGS = {
    "bridge": {
        "tasks": ["pick", "place", "stack"],
        "metrics": ["success_rate", "position_error"],
        "num_episodes": 50,
    },
    "simpler": {
        "tasks": ["reach", "push", "drawer"],
        "metrics": ["success_rate", "episode_length"],
        "num_episodes": 100,
    },
    "libero": {
        "tasks": ["libero_90", "libero_goal", "libero_object"],
        "metrics": ["success_rate"],
        "num_episodes": 50,
    },
}


def create_metrics(metric_names: list[str]) -> MetricCollection:
    """Create metrics collection from names."""
    metric_map = {
        "mse": MSE,
        "mae": MAE,
        "success_rate": SuccessRate,
        "position_error": PositionError,
        "rotation_error": RotationError,
        "trajectory_length": TrajectoryLength,
        "smoothness": Smoothness,
        "episode_return": EpisodeReturn,
    }

    metrics = []
    for name in metric_names:
        if name.lower() in metric_map:
            metrics.append(metric_map[name.lower()]())

    return MetricCollection(metrics)


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
