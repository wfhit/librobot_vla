"""Buffers for action smoothing and history."""

from collections import deque
from typing import Any, Optional

import numpy as np


class ActionBuffer:
    """Buffer for storing and retrieving actions."""

    def __init__(
        self,
        buffer_size: int = 10,
        action_dim: int = 7,
    ):
        """
        Args:
            buffer_size: Maximum buffer size
            action_dim: Action dimension
        """
        self.buffer_size = buffer_size
        self.action_dim = action_dim
        self._buffer: deque[np.ndarray] = deque(maxlen=buffer_size)

    def add(self, action: np.ndarray) -> None:
        """Add action to buffer."""
        self._buffer.append(np.asarray(action))

    def get_latest(self) -> Optional[np.ndarray]:
        """Get most recent action."""
        return self._buffer[-1] if self._buffer else None

    def get_history(self, n: int = None) -> np.ndarray:
        """Get history of actions."""
        n = n or len(self._buffer)
        if not self._buffer:
            return np.zeros((0, self.action_dim))
        return np.stack(list(self._buffer)[-n:])

    def clear(self) -> None:
        """Clear buffer."""
        self._buffer.clear()

    def __len__(self) -> int:
        return len(self._buffer)


class ActionSmoothingBuffer:
    """Buffer for temporal smoothing of actions."""

    def __init__(
        self,
        window_size: int = 5,
        action_dim: int = 7,
        smoothing_type: str = "exponential",
        alpha: float = 0.7,
    ):
        """
        Args:
            window_size: Size of smoothing window
            action_dim: Action dimension
            smoothing_type: Type of smoothing ("moving_average", "exponential", "gaussian")
            alpha: Smoothing factor (for exponential)
        """
        self.window_size = window_size
        self.action_dim = action_dim
        self.smoothing_type = smoothing_type
        self.alpha = alpha

        self._buffer: deque[np.ndarray] = deque(maxlen=window_size)
        self._smoothed_action: Optional[np.ndarray] = None

    def add(self, action: np.ndarray) -> np.ndarray:
        """Add action and return smoothed version."""
        action = np.asarray(action)
        self._buffer.append(action)

        if self.smoothing_type == "moving_average":
            return self._moving_average()
        elif self.smoothing_type == "exponential":
            return self._exponential_smoothing(action)
        elif self.smoothing_type == "gaussian":
            return self._gaussian_smoothing()

        return action

    def _moving_average(self) -> np.ndarray:
        """Simple moving average."""
        if not self._buffer:
            return np.zeros(self.action_dim)
        return np.mean(list(self._buffer), axis=0)

    def _exponential_smoothing(self, new_action: np.ndarray) -> np.ndarray:
        """Exponential moving average."""
        if self._smoothed_action is None:
            self._smoothed_action = new_action
        else:
            self._smoothed_action = self.alpha * new_action + (1 - self.alpha) * self._smoothed_action
        return self._smoothed_action

    def _gaussian_smoothing(self) -> np.ndarray:
        """Gaussian-weighted moving average."""
        if not self._buffer:
            return np.zeros(self.action_dim)

        n = len(self._buffer)
        weights = np.exp(-np.linspace(0, 2, n) ** 2)
        weights = weights / weights.sum()

        actions = np.array(list(self._buffer))
        return np.sum(actions * weights[:, np.newaxis], axis=0)

    def get_smoothed(self) -> Optional[np.ndarray]:
        """Get current smoothed action."""
        return self._smoothed_action

    def reset(self) -> None:
        """Reset buffer."""
        self._buffer.clear()
        self._smoothed_action = None


class HistoryBuffer:
    """Buffer for observation/action history."""

    def __init__(
        self,
        history_length: int = 4,
        keys: list[str] = None,
    ):
        """
        Args:
            history_length: Length of history to maintain
            keys: Keys to track
        """
        self.history_length = history_length
        self.keys = keys or ['images', 'proprioception', 'actions']
        self._buffers: dict[str, deque] = {k: deque(maxlen=history_length) for k in self.keys}

    def add(self, data: dict[str, Any]) -> None:
        """Add data to history."""
        for key in self.keys:
            if key in data:
                self._buffers[key].append(np.asarray(data[key]))

    def get(self, key: str, n: int = None) -> np.ndarray:
        """Get history for a key."""
        n = n or self.history_length
        buffer = self._buffers.get(key, deque())
        if not buffer:
            return None
        return np.stack(list(buffer)[-n:])

    def get_all(self) -> dict[str, np.ndarray]:
        """Get all history."""
        return {k: self.get(k) for k in self.keys if self._buffers[k]}

    def reset(self) -> None:
        """Reset all buffers."""
        for buffer in self._buffers.values():
            buffer.clear()


class ActionChunkBuffer:
    """Buffer for action chunking (ACT-style)."""

    def __init__(
        self,
        chunk_size: int = 10,
        action_dim: int = 7,
        temporal_aggregation: str = "first",
    ):
        """
        Args:
            chunk_size: Size of action chunks
            action_dim: Action dimension
            temporal_aggregation: How to aggregate chunks ("first", "mean", "exponential")
        """
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        self.temporal_aggregation = temporal_aggregation

        self._chunk: Optional[np.ndarray] = None
        self._index: int = 0
        self._chunk_history: deque[np.ndarray] = deque(maxlen=3)

    def set_chunk(self, chunk: np.ndarray) -> None:
        """Set a new action chunk."""
        self._chunk_history.append(chunk)
        self._chunk = chunk
        self._index = 0

    def get_action(self) -> Optional[np.ndarray]:
        """Get next action from chunk."""
        if self._chunk is None or self._index >= len(self._chunk):
            return None

        if self.temporal_aggregation == "first":
            action = self._chunk[self._index]
        elif self.temporal_aggregation == "mean" and len(self._chunk_history) > 1:
            # Weighted average of overlapping chunks
            actions = []
            weights = []
            for i, chunk in enumerate(self._chunk_history):
                offset = (len(self._chunk_history) - 1 - i) * self.chunk_size
                idx = self._index + offset
                if 0 <= idx < len(chunk):
                    actions.append(chunk[idx])
                    weights.append(1.0 / (i + 1))
            if actions:
                weights = np.array(weights) / sum(weights)
                action = np.average(actions, axis=0, weights=weights)
            else:
                action = self._chunk[self._index]
        else:
            action = self._chunk[self._index]

        self._index += 1
        return action

    def needs_new_chunk(self) -> bool:
        """Check if new chunk is needed."""
        return self._chunk is None or self._index >= len(self._chunk)

    def reset(self) -> None:
        """Reset buffer."""
        self._chunk = None
        self._index = 0
        self._chunk_history.clear()


__all__ = [
    'ActionBuffer',
    'ActionSmoothingBuffer',
    'HistoryBuffer',
    'ActionChunkBuffer',
]
