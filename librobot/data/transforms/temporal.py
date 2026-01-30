"""Temporal transforms for sequence data."""

from typing import Any, Optional

import numpy as np


class TemporalTransform:
    """Base class for temporal transforms."""

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Apply transform to sample."""
        return self.transform(sample)

    def transform(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Transform sample. Override in subclasses."""
        return sample


class TemporalSubsample(TemporalTransform):
    """Subsample sequence at regular intervals."""

    def __init__(
        self,
        subsample_rate: int = 1,
        keys: Optional[list[str]] = None,
    ):
        """
        Args:
            subsample_rate: Take every Nth frame
            keys: Keys to subsample (None = all array keys)
        """
        self.subsample_rate = subsample_rate
        self.keys = keys

    def transform(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Subsample temporal sequences."""
        keys = self.keys or list(sample.keys())

        for key in keys:
            if key in sample:
                value = sample[key]
                if isinstance(value, np.ndarray) and value.ndim >= 1:
                    sample[key] = value[::self.subsample_rate]

        return sample


class TemporalCrop(TemporalTransform):
    """Crop temporal sequence to fixed length."""

    def __init__(
        self,
        length: int,
        start: Optional[int] = None,
        random_start: bool = True,
        keys: Optional[list[str]] = None,
    ):
        """
        Args:
            length: Target sequence length
            start: Fixed start index (overrides random)
            random_start: Whether to randomly select start
            keys: Keys to crop
        """
        self.length = length
        self.start = start
        self.random_start = random_start
        self.keys = keys

    def transform(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Crop temporal sequences."""
        keys = self.keys or list(sample.keys())

        # Find sequence length
        seq_len = None
        for key in keys:
            if key in sample and isinstance(sample[key], np.ndarray):
                if sample[key].ndim >= 1:
                    seq_len = sample[key].shape[0]
                    break

        if seq_len is None or seq_len <= self.length:
            return sample

        # Determine start index
        if self.start is not None:
            start = self.start
        elif self.random_start:
            start = np.random.randint(0, seq_len - self.length + 1)
        else:
            start = 0

        end = start + self.length

        # Crop all keys
        for key in keys:
            if key in sample and isinstance(sample[key], np.ndarray):
                if sample[key].ndim >= 1 and sample[key].shape[0] == seq_len:
                    sample[key] = sample[key][start:end]

        return sample


class ActionChunking(TemporalTransform):
    """Create action chunks for ACT-style prediction."""

    def __init__(
        self,
        chunk_size: int = 10,
        action_key: str = "actions",
        output_key: str = "action_chunk",
    ):
        """
        Args:
            chunk_size: Number of future actions per chunk
            action_key: Key for input actions
            output_key: Key for output action chunks
        """
        self.chunk_size = chunk_size
        self.action_key = action_key
        self.output_key = output_key

    def transform(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Create action chunks."""
        if self.action_key not in sample:
            return sample

        actions = sample[self.action_key]

        if actions.ndim == 1:
            # Single action - replicate for chunk
            sample[self.output_key] = np.tile(actions, (self.chunk_size, 1))
        elif actions.ndim == 2:
            # Sequence of actions - take future chunk
            T, D = actions.shape
            if T >= self.chunk_size:
                # Random start for training
                start = np.random.randint(0, T - self.chunk_size + 1)
                sample[self.output_key] = actions[start:start + self.chunk_size]
            else:
                # Pad with last action
                pad_len = self.chunk_size - T
                padding = np.tile(actions[-1:], (pad_len, 1))
                sample[self.output_key] = np.concatenate([actions, padding], axis=0)

        return sample


class TemporalStack(TemporalTransform):
    """Stack temporal observations."""

    def __init__(
        self,
        stack_size: int = 4,
        keys: list[str] = ['images'],
    ):
        """
        Args:
            stack_size: Number of frames to stack
            keys: Keys to stack
        """
        self.stack_size = stack_size
        self.keys = keys
        self._buffers: dict[str, list[np.ndarray]] = {k: [] for k in keys}

    def transform(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Stack temporal observations."""
        for key in self.keys:
            if key in sample:
                current = sample[key]
                self._buffers[key].append(current)

                if len(self._buffers[key]) > self.stack_size:
                    self._buffers[key] = self._buffers[key][-self.stack_size:]

                while len(self._buffers[key]) < self.stack_size:
                    self._buffers[key].insert(0, current)

                sample[f'{key}_stacked'] = np.stack(self._buffers[key])

        return sample

    def reset(self):
        """Reset buffers."""
        self._buffers = {k: [] for k in self.keys}


class FrameSkip(TemporalTransform):
    """Skip frames for temporal abstraction."""

    def __init__(
        self,
        skip: int = 1,
        action_repeat: int = 1,
        keys: Optional[list[str]] = None,
    ):
        """
        Args:
            skip: Number of frames to skip
            action_repeat: Times to repeat action
            keys: Keys to apply frame skip
        """
        self.skip = skip
        self.action_repeat = action_repeat
        self.keys = keys

    def transform(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Apply frame skip."""
        keys = self.keys or list(sample.keys())

        for key in keys:
            if key in sample and isinstance(sample[key], np.ndarray):
                if sample[key].ndim >= 1:
                    sample[key] = sample[key][::self.skip]

        return sample


class DeltaActions(TemporalTransform):
    """Convert to delta actions between consecutive timesteps."""

    def __init__(self, action_key: str = "actions"):
        """
        Args:
            action_key: Key for actions
        """
        self.action_key = action_key

    def transform(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Compute delta actions."""
        if self.action_key not in sample:
            return sample

        actions = sample[self.action_key]

        if actions.ndim >= 2 and actions.shape[0] > 1:
            delta = np.diff(actions, axis=0)
            # Pad to keep same length
            delta = np.concatenate([delta, delta[-1:]], axis=0)
            sample[f'{self.action_key}_delta'] = delta

        return sample


__all__ = [
    'TemporalTransform',
    'TemporalSubsample',
    'TemporalCrop',
    'ActionChunking',
    'TemporalStack',
    'FrameSkip',
    'DeltaActions',
]
