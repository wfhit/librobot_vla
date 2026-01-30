"""State transforms for proprioceptive data."""

from typing import Any, Optional

import numpy as np


class StateTransform:
    """Base class for state transforms."""

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Apply transform to sample."""
        if "proprioception" in sample:
            sample["proprioception"] = self.transform(sample["proprioception"])
        return sample

    def transform(self, state: np.ndarray) -> np.ndarray:
        """Transform state. Override in subclasses."""
        return state


class StateNormalize(StateTransform):
    """Normalize state to standard range."""

    def __init__(
        self,
        mean: Optional[np.ndarray] = None,
        std: Optional[np.ndarray] = None,
        min_val: Optional[np.ndarray] = None,
        max_val: Optional[np.ndarray] = None,
        normalize_type: str = "standard",
    ):
        """
        Args:
            mean: Mean for standard normalization
            std: Std for standard normalization
            min_val: Min for min-max normalization
            max_val: Max for min-max normalization
            normalize_type: "standard" or "minmax"
        """
        self.mean = mean
        self.std = std
        self.min_val = min_val
        self.max_val = max_val
        self.normalize_type = normalize_type

    def fit(self, states: np.ndarray) -> "StateNormalize":
        """Fit normalization statistics from data."""
        self.mean = np.mean(states, axis=0)
        self.std = np.std(states, axis=0) + 1e-6
        self.min_val = np.min(states, axis=0)
        self.max_val = np.max(states, axis=0)
        return self

    def transform(self, state: np.ndarray) -> np.ndarray:
        """Normalize state."""
        if self.normalize_type == "standard":
            if self.mean is not None and self.std is not None:
                return (state - self.mean) / self.std
        elif self.normalize_type == "minmax":
            if self.min_val is not None and self.max_val is not None:
                return (state - self.min_val) / (self.max_val - self.min_val + 1e-8)
        return state

    def inverse_transform(self, state: np.ndarray) -> np.ndarray:
        """Denormalize state."""
        if self.normalize_type == "standard":
            if self.mean is not None and self.std is not None:
                return state * self.std + self.mean
        elif self.normalize_type == "minmax":
            if self.min_val is not None and self.max_val is not None:
                return state * (self.max_val - self.min_val) + self.min_val
        return state


class StateNoise(StateTransform):
    """Add noise to state for augmentation."""

    def __init__(
        self,
        noise_std: float = 0.01,
        noise_type: str = "gaussian",
    ):
        """
        Args:
            noise_std: Standard deviation of noise
            noise_type: Type of noise
        """
        self.noise_std = noise_std
        self.noise_type = noise_type

    def transform(self, state: np.ndarray) -> np.ndarray:
        """Add noise to state."""
        if self.noise_type == "gaussian":
            noise = np.random.normal(0, self.noise_std, state.shape)
        else:
            noise = np.random.uniform(-self.noise_std, self.noise_std, state.shape)
        return (state + noise).astype(np.float32)


class StateSelect(StateTransform):
    """Select specific indices from state."""

    def __init__(self, indices: list[int]):
        """
        Args:
            indices: Indices to select
        """
        self.indices = indices

    def transform(self, state: np.ndarray) -> np.ndarray:
        """Select state indices."""
        return state[..., self.indices]


class StateStack(StateTransform):
    """Stack multiple state components."""

    def __init__(self, keys: list[str]):
        """
        Args:
            keys: Keys to stack into state
        """
        self.keys = keys

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Stack state components."""
        components = []
        for key in self.keys:
            if key in sample:
                val = np.asarray(sample[key])
                if val.ndim == 0:
                    val = val[np.newaxis]
                components.append(val.flatten())

        if components:
            sample["proprioception"] = np.concatenate(components)

        return sample


class StateHistory(StateTransform):
    """Stack history of states."""

    def __init__(
        self,
        history_length: int = 4,
        keys: list[str] = ["proprioception"],
    ):
        """
        Args:
            history_length: Number of historical states
            keys: Keys to apply history to
        """
        self.history_length = history_length
        self.keys = keys
        self._history: dict[str, list[np.ndarray]] = {k: [] for k in keys}

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Add history to sample."""
        for key in self.keys:
            if key in sample:
                current = sample[key]
                self._history[key].append(current)

                # Keep only recent history
                if len(self._history[key]) > self.history_length:
                    self._history[key] = self._history[key][-self.history_length :]

                # Pad if not enough history
                while len(self._history[key]) < self.history_length:
                    self._history[key].insert(0, current)

                sample[f"{key}_history"] = np.stack(self._history[key])

        return sample

    def reset(self):
        """Reset history buffers."""
        self._history = {k: [] for k in self.keys}


__all__ = [
    "StateTransform",
    "StateNormalize",
    "StateNoise",
    "StateSelect",
    "StateStack",
    "StateHistory",
]
