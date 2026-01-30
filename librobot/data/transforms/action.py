"""Action transforms for data augmentation and normalization."""

from typing import Any, Optional, Union

import numpy as np


class ActionTransform:
    """Base class for action transforms."""

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Apply transform to sample."""
        if "actions" in sample:
            sample["actions"] = self.transform(sample["actions"])
        return sample

    def transform(self, action: np.ndarray) -> np.ndarray:
        """Transform action. Override in subclasses."""
        return action


class ActionNormalize(ActionTransform):
    """Normalize actions to standard range."""

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

    def fit(self, actions: np.ndarray) -> "ActionNormalize":
        """Fit normalization statistics from data."""
        self.mean = np.mean(actions, axis=0)
        self.std = np.std(actions, axis=0) + 1e-6
        self.min_val = np.min(actions, axis=0)
        self.max_val = np.max(actions, axis=0)
        return self

    def transform(self, action: np.ndarray) -> np.ndarray:
        """Normalize action."""
        if self.normalize_type == "standard":
            if self.mean is not None and self.std is not None:
                return (action - self.mean) / self.std
        elif self.normalize_type == "minmax":
            if self.min_val is not None and self.max_val is not None:
                return (action - self.min_val) / (self.max_val - self.min_val + 1e-8)
        return action

    def inverse_transform(self, action: np.ndarray) -> np.ndarray:
        """Denormalize action."""
        if self.normalize_type == "standard":
            if self.mean is not None and self.std is not None:
                return action * self.std + self.mean
        elif self.normalize_type == "minmax":
            if self.min_val is not None and self.max_val is not None:
                return action * (self.max_val - self.min_val) + self.min_val
        return action


class ActionNoise(ActionTransform):
    """Add noise to actions for data augmentation."""

    def __init__(
        self,
        noise_std: float = 0.01,
        noise_type: str = "gaussian",
        clip: Optional[float] = None,
    ):
        """
        Args:
            noise_std: Standard deviation of noise
            noise_type: Type of noise ("gaussian", "uniform")
            clip: Optional clipping range after adding noise
        """
        self.noise_std = noise_std
        self.noise_type = noise_type
        self.clip = clip

    def transform(self, action: np.ndarray) -> np.ndarray:
        """Add noise to action."""
        if self.noise_type == "gaussian":
            noise = np.random.normal(0, self.noise_std, action.shape)
        elif self.noise_type == "uniform":
            noise = np.random.uniform(-self.noise_std, self.noise_std, action.shape)
        else:
            noise = 0

        action = action + noise

        if self.clip is not None:
            action = np.clip(action, -self.clip, self.clip)

        return action.astype(np.float32)


class ActionScale(ActionTransform):
    """Scale actions by a factor."""

    def __init__(self, scale: Union[float, np.ndarray] = 1.0):
        """
        Args:
            scale: Scaling factor (scalar or per-dimension)
        """
        self.scale = np.asarray(scale)

    def transform(self, action: np.ndarray) -> np.ndarray:
        return action * self.scale


class ActionClip(ActionTransform):
    """Clip actions to valid range."""

    def __init__(
        self,
        min_val: Union[float, np.ndarray] = -1.0,
        max_val: Union[float, np.ndarray] = 1.0,
    ):
        """
        Args:
            min_val: Minimum value(s)
            max_val: Maximum value(s)
        """
        self.min_val = np.asarray(min_val)
        self.max_val = np.asarray(max_val)

    def transform(self, action: np.ndarray) -> np.ndarray:
        return np.clip(action, self.min_val, self.max_val)


class ActionDelta(ActionTransform):
    """Convert absolute actions to delta actions."""

    def __init__(self, reference_key: str = "proprioception"):
        """
        Args:
            reference_key: Key for reference state in sample
        """
        self.reference_key = reference_key
        self._reference = None

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Apply delta transform using reference state."""
        if self.reference_key in sample and "actions" in sample:
            reference = sample[self.reference_key]
            action = sample["actions"]

            # Compute delta (assuming action and state have same dimensions for position)
            action_dim = min(action.shape[-1], reference.shape[-1])
            delta_action = action.copy()
            delta_action[..., :action_dim] = action[..., :action_dim] - reference[..., :action_dim]
            sample["actions"] = delta_action

        return sample


class RelativeAction(ActionTransform):
    """Convert actions to be relative to current state."""

    def __init__(
        self,
        position_indices: Optional[list] = None,
        rotation_indices: Optional[list] = None,
    ):
        """
        Args:
            position_indices: Indices of position components
            rotation_indices: Indices of rotation components
        """
        self.position_indices = position_indices or [0, 1, 2]
        self.rotation_indices = rotation_indices or [3, 4, 5]

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Convert to relative actions."""
        if "actions" in sample and "proprioception" in sample:
            action = sample["actions"]
            state = sample["proprioception"]

            # Make position relative
            for i in self.position_indices:
                if i < action.shape[-1] and i < state.shape[-1]:
                    action[..., i] = action[..., i] - state[..., i]

            sample["actions"] = action

        return sample


__all__ = [
    "ActionTransform",
    "ActionNormalize",
    "ActionNoise",
    "ActionScale",
    "ActionClip",
    "ActionDelta",
    "RelativeAction",
]
