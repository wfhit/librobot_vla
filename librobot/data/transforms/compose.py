"""Compose transforms together."""

from typing import Any, Callable, Optional

import numpy as np


class Compose:
    """Compose multiple transforms together."""

    def __init__(self, transforms: list[Callable]):
        """
        Args:
            transforms: List of transform functions
        """
        self.transforms = transforms

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Apply all transforms in sequence."""
        for t in self.transforms:
            sample = t(sample)
        return sample

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += f"\n    {t}"
        format_string += "\n)"
        return format_string


class RandomApply:
    """Randomly apply a transform with given probability."""

    def __init__(
        self,
        transform: Callable,
        p: float = 0.5,
    ):
        """
        Args:
            transform: Transform to apply
            p: Probability of applying
        """
        self.transform = transform
        self.p = p

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Apply transform with probability p."""
        if np.random.random() < self.p:
            return self.transform(sample)
        return sample


class RandomChoice:
    """Randomly choose one transform from a list."""

    def __init__(self, transforms: list[Callable]):
        """
        Args:
            transforms: List of transforms to choose from
        """
        self.transforms = transforms

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Apply randomly chosen transform."""
        t = np.random.choice(self.transforms)
        return t(sample)


class Identity:
    """Identity transform (no-op)."""

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Return sample unchanged."""
        return sample


class Lambda:
    """Apply a custom lambda function."""

    def __init__(self, func: Callable):
        """
        Args:
            func: Function to apply
        """
        self.func = func

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Apply function."""
        return self.func(sample)


class KeyRename:
    """Rename keys in sample."""

    def __init__(self, mapping: dict[str, str]):
        """
        Args:
            mapping: Old key -> New key mapping
        """
        self.mapping = mapping

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Rename keys."""
        for old_key, new_key in self.mapping.items():
            if old_key in sample:
                sample[new_key] = sample.pop(old_key)
        return sample


class KeySelect:
    """Select only specified keys from sample."""

    def __init__(self, keys: list[str]):
        """
        Args:
            keys: Keys to keep
        """
        self.keys = keys

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Select keys."""
        return {k: sample[k] for k in self.keys if k in sample}


class ToTensor:
    """Convert numpy arrays to tensors."""

    def __init__(
        self,
        keys: Optional[list[str]] = None,
        device: str = "cpu",
    ):
        """
        Args:
            keys: Keys to convert (None = all)
            device: Target device
        """
        self.keys = keys
        self.device = device

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Convert to tensors."""
        try:
            import torch

            keys = self.keys or list(sample.keys())

            for key in keys:
                if key in sample and isinstance(sample[key], np.ndarray):
                    sample[key] = torch.from_numpy(sample[key]).to(self.device)

            return sample
        except ImportError:
            return sample


__all__ = [
    "Compose",
    "RandomApply",
    "RandomChoice",
    "Identity",
    "Lambda",
    "KeyRename",
    "KeySelect",
    "ToTensor",
]
