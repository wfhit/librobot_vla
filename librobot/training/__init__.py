"""Training package for LibroBot VLA."""

from .losses import AbstractLoss
from .callbacks import AbstractCallback

__all__ = [
    'AbstractLoss',
    'AbstractCallback',
]
