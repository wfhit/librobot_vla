"""Data transforms for robotics datasets.

This package provides transforms for preprocessing and augmenting robotics data
including images, actions, and states. Transforms follow PyTorch/torchvision
conventions and can be composed together.

See docs/design/data_pipeline.md for detailed design documentation.
"""

from .image_transforms import (
    ImageTransform,
    RandomCrop,
    CenterCrop,
    Resize,
    Normalize,
    RandomColorJitter,
    RandomGaussianBlur,
)
from .action_transforms import (
    ActionTransform,
    NormalizeAction,
    DenormalizeAction,
    ClipAction,
    AddActionNoise,
    DeltaToAbsolute,
    AbsoluteToDelta,
)
from .state_transforms import (
    StateTransform,
    NormalizeState,
    DenormalizeState,
    ClipState,
    AddStateNoise,
)

__all__ = [
    # Image transforms
    'ImageTransform',
    'RandomCrop',
    'CenterCrop',
    'Resize',
    'Normalize',
    'RandomColorJitter',
    'RandomGaussianBlur',
    # Action transforms
    'ActionTransform',
    'NormalizeAction',
    'DenormalizeAction',
    'ClipAction',
    'AddActionNoise',
    'DeltaToAbsolute',
    'AbsoluteToDelta',
    # State transforms
    'StateTransform',
    'NormalizeState',
    'DenormalizeState',
    'ClipState',
    'AddStateNoise',
]
