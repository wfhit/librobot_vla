"""Loss functions for VLA training."""

from .base import AbstractLoss
from .mse import MSELoss, L1Loss, SmoothL1Loss, ActionLoss
from .cross_entropy import CrossEntropyLoss, FocalLoss, TokenLoss, BCELoss
from .diffusion import DiffusionLoss, DDPMLoss, ScoreMatchingLoss
from .flow_matching import FlowMatchingLoss, RectifiedFlowLoss, OTCFMLoss, ConsistencyLoss

__all__ = [
    # Base
    'AbstractLoss',
    # Regression
    'MSELoss',
    'L1Loss',
    'SmoothL1Loss',
    'ActionLoss',
    # Classification
    'CrossEntropyLoss',
    'FocalLoss',
    'TokenLoss',
    'BCELoss',
    # Diffusion
    'DiffusionLoss',
    'DDPMLoss',
    'ScoreMatchingLoss',
    # Flow matching
    'FlowMatchingLoss',
    'RectifiedFlowLoss',
    'OTCFMLoss',
    'ConsistencyLoss',
]
