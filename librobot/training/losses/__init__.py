"""Loss functions for VLA training."""

from .base import AbstractLoss
from .cross_entropy import BCELoss, CrossEntropyLoss, FocalLoss, TokenLoss
from .diffusion import DDPMLoss, DiffusionLoss, ScoreMatchingLoss
from .flow_matching import ConsistencyLoss, FlowMatchingLoss, OTCFMLoss, RectifiedFlowLoss
from .mse import ActionLoss, L1Loss, MSELoss, SmoothL1Loss

__all__ = [
    # Base
    "AbstractLoss",
    # Regression
    "MSELoss",
    "L1Loss",
    "SmoothL1Loss",
    "ActionLoss",
    # Classification
    "CrossEntropyLoss",
    "FocalLoss",
    "TokenLoss",
    "BCELoss",
    # Diffusion
    "DiffusionLoss",
    "DDPMLoss",
    "ScoreMatchingLoss",
    # Flow matching
    "FlowMatchingLoss",
    "RectifiedFlowLoss",
    "OTCFMLoss",
    "ConsistencyLoss",
]
