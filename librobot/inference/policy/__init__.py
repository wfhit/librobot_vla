"""Policy wrappers for inference."""

from .base import BasePolicy, DiffusionPolicy, AutoregressivePolicy, EnsemblePolicy

# VLAPolicy is an alias for BasePolicy to maintain API compatibility.
# BasePolicy provides the core functionality needed for VLA inference:
# - Model wrapping and device management
# - Observation preprocessing (images, text, state)
# - Action extraction from model outputs
# - Reset functionality for stateful policies
#
# For specialized VLA inference needs (KV caching, action buffering, etc.),
# see the VLAPolicy class in librobot/inference/policy.py or extend BasePolicy.
VLAPolicy = BasePolicy

__all__ = [
    'BasePolicy',
    'DiffusionPolicy',
    'AutoregressivePolicy',
    'EnsemblePolicy',
    'VLAPolicy',
]
