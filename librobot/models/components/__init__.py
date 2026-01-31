"""Shared components for LibroBot VLA models.

Provides reusable building blocks including:
- Attention mechanisms (Standard, Flash, BlockWise, SlidingWindow)
- Positional encodings (Sinusoidal, Rotary, ALiBi)
- Normalization layers (LayerNorm, RMSNorm, GroupNorm)
- Activation functions (GELU, SwiGLU, GeGLU, Mish, QuickGELU)
"""

from . import attention, normalization, positional

# Activation functions
from .activations import GELU, GeGLU, Mish, QuickGELU, SwiGLU

# Attention mechanisms
from .attention import BlockWiseAttention, FlashAttention, SlidingWindowAttention, StandardAttention

# Normalization layers
from .normalization import GroupNorm, LayerNorm, RMSNorm

# Positional encodings
from .positional import ALiBiPositionalBias, RotaryPositionEmbedding, SinusoidalPositionalEncoding

__all__ = [
    # Submodules
    "attention",
    "positional",
    "normalization",
    # Activations
    "GELU",
    "SwiGLU",
    "GeGLU",
    "Mish",
    "QuickGELU",
    # Attention
    "StandardAttention",
    "FlashAttention",
    "BlockWiseAttention",
    "SlidingWindowAttention",
    # Positional
    "SinusoidalPositionalEncoding",
    "RotaryPositionEmbedding",
    "ALiBiPositionalBias",
    # Normalization
    "LayerNorm",
    "RMSNorm",
    "GroupNorm",
]
