"""Shared components for LibroBot VLA models.

Provides reusable building blocks including:
- Attention mechanisms (Standard, Flash, BlockWise, SlidingWindow)
- Positional encodings (Sinusoidal, Rotary, ALiBi)
- Normalization layers (LayerNorm, RMSNorm, GroupNorm)
- Activation functions (GELU, SwiGLU, GeGLU, Mish, QuickGELU)
"""

from . import attention
from . import positional
from . import normalization

# Activation functions
from .activations import GELU, SwiGLU, GeGLU, Mish, QuickGELU

# Attention mechanisms
from .attention import (
    StandardAttention,
    FlashAttention,
    BlockWiseAttention,
    SlidingWindowAttention,
)

# Positional encodings
from .positional import (
    SinusoidalPositionalEncoding,
    RotaryPositionEmbedding,
    ALiBiPositionalBias,
)

# Normalization layers
from .normalization import (
    LayerNorm,
    RMSNorm,
    GroupNorm,
)

__all__ = [
    # Submodules
    'attention',
    'positional',
    'normalization',
    # Activations
    'GELU',
    'SwiGLU',
    'GeGLU',
    'Mish',
    'QuickGELU',
    # Attention
    'StandardAttention',
    'FlashAttention',
    'BlockWiseAttention',
    'SlidingWindowAttention',
    # Positional
    'SinusoidalPositionalEncoding',
    'RotaryPositionEmbedding',
    'ALiBiPositionalBias',
    # Normalization
    'LayerNorm',
    'RMSNorm',
    'GroupNorm',
]
