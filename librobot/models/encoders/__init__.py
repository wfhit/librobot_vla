"""Encoders module for LibroBot VLA.

Provides various encoder implementations including:
- State encoders (MLP, Transformer, Tokenizer)
- History encoders (MLP, LSTM, Transformer, TemporalConv)
- Fusion modules (Concat, CrossAttention, FiLM, Gated)
"""

from .base import AbstractEncoder

# Fusion modules
from .fusion import (
    ConcatFusion,
    CrossAttentionFusion,
    FiLMFusion,
    GatedFusion,
)

# History encoders
from .history import (
    LSTMHistoryEncoder,
    MLPHistoryEncoder,
    TemporalConvEncoder,
    TransformerHistoryEncoder,
)
from .registry import (
    ENCODER_REGISTRY,
    create_encoder,
    get_encoder,
    list_encoders,
    register_encoder,
)

# State encoders
from .state import (
    MLPStateEncoder,
    TokenizerStateEncoder,
    TransformerStateEncoder,
)

__all__ = [
    # Base and registry
    "AbstractEncoder",
    "ENCODER_REGISTRY",
    "register_encoder",
    "get_encoder",
    "create_encoder",
    "list_encoders",
    # State encoders
    "MLPStateEncoder",
    "TransformerStateEncoder",
    "TokenizerStateEncoder",
    # History encoders
    "MLPHistoryEncoder",
    "LSTMHistoryEncoder",
    "TransformerHistoryEncoder",
    "TemporalConvEncoder",
    # Fusion modules
    "ConcatFusion",
    "CrossAttentionFusion",
    "FiLMFusion",
    "GatedFusion",
]
