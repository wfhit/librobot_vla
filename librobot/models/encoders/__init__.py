"""Encoders module for LibroBot VLA.

Provides various encoder implementations including:
- State encoders (MLP, Transformer, Tokenizer)
- History encoders (MLP, LSTM, Transformer, TemporalConv)
- Fusion modules (Concat, CrossAttention, FiLM, Gated)
"""

from .base import AbstractEncoder
from .registry import (
    ENCODER_REGISTRY,
    register_encoder,
    get_encoder,
    create_encoder,
    list_encoders,
)

# State encoders
from .state import (
    MLPStateEncoder,
    TransformerStateEncoder,
    TokenizerStateEncoder,
)

# History encoders
from .history import (
    MLPHistoryEncoder,
    LSTMHistoryEncoder,
    TransformerHistoryEncoder,
    TemporalConvEncoder,
)

# Fusion modules
from .fusion import (
    ConcatFusion,
    CrossAttentionFusion,
    FiLMFusion,
    GatedFusion,
)

__all__ = [
    # Base and registry
    'AbstractEncoder',
    'ENCODER_REGISTRY',
    'register_encoder',
    'get_encoder',
    'create_encoder',
    'list_encoders',
    # State encoders
    'MLPStateEncoder',
    'TransformerStateEncoder',
    'TokenizerStateEncoder',
    # History encoders
    'MLPHistoryEncoder',
    'LSTMHistoryEncoder',
    'TransformerHistoryEncoder',
    'TemporalConvEncoder',
    # Fusion modules
    'ConcatFusion',
    'CrossAttentionFusion',
    'FiLMFusion',
    'GatedFusion',
]
