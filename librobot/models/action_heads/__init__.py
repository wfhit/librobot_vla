"""Action heads module for LibroBot VLA.

Provides various action prediction mechanisms including:
- MLP-based heads (MLPActionHead)
- Autoregressive/FAST heads (FASTActionHead)
- Transformer ACT heads (ACTActionHead)
- Hybrid heads (HybridActionHead)
- Diffusion-based heads (DDPM, DDIM, DPM, Consistency)
- Flow matching heads (FlowMatching, OTCFM, RectifiedFlow)
"""

from .base import AbstractActionHead
from .registry import (
    ACTION_HEAD_REGISTRY,
    register_action_head,
    get_action_head,
    create_action_head,
    list_action_heads,
)

# Core action heads
from .mlp_oft import MLPActionHead
from .autoregressive_fast import FASTActionHead
from .transformer_act import ACTActionHead
from .hybrid import HybridActionHead

# Diffusion-based action heads
from .diffusion import (
    DDPMActionHead,
    DDIMActionHead,
    DPMActionHead,
    UNet1D,
    DiT,
    ConsistencyActionHead,
)

# Flow matching action heads
from .flow_matching import (
    FlowMatchingHead,
    OTCFMHead,
    RectifiedFlowHead,
)

__all__ = [
    # Base and registry
    'AbstractActionHead',
    'ACTION_HEAD_REGISTRY',
    'register_action_head',
    'get_action_head',
    'create_action_head',
    'list_action_heads',
    # Core heads
    'MLPActionHead',
    'FASTActionHead',
    'ACTActionHead',
    'HybridActionHead',
    # Diffusion heads
    'DDPMActionHead',
    'DDIMActionHead',
    'DPMActionHead',
    'UNet1D',
    'DiT',
    'ConsistencyActionHead',
    # Flow matching heads
    'FlowMatchingHead',
    'OTCFMHead',
    'RectifiedFlowHead',
]
