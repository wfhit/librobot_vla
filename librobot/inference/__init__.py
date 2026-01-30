"""Inference package for LibroBot VLA.

Provides inference infrastructure including:
- Policies (Base, VLA, Ensemble, Diffusion, Autoregressive)
- Buffers (Action, TemporalEnsemble, Adaptive, Smoothing, History)
- KV Cache implementations (KVCache, MultiHead, Static)
- Servers (REST, gRPC, WebSocket, ROS2)
- Clients (REST, WebSocket, gRPC)
- Optimization (Quantization, ONNX export)
"""

from .server import AbstractServer
from .policy import BasePolicy, VLAPolicy, EnsemblePolicy
from .kv_cache import KVCache, MultiHeadKVCache, StaticKVCache
from .action_buffer import ActionBuffer, TemporalEnsembleBuffer, AdaptiveActionBuffer
from .quantization import (
    BaseQuantizer,
    BitsAndBytesQuantizer,
    GPTQQuantizer,
    DynamicQuantizer,
    StaticQuantizer,
    get_quantizer,
)

# Import submodules
from . import policy
from . import buffers
from . import optimization
from . import servers
from . import clients

# Convenience imports
from .policy import BasePolicy, DiffusionPolicy, AutoregressivePolicy
from .buffers import ActionBuffer, ActionSmoothingBuffer, HistoryBuffer
from .servers import RESTServer, GRPCServer, WebSocketServer, ROS2Server
from .clients import RESTClient, WebSocketClient, GRPCClient
from .optimization import ModelQuantizer, ONNXExporter, OptimizedModel

__all__ = [
    # Base
    'AbstractServer',
    # Policy
    'BasePolicy',
    'VLAPolicy',
    'EnsemblePolicy',
    'DiffusionPolicy',
    'AutoregressivePolicy',
    # KV Cache
    'KVCache',
    'MultiHeadKVCache',
    'StaticKVCache',
    # Buffers
    'ActionBuffer',
    'TemporalEnsembleBuffer',
    'AdaptiveActionBuffer',
    'ActionSmoothingBuffer',
    'HistoryBuffer',
    # Quantization
    'BaseQuantizer',
    'BitsAndBytesQuantizer',
    'GPTQQuantizer',
    'DynamicQuantizer',
    'StaticQuantizer',
    'get_quantizer',
    # Servers
    'RESTServer',
    'GRPCServer',
    'WebSocketServer',
    'ROS2Server',
    # Clients
    'RESTClient',
    'WebSocketClient',
    'GRPCClient',
    # Optimization
    'ModelQuantizer',
    'ONNXExporter',
    'OptimizedModel',
    # Submodules
    'policy',
    'buffers',
    'optimization',
    'servers',
    'clients',
]
