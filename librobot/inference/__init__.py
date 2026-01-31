"""Inference package for LibroBot VLA.

Provides inference infrastructure including:
- Policies (Base, VLA, Ensemble, Diffusion, Autoregressive)
- Buffers (Action, TemporalEnsemble, Adaptive, Smoothing, History)
- KV Cache implementations (KVCache, MultiHead, Static)
- Servers (REST, gRPC, WebSocket, ROS2)
- Clients (REST, WebSocket, gRPC)
- Optimization (Quantization, ONNX export)
"""

# Import submodules
from . import buffers, clients, optimization, policy, servers
from .action_buffer import AdaptiveActionBuffer, TemporalEnsembleBuffer
from .buffers import ActionBuffer, ActionSmoothingBuffer, HistoryBuffer
from .clients import GRPCClient, RESTClient, WebSocketClient
from .kv_cache import KVCache, MultiHeadKVCache, StaticKVCache
from .optimization import ModelQuantizer, ONNXExporter, OptimizedModel

# Convenience imports
from .policy import AutoregressivePolicy, BasePolicy, DiffusionPolicy, EnsemblePolicy, VLAPolicy
from .quantization import (
    BaseQuantizer,
    BitsAndBytesQuantizer,
    DynamicQuantizer,
    GPTQQuantizer,
    StaticQuantizer,
    get_quantizer,
)
from .server import AbstractServer
from .servers import GRPCServer, RESTServer, ROS2Server, WebSocketServer

__all__ = [
    # Base
    "AbstractServer",
    # Policy
    "BasePolicy",
    "VLAPolicy",
    "EnsemblePolicy",
    "DiffusionPolicy",
    "AutoregressivePolicy",
    # KV Cache
    "KVCache",
    "MultiHeadKVCache",
    "StaticKVCache",
    # Buffers
    "ActionBuffer",
    "TemporalEnsembleBuffer",
    "AdaptiveActionBuffer",
    "ActionSmoothingBuffer",
    "HistoryBuffer",
    # Quantization
    "BaseQuantizer",
    "BitsAndBytesQuantizer",
    "GPTQQuantizer",
    "DynamicQuantizer",
    "StaticQuantizer",
    "get_quantizer",
    # Servers
    "RESTServer",
    "GRPCServer",
    "WebSocketServer",
    "ROS2Server",
    # Clients
    "RESTClient",
    "WebSocketClient",
    "GRPCClient",
    # Optimization
    "ModelQuantizer",
    "ONNXExporter",
    "OptimizedModel",
    # Submodules
    "policy",
    "buffers",
    "optimization",
    "servers",
    "clients",
]
