"""Inference server module."""

from .base_server import AbstractServer
from .grpc_server import GRPCServer, create_grpc_server


def __getattr__(name: str):
    """Lazy import for optional dependencies."""
    if name in ("RESTServer", "create_server"):
        from .rest_server import RESTServer, create_server

        if name == "RESTServer":
            return RESTServer
        return create_server
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AbstractServer",
    "RESTServer",
    "create_server",
    "GRPCServer",
    "create_grpc_server",
]
