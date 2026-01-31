"""Inference server module."""

from .base_server import AbstractServer
from .grpc_server import GRPCServer, create_grpc_server


def __getattr__(name: str):
    """Lazy import for optional dependencies."""
    if name == "RESTServer":
        from .rest_server import RESTServer

        return RESTServer
    if name == "create_server":
        from .rest_server import create_server

        return create_server
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AbstractServer",
    "RESTServer",
    "create_server",
    "GRPCServer",
    "create_grpc_server",
]
