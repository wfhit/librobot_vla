"""Base class and registry for data format converters."""

from librobot.collection.base import AbstractConverter
from librobot.utils.registry import GlobalRegistry

# Get converter registry
CONVERTER_REGISTRY = GlobalRegistry.get_registry("converters")


def register_converter(name=None, aliases=None, **kwargs):
    """
    Decorator to register a data format converter.

    Args:
        name: Registration name
        aliases: List of aliases
        **kwargs: Additional metadata

    Returns:
        Decorator function
    """
    return CONVERTER_REGISTRY.register(name=name, aliases=aliases, **kwargs)


def get_converter(name):
    """
    Get registered converter class.

    Args:
        name: Converter name or alias

    Returns:
        Converter class
    """
    return CONVERTER_REGISTRY.get(name)


def create_converter(name, *args, **kwargs):
    """
    Create converter instance.

    Args:
        name: Converter name or alias
        *args: Positional arguments
        **kwargs: Keyword arguments

    Returns:
        Converter instance
    """
    return CONVERTER_REGISTRY.create(name, *args, **kwargs)


def list_converters():
    """
    List all registered converters.

    Returns:
        List of converter names
    """
    return CONVERTER_REGISTRY.list()


__all__ = [
    "AbstractConverter",
    "CONVERTER_REGISTRY",
    "register_converter",
    "get_converter",
    "create_converter",
    "list_converters",
]
