"""Base class and registry for teleoperation interfaces."""

from librobot.collection.base import AbstractTeleop
from librobot.utils.registry import GlobalRegistry

# Get teleoperation registry
TELEOP_REGISTRY = GlobalRegistry.get_registry("teleoperation")


def register_teleop(name=None, aliases=None, **kwargs):
    """
    Decorator to register a teleoperation interface.

    Args:
        name: Registration name
        aliases: List of aliases
        **kwargs: Additional metadata

    Returns:
        Decorator function
    """
    return TELEOP_REGISTRY.register(name=name, aliases=aliases, **kwargs)


def get_teleop(name):
    """
    Get registered teleoperation class.

    Args:
        name: Teleoperation name or alias

    Returns:
        Teleoperation class
    """
    return TELEOP_REGISTRY.get(name)


def create_teleop(name, *args, **kwargs):
    """
    Create teleoperation instance.

    Args:
        name: Teleoperation name or alias
        *args: Positional arguments
        **kwargs: Keyword arguments

    Returns:
        Teleoperation instance
    """
    return TELEOP_REGISTRY.create(name, *args, **kwargs)


def list_teleoperation():
    """
    List all registered teleoperation interfaces.

    Returns:
        List of teleoperation names
    """
    return TELEOP_REGISTRY.list()


__all__ = [
    "AbstractTeleop",
    "TELEOP_REGISTRY",
    "register_teleop",
    "get_teleop",
    "create_teleop",
    "list_teleoperation",
]
