"""Registry for robot interfaces."""

from librobot.utils.registry import GlobalRegistry

# Get robot registry
ROBOT_REGISTRY = GlobalRegistry.get_registry("robots")


def register_robot(name=None, aliases=None, **kwargs):
    """
    Decorator to register a robot interface.

    Args:
        name: Registration name
        aliases: List of aliases
        **kwargs: Additional metadata

    Returns:
        Decorator function
    """
    return ROBOT_REGISTRY.register(name=name, aliases=aliases, **kwargs)


def get_robot(name):
    """
    Get registered robot class.

    Args:
        name: Robot name or alias

    Returns:
        Robot class
    """
    return ROBOT_REGISTRY.get(name)


def create_robot(name, *args, **kwargs):
    """
    Create robot instance.

    Args:
        name: Robot name or alias
        *args: Positional arguments
        **kwargs: Keyword arguments

    Returns:
        Robot instance
    """
    return ROBOT_REGISTRY.create(name, *args, **kwargs)


def list_robots():
    """
    List all registered robots.

    Returns:
        List of robot names
    """
    return ROBOT_REGISTRY.list()


__all__ = [
    "ROBOT_REGISTRY",
    "register_robot",
    "get_robot",
    "create_robot",
    "list_robots",
]
