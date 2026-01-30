"""Registry for action heads."""

from librobot.utils.registry import GlobalRegistry

# Get action head registry
ACTION_HEAD_REGISTRY = GlobalRegistry.get_registry("action_heads")


def register_action_head(name=None, aliases=None, **kwargs):
    """
    Decorator to register an action head.

    Args:
        name: Registration name
        aliases: List of aliases
        **kwargs: Additional metadata

    Returns:
        Decorator function
    """
    return ACTION_HEAD_REGISTRY.register(name=name, aliases=aliases, **kwargs)


def get_action_head(name):
    """
    Get registered action head class.

    Args:
        name: Action head name or alias

    Returns:
        Action head class
    """
    return ACTION_HEAD_REGISTRY.get(name)


def create_action_head(name, *args, **kwargs):
    """
    Create action head instance.

    Args:
        name: Action head name or alias
        *args: Positional arguments
        **kwargs: Keyword arguments

    Returns:
        Action head instance
    """
    return ACTION_HEAD_REGISTRY.create(name, *args, **kwargs)


def list_action_heads():
    """
    List all registered action heads.

    Returns:
        List of action head names
    """
    return ACTION_HEAD_REGISTRY.list()


__all__ = [
    "ACTION_HEAD_REGISTRY",
    "register_action_head",
    "get_action_head",
    "create_action_head",
    "list_action_heads",
]
