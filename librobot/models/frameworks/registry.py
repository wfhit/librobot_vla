"""Registry for VLA frameworks."""

from librobot.utils.registry import GlobalRegistry

# Get VLA framework registry
VLA_REGISTRY = GlobalRegistry.get_registry("vla_frameworks")


def register_vla(name=None, aliases=None, **kwargs):
    """
    Decorator to register a VLA framework.

    Args:
        name: Registration name
        aliases: List of aliases
        **kwargs: Additional metadata

    Returns:
        Decorator function
    """
    return VLA_REGISTRY.register(name=name, aliases=aliases, **kwargs)


def get_vla(name):
    """
    Get registered VLA framework class.

    Args:
        name: VLA framework name or alias

    Returns:
        VLA framework class
    """
    return VLA_REGISTRY.get(name)


def create_vla(name, *args, **kwargs):
    """
    Create VLA framework instance.

    Args:
        name: VLA framework name or alias
        *args: Positional arguments
        **kwargs: Keyword arguments

    Returns:
        VLA framework instance
    """
    return VLA_REGISTRY.create(name, *args, **kwargs)


def list_vlas():
    """
    List all registered VLA frameworks.

    Returns:
        List of VLA framework names
    """
    return VLA_REGISTRY.list()


__all__ = [
    "VLA_REGISTRY",
    "register_vla",
    "get_vla",
    "create_vla",
    "list_vlas",
]
