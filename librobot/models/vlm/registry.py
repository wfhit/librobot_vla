"""Registry for Vision-Language Models."""

from librobot.utils.registry import GlobalRegistry

# Get VLM registry
VLM_REGISTRY = GlobalRegistry.get_registry("vlm")


def register_vlm(name=None, aliases=None, **kwargs):
    """
    Decorator to register a VLM.

    Args:
        name: Registration name
        aliases: List of aliases
        **kwargs: Additional metadata

    Returns:
        Decorator function
    """
    return VLM_REGISTRY.register(name=name, aliases=aliases, **kwargs)


def get_vlm(name):
    """
    Get registered VLM class.

    Args:
        name: VLM name or alias

    Returns:
        VLM class
    """
    return VLM_REGISTRY.get(name)


def create_vlm(name, *args, **kwargs):
    """
    Create VLM instance.

    Args:
        name: VLM name or alias
        *args: Positional arguments
        **kwargs: Keyword arguments

    Returns:
        VLM instance
    """
    return VLM_REGISTRY.create(name, *args, **kwargs)


def list_vlms():
    """
    List all registered VLMs.

    Returns:
        List of VLM names
    """
    return VLM_REGISTRY.list()


__all__ = [
    'VLM_REGISTRY',
    'register_vlm',
    'get_vlm',
    'create_vlm',
    'list_vlms',
]
