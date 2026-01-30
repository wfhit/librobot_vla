"""Registry for encoders."""

from librobot.utils.registry import GlobalRegistry

# Get encoder registry
ENCODER_REGISTRY = GlobalRegistry.get_registry("encoders")


def register_encoder(name=None, aliases=None, **kwargs):
    """
    Decorator to register an encoder.

    Args:
        name: Registration name
        aliases: List of aliases
        **kwargs: Additional metadata

    Returns:
        Decorator function
    """
    return ENCODER_REGISTRY.register(name=name, aliases=aliases, **kwargs)


def get_encoder(name):
    """
    Get registered encoder class.

    Args:
        name: Encoder name or alias

    Returns:
        Encoder class
    """
    return ENCODER_REGISTRY.get(name)


def create_encoder(name, *args, **kwargs):
    """
    Create encoder instance.

    Args:
        name: Encoder name or alias
        *args: Positional arguments
        **kwargs: Keyword arguments

    Returns:
        Encoder instance
    """
    return ENCODER_REGISTRY.create(name, *args, **kwargs)


def list_encoders():
    """
    List all registered encoders.

    Returns:
        List of encoder names
    """
    return ENCODER_REGISTRY.list()


__all__ = [
    "ENCODER_REGISTRY",
    "register_encoder",
    "get_encoder",
    "create_encoder",
    "list_encoders",
]
