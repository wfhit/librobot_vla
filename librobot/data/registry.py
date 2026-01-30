"""Registry for datasets and tokenizers."""

from librobot.utils.registry import GlobalRegistry

# Get registries
DATASET_REGISTRY = GlobalRegistry.get_registry("datasets")
TOKENIZER_REGISTRY = GlobalRegistry.get_registry("tokenizers")


def register_dataset(name=None, aliases=None, **kwargs):
    """
    Decorator to register a dataset class.

    Args:
        name: Registration name
        aliases: List of aliases
        **kwargs: Additional metadata

    Returns:
        Decorator function

    Example:
        @register_dataset(name="bridge", aliases=["bridge_v2"])
        class BridgeDataset(AbstractDataset):
            ...
    """
    return DATASET_REGISTRY.register(name=name, aliases=aliases, **kwargs)


def get_dataset(name):
    """
    Get registered dataset class.

    Args:
        name: Dataset name or alias

    Returns:
        Dataset class
    """
    return DATASET_REGISTRY.get(name)


def create_dataset(name, *args, **kwargs):
    """
    Create dataset instance.

    Args:
        name: Dataset name or alias
        *args: Positional arguments
        **kwargs: Keyword arguments

    Returns:
        Dataset instance
    """
    return DATASET_REGISTRY.create(name, *args, **kwargs)


def list_datasets():
    """
    List all registered datasets.

    Returns:
        List of dataset names
    """
    return DATASET_REGISTRY.list()


def register_tokenizer(name=None, aliases=None, **kwargs):
    """
    Decorator to register a tokenizer class.

    Args:
        name: Registration name
        aliases: List of aliases
        **kwargs: Additional metadata

    Returns:
        Decorator function

    Example:
        @register_tokenizer(name="clip", aliases=["clip_tokenizer"])
        class CLIPTokenizer(AbstractTokenizer):
            ...
    """
    return TOKENIZER_REGISTRY.register(name=name, aliases=aliases, **kwargs)


def get_tokenizer(name):
    """
    Get registered tokenizer class.

    Args:
        name: Tokenizer name or alias

    Returns:
        Tokenizer class
    """
    return TOKENIZER_REGISTRY.get(name)


def create_tokenizer(name, *args, **kwargs):
    """
    Create tokenizer instance.

    Args:
        name: Tokenizer name or alias
        *args: Positional arguments
        **kwargs: Keyword arguments

    Returns:
        Tokenizer instance
    """
    return TOKENIZER_REGISTRY.create(name, *args, **kwargs)


def list_tokenizers():
    """
    List all registered tokenizers.

    Returns:
        List of tokenizer names
    """
    return TOKENIZER_REGISTRY.list()


__all__ = [
    # Dataset registry
    "DATASET_REGISTRY",
    "register_dataset",
    "get_dataset",
    "create_dataset",
    "list_datasets",
    # Tokenizer registry
    "TOKENIZER_REGISTRY",
    "register_tokenizer",
    "get_tokenizer",
    "create_tokenizer",
    "list_tokenizers",
]
