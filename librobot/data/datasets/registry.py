"""Registry for dataset implementations."""

from librobot.utils.registry import GlobalRegistry

# Get dataset registry
DATASET_REGISTRY = GlobalRegistry.get_registry("dataset")


def register_dataset(name=None, aliases=None, **kwargs):
    """
    Decorator to register a dataset.
    
    Args:
        name: Registration name
        aliases: List of aliases
        **kwargs: Additional metadata
        
    Returns:
        Decorator function
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


__all__ = [
    'DATASET_REGISTRY',
    'register_dataset',
    'get_dataset',
    'create_dataset',
    'list_datasets',
]
