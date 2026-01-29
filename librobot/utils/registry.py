"""Component registry system for LibroBot.

The registry pattern allows for dynamic discovery and registration of components
like VLMs, action heads, encoders, frameworks, datasets, etc.
"""

from typing import Any, Callable, Dict, List, Optional, Type


class Registry:
    """Component registry for dynamic discovery and instantiation."""

    def __init__(self):
        self._registry: Dict[str, Dict[str, Any]] = {
            "vlm": {},
            "action_head": {},
            "encoder": {},
            "framework": {},
            "dataset": {},
            "robot": {},
            "optimizer": {},
            "scheduler": {},
            "loss": {},
            "transform": {},
        }
        self._aliases: Dict[str, Dict[str, str]] = {key: {} for key in self._registry.keys()}

    def register(
        self,
        category: str,
        name: str,
        cls: Type,
        aliases: Optional[List[str]] = None,
    ) -> Type:
        """Register a component.

        Args:
            category: Category of the component (e.g., 'vlm', 'action_head')
            name: Name of the component
            cls: Class to register
            aliases: Optional list of aliases for the component

        Returns:
            The registered class (for use as decorator)
        """
        if category not in self._registry:
            raise ValueError(f"Unknown category: {category}")

        if name in self._registry[category]:
            raise ValueError(f"Component {name} already registered in {category}")

        self._registry[category][name] = cls

        # Register aliases
        if aliases:
            for alias in aliases:
                self._aliases[category][alias] = name

        return cls

    def get(self, category: str, name: str) -> Type:
        """Get a registered component by name or alias.

        Args:
            category: Category of the component
            name: Name or alias of the component

        Returns:
            The registered class
        """
        if category not in self._registry:
            raise ValueError(f"Unknown category: {category}")

        # Check if it's an alias
        if name in self._aliases[category]:
            name = self._aliases[category][name]

        if name not in self._registry[category]:
            raise ValueError(
                f"Component {name} not found in {category}. "
                f"Available: {list(self._registry[category].keys())}"
            )

        return self._registry[category][name]

    def list(self, category: str) -> List[str]:
        """List all registered components in a category.

        Args:
            category: Category to list

        Returns:
            List of registered component names
        """
        if category not in self._registry:
            raise ValueError(f"Unknown category: {category}")

        return list(self._registry[category].keys())

    def is_registered(self, category: str, name: str) -> bool:
        """Check if a component is registered.

        Args:
            category: Category of the component
            name: Name or alias of the component

        Returns:
            True if registered, False otherwise
        """
        if category not in self._registry:
            return False

        # Check both direct name and aliases
        if name in self._aliases[category]:
            name = self._aliases[category][name]

        return name in self._registry[category]


# Global registry instance
REGISTRY = Registry()


# Decorator functions for easy registration
def register_vlm(name: str, aliases: Optional[List[str]] = None) -> Callable:
    """Decorator to register a VLM implementation."""
    def decorator(cls: Type) -> Type:
        return REGISTRY.register("vlm", name, cls, aliases)
    return decorator


def register_action_head(name: str, aliases: Optional[List[str]] = None) -> Callable:
    """Decorator to register an action head implementation."""
    def decorator(cls: Type) -> Type:
        return REGISTRY.register("action_head", name, cls, aliases)
    return decorator


def register_encoder(name: str, aliases: Optional[List[str]] = None) -> Callable:
    """Decorator to register an encoder implementation."""
    def decorator(cls: Type) -> Type:
        return REGISTRY.register("encoder", name, cls, aliases)
    return decorator


def register_framework(name: str, aliases: Optional[List[str]] = None) -> Callable:
    """Decorator to register a VLA framework implementation."""
    def decorator(cls: Type) -> Type:
        return REGISTRY.register("framework", name, cls, aliases)
    return decorator


def register_dataset(name: str, aliases: Optional[List[str]] = None) -> Callable:
    """Decorator to register a dataset implementation."""
    def decorator(cls: Type) -> Type:
        return REGISTRY.register("dataset", name, cls, aliases)
    return decorator


def register_robot(name: str, aliases: Optional[List[str]] = None) -> Callable:
    """Decorator to register a robot implementation."""
    def decorator(cls: Type) -> Type:
        return REGISTRY.register("robot", name, cls, aliases)
    return decorator


def register_optimizer(name: str, aliases: Optional[List[str]] = None) -> Callable:
    """Decorator to register an optimizer builder."""
    def decorator(fn: Callable) -> Callable:
        return REGISTRY.register("optimizer", name, fn, aliases)
    return decorator


def register_scheduler(name: str, aliases: Optional[List[str]] = None) -> Callable:
    """Decorator to register a scheduler builder."""
    def decorator(fn: Callable) -> Callable:
        return REGISTRY.register("scheduler", name, fn, aliases)
    return decorator


def register_loss(name: str, aliases: Optional[List[str]] = None) -> Callable:
    """Decorator to register a loss function."""
    def decorator(fn: Callable) -> Callable:
        return REGISTRY.register("loss", name, fn, aliases)
    return decorator


def register_transform(name: str, aliases: Optional[List[str]] = None) -> Callable:
    """Decorator to register a data transform."""
    def decorator(cls: Type) -> Type:
        return REGISTRY.register("transform", name, cls, aliases)
    return decorator
