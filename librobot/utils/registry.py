"""Central registry system for component registration and discovery."""

import inspect
import importlib
import pkgutil
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Type, TypeVar, Union
from functools import wraps
import threading


T = TypeVar('T')


class RegistryError(Exception):
    """Exception raised for registry-related errors."""
    pass


class Registry:
    """
    Thread-safe registry for managing component registration.

    Supports:
    - Decorator-based registration
    - Explicit registration
    - Alias support
    - Auto-discovery from modules
    - Thread-safe operations

    Examples:
        >>> registry = Registry("models")
        >>> 
        >>> # Register with decorator
        >>> @registry.register("my_model")
        >>> class MyModel:
        ...     pass
        >>> 
        >>> # Register explicitly
        >>> registry.register("another_model", AnotherModel)
        >>> 
        >>> # Register with aliases
        >>> registry.register("model", MyModel, aliases=["m", "default"])
        >>> 
        >>> # Get registered class
        >>> model_cls = registry.get("my_model")
        >>> 
        >>> # Create instance
        >>> model = registry.create("my_model", param1=value1)
    """

    def __init__(self, name: str):
        """
        Initialize registry.

        Args:
            name: Name of this registry
        """
        self.name = name
        self._registry: Dict[str, Type] = {}
        self._aliases: Dict[str, str] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()

    def register(
        self,
        name: Optional[str] = None,
        obj: Optional[Type] = None,
        aliases: Optional[List[str]] = None,
        force: bool = False,
        **metadata
    ) -> Callable:
        """
        Register a class or function.

        Can be used as decorator or called directly.

        Args:
            name: Registration name. If None, uses class/function name
            obj: Object to register. If None, returns decorator
            aliases: Alternative names for this object
            force: If True, overwrites existing registration
            **metadata: Additional metadata to store

        Returns:
            Decorator if obj is None, otherwise the registered object

        Examples:
            >>> # As decorator
            >>> @registry.register("my_class")
            >>> class MyClass:
            ...     pass
            >>> 
            >>> # Direct registration
            >>> registry.register("my_class", MyClass)
            >>> 
            >>> # With aliases
            >>> @registry.register("model", aliases=["m", "default"])
            >>> class Model:
            ...     pass
        """
        def decorator(obj_to_register: Type) -> Type:
            # Determine registration name
            reg_name = name
            if reg_name is None:
                if hasattr(obj_to_register, '__name__'):
                    reg_name = obj_to_register.__name__
                else:
                    raise RegistryError(f"Cannot infer name for {obj_to_register}")

            with self._lock:
                # Check if already registered
                if reg_name in self._registry and not force:
                    raise RegistryError(
                        f"'{reg_name}' is already registered in {self.name}. "
                        f"Use force=True to overwrite."
                    )

                # Register object
                self._registry[reg_name] = obj_to_register

                # Store metadata
                self._metadata[reg_name] = {
                    'name': reg_name,
                    'type': type(obj_to_register).__name__,
                    'module': getattr(obj_to_register, '__module__', None),
                    **metadata
                }

                # Register aliases
                if aliases:
                    for alias in aliases:
                        if alias in self._aliases and not force:
                            raise RegistryError(
                                f"Alias '{alias}' is already registered in {self.name}"
                            )
                        self._aliases[alias] = reg_name

            return obj_to_register

        # If obj is provided, register directly
        if obj is not None:
            return decorator(obj)

        # Otherwise return decorator
        return decorator

    def unregister(self, name: str) -> None:
        """
        Unregister a component.

        Args:
            name: Name to unregister
        """
        with self._lock:
            # Resolve alias if needed
            actual_name = self._aliases.get(name, name)

            if actual_name not in self._registry:
                raise RegistryError(f"'{name}' is not registered in {self.name}")

            # Remove from registry
            del self._registry[actual_name]
            del self._metadata[actual_name]

            # Remove aliases
            aliases_to_remove = [k for k, v in self._aliases.items() if v == actual_name]
            for alias in aliases_to_remove:
                del self._aliases[alias]

    def get(self, name: str, default: Optional[Type] = None) -> Optional[Type]:
        """
        Get registered object by name or alias.

        Args:
            name: Registration name or alias
            default: Default value if not found

        Returns:
            Registered object or default
        """
        with self._lock:
            # Resolve alias
            actual_name = self._aliases.get(name, name)
            return self._registry.get(actual_name, default)

    def get_metadata(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for registered object.

        Args:
            name: Registration name or alias

        Returns:
            Metadata dictionary or None
        """
        with self._lock:
            actual_name = self._aliases.get(name, name)
            return self._metadata.get(actual_name)

    def create(self, name: str, *args, **kwargs) -> Any:
        """
        Create instance of registered class.

        Args:
            name: Registration name or alias
            *args: Positional arguments for constructor
            **kwargs: Keyword arguments for constructor

        Returns:
            Instance of registered class
        """
        cls = self.get(name)
        if cls is None:
            raise RegistryError(f"'{name}' is not registered in {self.name}")

        return cls(*args, **kwargs)

    def contains(self, name: str) -> bool:
        """
        Check if name is registered.

        Args:
            name: Name to check

        Returns:
            bool: True if registered
        """
        with self._lock:
            actual_name = self._aliases.get(name, name)
            return actual_name in self._registry

    def list(self) -> List[str]:
        """
        List all registered names.

        Returns:
            List of registered names
        """
        with self._lock:
            return sorted(self._registry.keys())

    def list_with_aliases(self) -> Dict[str, List[str]]:
        """
        List all registered names with their aliases.

        Returns:
            Dictionary mapping names to their aliases
        """
        with self._lock:
            result = {name: [] for name in self._registry.keys()}

            for alias, name in self._aliases.items():
                if name in result:
                    result[name].append(alias)

            return result

    def clear(self) -> None:
        """Clear all registrations."""
        with self._lock:
            self._registry.clear()
            self._aliases.clear()
            self._metadata.clear()

    def auto_discover(
        self,
        package_name: str,
        base_class: Optional[Type] = None,
        exclude: Optional[Set[str]] = None
    ) -> int:
        """
        Auto-discover and register classes from a package.

        Args:
            package_name: Package to scan for classes
            base_class: Only register subclasses of this class
            exclude: Set of class names to exclude

        Returns:
            int: Number of classes registered
        """
        if exclude is None:
            exclude = set()

        count = 0

        try:
            package = importlib.import_module(package_name)
        except ImportError:
            return count

        # Get package path
        if hasattr(package, '__path__'):
            package_path = package.__path__
        else:
            return count

        # Walk through package
        for importer, modname, ispkg in pkgutil.walk_packages(
            path=package_path,
            prefix=package_name + '.'
        ):
            try:
                module = importlib.import_module(modname)

                # Inspect module for classes
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    # Skip if in exclude list
                    if name in exclude:
                        continue

                    # Skip if not in this module (imported from elsewhere)
                    if obj.__module__ != modname:
                        continue

                    # Check base class requirement
                    if base_class is not None:
                        if not issubclass(obj, base_class) or obj is base_class:
                            continue

                    # Register
                    try:
                        self.register(name, obj)
                        count += 1
                    except RegistryError:
                        # Already registered, skip
                        pass

            except ImportError:
                continue

        return count

    def __contains__(self, name: str) -> bool:
        """Check if name is registered."""
        return self.contains(name)

    def __getitem__(self, name: str) -> Type:
        """Get registered object."""
        obj = self.get(name)
        if obj is None:
            raise KeyError(f"'{name}' is not registered in {self.name}")
        return obj

    def __len__(self) -> int:
        """Get number of registered objects."""
        with self._lock:
            return len(self._registry)

    def __repr__(self) -> str:
        """String representation."""
        with self._lock:
            names = ', '.join(self._registry.keys())
            return f"Registry('{self.name}', {len(self._registry)} items: [{names}])"


class GlobalRegistry:
    """
    Global registry manager for all component registries.

    Examples:
        >>> # Get or create a registry
        >>> model_registry = GlobalRegistry.get_registry("models")
        >>> 
        >>> # Register in registry
        >>> @model_registry.register("my_model")
        >>> class MyModel:
        ...     pass
    """

    _registries: Dict[str, Registry] = {}
    _lock = threading.RLock()

    @classmethod
    def get_registry(cls, name: str) -> Registry:
        """
        Get or create a registry by name.

        Args:
            name: Registry name

        Returns:
            Registry: Registry instance
        """
        with cls._lock:
            if name not in cls._registries:
                cls._registries[name] = Registry(name)
            return cls._registries[name]

    @classmethod
    def list_registries(cls) -> List[str]:
        """
        List all registry names.

        Returns:
            List of registry names
        """
        with cls._lock:
            return sorted(cls._registries.keys())

    @classmethod
    def clear_all(cls) -> None:
        """Clear all registries."""
        with cls._lock:
            cls._registries.clear()


def build_from_config(config: Dict[str, Any], registry: Registry, **kwargs) -> Any:
    """
    Build object from configuration.

    Args:
        config: Configuration dictionary with 'type' key
        registry: Registry to look up the type
        **kwargs: Additional arguments to pass to constructor

    Returns:
        Created object instance

    Examples:
        >>> config = {"type": "MyModel", "hidden_size": 768}
        >>> model = build_from_config(config, model_registry)
    """
    if not isinstance(config, dict):
        raise TypeError(f"Config must be dict, got {type(config)}")

    if 'type' not in config:
        raise KeyError("Config must have 'type' key")

    config = config.copy()
    obj_type = config.pop('type')

    # Merge kwargs with config
    config.update(kwargs)

    return registry.create(obj_type, **config)
