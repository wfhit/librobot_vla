"""
Unit tests for the registry system.

Tests the registration and retrieval of models, encoders, action heads,
and other components through the registry system.
"""

import threading
import pytest

from librobot.utils.registry import (
    Registry,
    GlobalRegistry,
    RegistryError,
    build_from_config,
)


@pytest.fixture
def registry():
    """Create a fresh registry for testing."""
    return Registry("test")


@pytest.fixture
def sample_class():
    """Create a sample class for testing registration."""

    class SampleClass:
        def __init__(self, param1=None, param2=None):
            self.param1 = param1
            self.param2 = param2

    return SampleClass


@pytest.fixture
def another_class():
    """Create another sample class for testing."""

    class AnotherClass:
        def __init__(self, value=0):
            self.value = value

    return AnotherClass


class TestRegistry:
    """Test suite for the Registry class."""

    def test_registry_initialization(self, registry):
        """Test that registry initializes correctly."""
        assert registry.name == "test"
        assert len(registry) == 0
        assert registry.list() == []

    def test_register_with_explicit_name(self, registry, sample_class):
        """Test registering a class with explicit name."""
        registry.register("my_class", sample_class)
        assert "my_class" in registry
        assert registry.get("my_class") == sample_class

    def test_register_with_inferred_name(self, registry, sample_class):
        """Test registering a class with inferred name."""
        registry.register(obj=sample_class)
        assert "SampleClass" in registry
        assert registry.get("SampleClass") == sample_class

    def test_register_as_decorator(self, registry):
        """Test using register as a decorator."""

        @registry.register("decorated_class")
        class DecoratedClass:
            pass

        assert "decorated_class" in registry
        assert registry.get("decorated_class") == DecoratedClass

    def test_register_decorator_without_name(self, registry):
        """Test using register as decorator without explicit name."""

        @registry.register()
        class AutoNamedClass:
            pass

        assert "AutoNamedClass" in registry

    def test_register_with_aliases(self, registry, sample_class):
        """Test registering with aliases."""
        registry.register("main_name", sample_class, aliases=["alias1", "alias2"])

        assert "main_name" in registry
        assert "alias1" in registry
        assert "alias2" in registry
        assert registry.get("alias1") == sample_class
        assert registry.get("alias2") == sample_class

    def test_register_with_metadata(self, registry, sample_class):
        """Test registering with additional metadata."""
        registry.register("with_meta", sample_class, version="1.0", author="test")

        metadata = registry.get_metadata("with_meta")
        assert metadata is not None
        assert metadata["version"] == "1.0"
        assert metadata["author"] == "test"
        assert metadata["name"] == "with_meta"

    def test_duplicate_registration_raises_error(self, registry, sample_class, another_class):
        """Test that duplicate registration raises error."""
        registry.register("duplicate", sample_class)

        with pytest.raises(RegistryError):
            registry.register("duplicate", another_class)

    def test_duplicate_registration_with_force(self, registry, sample_class, another_class):
        """Test that force=True allows overwriting."""
        registry.register("overwrite", sample_class)
        registry.register("overwrite", another_class, force=True)

        assert registry.get("overwrite") == another_class

    def test_duplicate_alias_raises_error(self, registry, sample_class, another_class):
        """Test that duplicate alias raises error."""
        registry.register("first", sample_class, aliases=["shared"])

        with pytest.raises(RegistryError):
            registry.register("second", another_class, aliases=["shared"])

    def test_get_nonexistent_returns_none(self, registry):
        """Test that getting nonexistent key returns None."""
        assert registry.get("nonexistent") is None

    def test_get_with_default(self, registry, sample_class):
        """Test getting with default value."""
        default = sample_class
        result = registry.get("nonexistent", default=default)
        assert result == default

    def test_getitem_raises_keyerror(self, registry):
        """Test that [] access raises KeyError for missing keys."""
        with pytest.raises(KeyError):
            _ = registry["nonexistent"]

    def test_contains(self, registry, sample_class):
        """Test contains check."""
        registry.register("exists", sample_class)

        assert "exists" in registry
        assert registry.contains("exists")
        assert "not_exists" not in registry
        assert not registry.contains("not_exists")

    def test_create_instance(self, registry, sample_class):
        """Test creating instance from registry."""
        registry.register("creatable", sample_class)

        instance = registry.create("creatable", param1="value1", param2=42)
        assert isinstance(instance, sample_class)
        assert instance.param1 == "value1"
        assert instance.param2 == 42

    def test_create_nonexistent_raises_error(self, registry):
        """Test creating nonexistent class raises error."""
        with pytest.raises(RegistryError):
            registry.create("nonexistent")

    def test_unregister(self, registry, sample_class):
        """Test unregistering a component."""
        registry.register("to_remove", sample_class, aliases=["alias"])

        registry.unregister("to_remove")

        assert "to_remove" not in registry
        assert "alias" not in registry

    def test_unregister_by_alias(self, registry, sample_class):
        """Test unregistering by alias."""
        registry.register("main", sample_class, aliases=["alias"])

        registry.unregister("alias")

        assert "main" not in registry
        assert "alias" not in registry

    def test_unregister_nonexistent_raises_error(self, registry):
        """Test unregistering nonexistent raises error."""
        with pytest.raises(RegistryError):
            registry.unregister("nonexistent")

    def test_list_registered_names(self, registry, sample_class, another_class):
        """Test listing all registered names."""
        registry.register("class_a", sample_class)
        registry.register("class_b", another_class)

        names = registry.list()
        assert sorted(names) == ["class_a", "class_b"]

    def test_list_with_aliases(self, registry, sample_class, another_class):
        """Test listing names with their aliases."""
        registry.register("main1", sample_class, aliases=["a1", "a2"])
        registry.register("main2", another_class)

        result = registry.list_with_aliases()
        assert "main1" in result
        assert "main2" in result
        assert set(result["main1"]) == {"a1", "a2"}
        assert result["main2"] == []

    def test_clear(self, registry, sample_class, another_class):
        """Test clearing all registrations."""
        registry.register("class1", sample_class)
        registry.register("class2", another_class, aliases=["alias"])

        registry.clear()

        assert len(registry) == 0
        assert "class1" not in registry
        assert "alias" not in registry

    def test_len(self, registry, sample_class, another_class):
        """Test length."""
        assert len(registry) == 0

        registry.register("c1", sample_class)
        assert len(registry) == 1

        registry.register("c2", another_class)
        assert len(registry) == 2

    def test_repr(self, registry, sample_class):
        """Test string representation."""
        registry.register("test_class", sample_class)

        repr_str = repr(registry)
        assert "test" in repr_str
        assert "1 items" in repr_str
        assert "test_class" in repr_str


class TestRegistryThreadSafety:
    """Test thread safety of Registry."""

    def test_concurrent_registration(self):
        """Test concurrent registration from multiple threads."""
        registry = Registry("concurrent")
        num_threads = 10
        num_items = 100
        results = []

        def register_items(thread_id):
            for i in range(num_items):
                name = f"item_{thread_id}_{i}"

                class DummyClass:
                    pass

                try:
                    registry.register(name, DummyClass)
                    results.append((thread_id, i, True))
                except RegistryError:
                    results.append((thread_id, i, False))

        threads = [
            threading.Thread(target=register_items, args=(i,)) for i in range(num_threads)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All unique registrations should succeed
        assert len(registry) == num_threads * num_items

    def test_concurrent_get(self, sample_class):
        """Test concurrent get operations."""
        registry = Registry("get_test")
        registry.register("shared", sample_class)

        results = []

        def get_item():
            for _ in range(100):
                result = registry.get("shared")
                results.append(result == sample_class)

        threads = [threading.Thread(target=get_item) for _ in range(10)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert all(results)


class TestDecoratorRegistration:
    """Test suite for decorator-based registration."""

    def test_register_decorator_returns_original_class(self):
        """Test that decorator returns the original class."""
        registry = Registry("decorator_test")

        @registry.register("my_class")
        class MyClass:
            pass

        # The decorator should return the original class
        instance = MyClass()
        assert isinstance(instance, MyClass)

    def test_register_decorator_with_all_options(self):
        """Test decorator with all registration options."""
        registry = Registry("full_decorator")

        @registry.register(
            "full_class", aliases=["fc", "full"], force=False, version="2.0"
        )
        class FullClass:
            pass

        assert "full_class" in registry
        assert "fc" in registry
        assert "full" in registry
        metadata = registry.get_metadata("full_class")
        assert metadata["version"] == "2.0"

    def test_multiple_decorators_on_same_class(self):
        """Test registering same class to multiple registries."""
        registry1 = Registry("reg1")
        registry2 = Registry("reg2")

        @registry1.register("shared")
        @registry2.register("shared")
        class SharedClass:
            pass

        assert "shared" in registry1
        assert "shared" in registry2


class TestGlobalRegistry:
    """Test suite for GlobalRegistry."""

    def setup_method(self):
        """Clear global registry before each test."""
        GlobalRegistry.clear_all()

    def test_get_registry_creates_new(self):
        """Test getting a new registry creates it."""
        registry = GlobalRegistry.get_registry("new_registry")

        assert isinstance(registry, Registry)
        assert registry.name == "new_registry"

    def test_get_registry_returns_existing(self):
        """Test getting existing registry returns same instance."""
        registry1 = GlobalRegistry.get_registry("shared")
        registry2 = GlobalRegistry.get_registry("shared")

        assert registry1 is registry2

    def test_list_registries(self):
        """Test listing all registry names."""
        GlobalRegistry.get_registry("reg_a")
        GlobalRegistry.get_registry("reg_b")
        GlobalRegistry.get_registry("reg_c")

        names = GlobalRegistry.list_registries()
        assert sorted(names) == ["reg_a", "reg_b", "reg_c"]

    def test_clear_all(self):
        """Test clearing all registries."""
        GlobalRegistry.get_registry("to_clear")
        GlobalRegistry.clear_all()

        assert GlobalRegistry.list_registries() == []

    def test_register_to_global_registry(self, sample_class):
        """Test registering components through global registry."""
        models = GlobalRegistry.get_registry("models")
        models.register("my_model", sample_class)

        # Get again and verify component is there
        models_again = GlobalRegistry.get_registry("models")
        assert "my_model" in models_again


class TestBuildFromConfig:
    """Test suite for build_from_config function."""

    def test_build_from_simple_config(self, sample_class):
        """Test building object from simple config."""
        registry = Registry("build_test")
        registry.register("SimpleClass", sample_class)

        config = {"type": "SimpleClass", "param1": "value1", "param2": 42}
        instance = build_from_config(config, registry)

        assert isinstance(instance, sample_class)
        assert instance.param1 == "value1"
        assert instance.param2 == 42

    def test_build_from_config_with_kwargs(self, sample_class):
        """Test building with additional kwargs."""
        registry = Registry("kwargs_test")
        registry.register("MyClass", sample_class)

        config = {"type": "MyClass", "param1": "from_config"}
        instance = build_from_config(config, registry, param2="from_kwargs")

        assert instance.param1 == "from_config"
        assert instance.param2 == "from_kwargs"

    def test_build_kwargs_override_config(self, sample_class):
        """Test that kwargs override config values."""
        registry = Registry("override_test")
        registry.register("MyClass", sample_class)

        config = {"type": "MyClass", "param1": "from_config"}
        instance = build_from_config(config, registry, param1="from_kwargs")

        assert instance.param1 == "from_kwargs"

    def test_build_missing_type_raises_error(self):
        """Test that missing 'type' key raises error."""
        registry = Registry("error_test")

        with pytest.raises(KeyError):
            build_from_config({"param": "value"}, registry)

    def test_build_invalid_config_type_raises_error(self):
        """Test that non-dict config raises error."""
        registry = Registry("type_error_test")

        with pytest.raises(TypeError):
            build_from_config("not a dict", registry)

    def test_build_unregistered_type_raises_error(self):
        """Test building unregistered type raises error."""
        registry = Registry("unregistered_test")

        config = {"type": "NonexistentClass"}
        with pytest.raises(RegistryError):
            build_from_config(config, registry)

    @pytest.mark.parametrize(
        "config",
        [
            {"type": "TestClass"},
            {"type": "TestClass", "param1": "value1"},
            {"type": "TestClass", "param1": "value1", "param2": 42},
        ],
    )
    def test_build_with_various_configs(self, config, sample_class):
        """Test building with various configuration formats."""
        registry = Registry("various_test")
        registry.register("TestClass", sample_class)

        instance = build_from_config(config, registry)
        assert isinstance(instance, sample_class)

    def test_build_does_not_modify_original_config(self, sample_class):
        """Test that build_from_config doesn't modify original config."""
        registry = Registry("immutable_test")
        registry.register("MyClass", sample_class)

        original_config = {"type": "MyClass", "param1": "value"}
        config_copy = original_config.copy()

        build_from_config(original_config, registry)

        assert original_config == config_copy
