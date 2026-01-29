"""Test registry system."""

import pytest

from librobot.utils.registry import REGISTRY, Registry, register_action_head


def test_registry_register_and_get():
    """Test registering and retrieving components."""
    registry = Registry()

    # Create a dummy class
    class DummyModel:
        pass

    # Register it
    registry.register("vlm", "dummy", DummyModel)

    # Retrieve it
    retrieved = registry.get("vlm", "dummy")
    assert retrieved is DummyModel


def test_registry_aliases():
    """Test component aliases."""
    registry = Registry()

    class DummyModel:
        pass

    # Register with aliases
    registry.register("vlm", "dummy", DummyModel, aliases=["d", "dum"])

    # Retrieve via alias
    assert registry.get("vlm", "d") is DummyModel
    assert registry.get("vlm", "dum") is DummyModel


def test_registry_decorator():
    """Test decorator registration."""

    @register_action_head("test_head")
    class TestHead:
        pass

    # Check it's registered in global registry
    assert REGISTRY.is_registered("action_head", "test_head")
    retrieved = REGISTRY.get("action_head", "test_head")
    assert retrieved is TestHead


def test_registry_list():
    """Test listing components."""
    registry = Registry()

    class Model1:
        pass

    class Model2:
        pass

    registry.register("vlm", "model1", Model1)
    registry.register("vlm", "model2", Model2)

    models = registry.list("vlm")
    assert "model1" in models
    assert "model2" in models


def test_registry_duplicate_error():
    """Test error on duplicate registration."""
    registry = Registry()

    class DummyModel:
        pass

    registry.register("vlm", "dummy", DummyModel)

    # Should raise error on duplicate
    with pytest.raises(ValueError, match="already registered"):
        registry.register("vlm", "dummy", DummyModel)
