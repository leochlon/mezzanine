import pytest
from mezzanine.core.registry import Registry


def test_register_and_get():
    registry = Registry(kind="test")

    @registry.register("example")
    class Example:
        pass

    assert registry.get("example") is Example


def test_missing_key_raises():
    registry = Registry(kind="test")

    with pytest.raises(KeyError):
        registry.get("does_not_exist")


def test_duplicate_registration_raises():
    registry = Registry(kind="test")

    @registry.register("dup")
    class A:
        pass

    with pytest.raises(KeyError):
        @registry.register("dup")
        class B:
            pass


def test_name_attribute_fallback():
    registry = Registry(kind="test")
    
    class MyClass:
        NAME = "custom_name"

    registry.register()(MyClass)

    assert registry.get("custom_name") is MyClass


def test_contains_and_list():
    registry = Registry(kind="test")

    @registry.register("item", description="test description")
    class Item:
        pass

    assert "item" in registry

    listed = registry.list()
    assert "item" in listed
    assert listed["item"]["description"] == "test description"
    assert listed["item"]["object"] == "Item"