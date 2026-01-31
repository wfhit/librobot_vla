"""
Unit tests for the version module.

Tests version information retrieval and consistency.
"""

import pytest

from librobot.version import (
    VERSION_INFO,
    __author__,
    __license__,
    __version__,
    get_version,
    get_version_info,
)


class TestVersionConstants:
    """Test suite for version constants."""

    def test_version_is_string(self):
        """Test that __version__ is a string."""
        assert isinstance(__version__, str)

    def test_version_format(self):
        """Test that version follows semver format."""
        parts = __version__.split(".")

        assert len(parts) >= 2
        assert all(part.isdigit() for part in parts[:2])

    def test_author_is_string(self):
        """Test that __author__ is a string."""
        assert isinstance(__author__, str)
        assert len(__author__) > 0

    def test_license_is_string(self):
        """Test that __license__ is a string."""
        assert isinstance(__license__, str)
        assert __license__ == "MIT"

    def test_version_info_is_dict(self):
        """Test that VERSION_INFO is a dictionary."""
        assert isinstance(VERSION_INFO, dict)

    def test_version_info_has_required_keys(self):
        """Test that VERSION_INFO has all required keys."""
        required_keys = ["major", "minor", "patch"]

        for key in required_keys:
            assert key in VERSION_INFO

    def test_version_info_types(self):
        """Test that VERSION_INFO values have correct types."""
        assert isinstance(VERSION_INFO["major"], int)
        assert isinstance(VERSION_INFO["minor"], int)
        assert isinstance(VERSION_INFO["patch"], int)

    def test_version_info_optional_keys(self):
        """Test optional keys in VERSION_INFO."""
        # These can be None or string
        if "prerelease" in VERSION_INFO:
            assert VERSION_INFO["prerelease"] is None or isinstance(VERSION_INFO["prerelease"], str)

        if "build" in VERSION_INFO:
            assert VERSION_INFO["build"] is None or isinstance(VERSION_INFO["build"], str)


class TestGetVersion:
    """Test suite for get_version function."""

    def test_get_version_returns_string(self):
        """Test that get_version returns a string."""
        version = get_version()

        assert isinstance(version, str)

    def test_get_version_matches_constant(self):
        """Test that get_version returns same as __version__."""
        version = get_version()

        assert version == __version__

    def test_get_version_not_empty(self):
        """Test that get_version returns non-empty string."""
        version = get_version()

        assert len(version) > 0

    def test_get_version_consistent(self):
        """Test that multiple calls return same version."""
        version1 = get_version()
        version2 = get_version()

        assert version1 == version2


class TestGetVersionInfo:
    """Test suite for get_version_info function."""

    def test_get_version_info_returns_dict(self):
        """Test that get_version_info returns a dictionary."""
        info = get_version_info()

        assert isinstance(info, dict)

    def test_get_version_info_has_required_keys(self):
        """Test that returned dict has required keys."""
        info = get_version_info()
        required_keys = ["major", "minor", "patch"]

        for key in required_keys:
            assert key in info

    def test_get_version_info_returns_copy(self):
        """Test that get_version_info returns a copy, not original."""
        info1 = get_version_info()
        info2 = get_version_info()

        # Should be equal but not same object
        assert info1 == info2
        assert info1 is not VERSION_INFO
        assert info2 is not VERSION_INFO

    def test_get_version_info_modification_safe(self):
        """Test that modifying returned dict doesn't affect original."""
        info = get_version_info()
        original_major = VERSION_INFO["major"]

        info["major"] = 999

        assert VERSION_INFO["major"] == original_major

    def test_version_info_consistency(self):
        """Test that VERSION_INFO and get_version_info are consistent."""
        info = get_version_info()

        assert info["major"] == VERSION_INFO["major"]
        assert info["minor"] == VERSION_INFO["minor"]
        assert info["patch"] == VERSION_INFO["patch"]


class TestVersionConsistency:
    """Test version consistency across different accessors."""

    def test_version_string_matches_info(self):
        """Test that version string matches VERSION_INFO components."""
        version = get_version()
        info = get_version_info()

        expected = f"{info['major']}.{info['minor']}.{info['patch']}"

        assert version == expected or version.startswith(expected)

    @pytest.mark.parametrize(
        "component,min_val,max_val",
        [
            ("major", 0, 100),
            ("minor", 0, 100),
            ("patch", 0, 1000),
        ],
    )
    def test_version_components_in_range(self, component, min_val, max_val):
        """Test that version components are in reasonable ranges."""
        info = get_version_info()

        assert min_val <= info[component] <= max_val
