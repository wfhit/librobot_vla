"""Version information for LibroBot VLA framework."""

__version__ = "0.1.0"
__author__ = "LibroBot Team"
__license__ = "MIT"

VERSION_INFO = {
    "major": 0,
    "minor": 1,
    "patch": 0,
    "prerelease": None,
    "build": None,
}


def get_version() -> str:
    """
    Get the current version string.

    Returns:
        str: Version string in format 'major.minor.patch'
    """
    return __version__


def get_version_info() -> dict:
    """
    Get detailed version information.

    Returns:
        dict: Dictionary containing version components
    """
    return VERSION_INFO.copy()
