"""mi_crow: helper package for the Engineer Thesis project.

This module is intentionally minimal. It exists to define the top-level package
and to enable code coverage to include the package. Importing it should succeed
without side effects.
"""

# A tiny bit of executable code to make the package measurable by coverage.
PACKAGE_NAME = "mi_crow"
__version__ = "0.0.0"


def ping() -> str:
    """Return a simple response to verify the package is wired correctly."""
    return "pong"
