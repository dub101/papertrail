"""Smoke test: confirms the package is importable and the test runner is wired up.

This is intentionally minimal — its purpose is to give CI something to run on day one,
not to test anything meaningful. Real tests land with each architecture.
"""

from __future__ import annotations

import papertrail


def test_package_imports() -> None:
    """The `papertrail` package can be imported."""
    assert papertrail.__name__ == "papertrail"


def test_package_exposes_version() -> None:
    """The package exposes `__version__` as a non-empty string."""
    assert isinstance(papertrail.__version__, str)
    assert papertrail.__version__


def test_architectures_subpackage_imports() -> None:
    """The `papertrail.architectures` subpackage can be imported."""
    from papertrail import architectures

    assert architectures.__name__ == "papertrail.architectures"
