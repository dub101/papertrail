"""Tests for ``papertrail.provenance``.

These tests previously lived in ``tests/architectures/test_arch_00_baseline.py``
where the helpers were originally defined. They moved here when the
helpers themselves moved out of arch_00 into the shared
``papertrail.provenance`` module (extracted once arch_01 needed the
same helpers).
"""

from __future__ import annotations

from datetime import UTC, datetime

from papertrail.benchmark import SCHEMA_VERSION
from papertrail.provenance import build_provenance, git_sha, papertrail_version


def test_papertrail_version_returns_non_empty_string() -> None:
    """Returns either the metadata version or the in-module fallback."""
    v = papertrail_version()
    assert isinstance(v, str)
    assert v


def test_git_sha_returns_non_empty_string() -> None:
    """Returns a hex sha when git is available; the 'unknown' sentinel otherwise."""
    sha = git_sha()
    assert isinstance(sha, str)
    assert sha  # Either a hex sha or "unknown"; never empty.


def test_build_provenance_returns_valid_provenance() -> None:
    """``build_provenance`` assembles a schema-valid Provenance with current values."""
    prov = build_provenance()
    assert prov.schema_version == SCHEMA_VERSION
    assert prov.papertrail_version  # non-empty
    assert prov.git_sha  # non-empty (sha or "unknown")
    # ``created_at`` defaults to now-ish.
    delta = datetime.now(UTC) - prov.created_at
    assert 0 <= delta.total_seconds() < 5


def test_build_provenance_honors_supplied_created_at() -> None:
    """When ``created_at`` is passed, that timestamp goes into the record verbatim."""
    fixed = datetime(2026, 5, 16, 12, 0, 0, tzinfo=UTC)
    prov = build_provenance(created_at=fixed)
    assert prov.created_at == fixed
