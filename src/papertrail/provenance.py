"""Shared provenance helpers for every PaperTrail architecture.

What this module does (in one paragraph):
    Provides the three functions every architecture needs to populate
    ``BenchmarkResult.provenance`` — the package version, the current
    git SHA, and a fully-built ``Provenance`` instance. Extracted from
    ``arch_00_baseline.baseline`` once a second concrete architecture
    (arch_01) appeared and needed the same logic. Lives at the
    ``papertrail`` package root because provenance is a cross-cutting
    concern, not an architecture-internal detail.

Why best-effort, not strict:
    Provenance is nice-to-have for reproducibility; it must never be
    the reason a run dies. If git is unavailable or slow, ``git_sha``
    returns ``"unknown"`` rather than raising. The schema enforces
    ``min_length=1`` on the field, which the sentinel satisfies.

Cert mapping:
    No direct cert hook. Indirectly supports D5 TS 5.6 (preserving
    information provenance) by giving every architecture's output a
    consistent reproducibility record.

Libraries: stdlib only (importlib.metadata, subprocess), plus
``papertrail`` for ``__version__`` fallback and ``papertrail.benchmark``
for ``SCHEMA_VERSION`` + ``Provenance``.
"""

from __future__ import annotations

import importlib.metadata
import subprocess
from datetime import UTC, datetime
from typing import Final

import papertrail
from papertrail.benchmark import SCHEMA_VERSION, Provenance

# Cap on how long ``git rev-parse`` is allowed to take. Provenance is
# nice to have, not critical — if git is slow or unavailable, fall back
# to "unknown" rather than blocking the run.
_GIT_SHA_TIMEOUT_SECONDS: Final[float] = 2.0

# Sentinel returned by ``git_sha`` when git is unavailable, errors, or
# times out. ``Provenance.git_sha`` requires ``min_length=1``; this
# satisfies the schema and is visibly-wrong enough to flag at review.
_UNKNOWN_SHA: Final[str] = "unknown"


def papertrail_version() -> str:
    """Read the package version. Falls back to the in-module ``__version__``.

    ``importlib.metadata`` is the canonical Python way to read a package's
    declared version; in editable installs it can occasionally fail, so we
    fall back to the constant declared in ``papertrail.__init__``.
    """
    try:
        return importlib.metadata.version("papertrail")
    except importlib.metadata.PackageNotFoundError:
        return papertrail.__version__


def git_sha() -> str:
    """Best-effort short git SHA of the current checkout, or ``"unknown"``.

    Network-free (uses local git). Capped at ``_GIT_SHA_TIMEOUT_SECONDS``
    so a slow filesystem or a hung git can't block a run. ``"unknown"``
    is the documented sentinel for missing-git environments.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],
            capture_output=True,
            text=True,
            timeout=_GIT_SHA_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return _UNKNOWN_SHA
    if result.returncode != 0:
        return _UNKNOWN_SHA
    sha = result.stdout.strip()
    return sha or _UNKNOWN_SHA


def build_provenance(*, created_at: datetime | None = None) -> Provenance:
    """Assemble a fully-populated ``Provenance`` for the current run.

    ``created_at`` defaults to ``datetime.now(UTC)``. Callers that already
    have a timestamp (e.g. an orchestrator's ``finished_at``) should pass
    it explicitly so the provenance and the telemetry agree.
    """
    return Provenance(
        schema_version=SCHEMA_VERSION,
        papertrail_version=papertrail_version(),
        git_sha=git_sha(),
        created_at=created_at or datetime.now(UTC),
    )


__all__ = ["build_provenance", "git_sha", "papertrail_version"]
