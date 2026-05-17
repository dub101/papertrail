"""Per-package test fixtures for tests under ``tests/tools/``.

Bypasses the real ``asyncio.sleep`` and resets the arxiv pacing state
between every test in this directory. Without it, the inter-request
pacing (``MIN_INTER_REQUEST_SECONDS = 4s``) would slow the tool test
suite proportionally to the number of arxiv-touching tests.

Individual tests that need to inspect specific sleep calls re-monkeypatch
``asyncio.sleep`` inside the test body; the test-local override wins
over this fixture.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _bypass_arxiv_pacing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reset arxiv pacing state and short-circuit ``asyncio.sleep``.

    Applies to every test under ``tests/tools/`` because the arxiv tool
    layer is the only place ``_pace_request`` runs, and every test that
    exercises that layer would otherwise wait 4 real seconds on the
    second-and-later call.
    """
    import papertrail.tools.arxiv as _arxiv_mod

    monkeypatch.setattr(_arxiv_mod, "_last_request_monotonic", 0.0)

    async def _fast_sleep(seconds: float) -> None:
        _ = seconds

    monkeypatch.setattr("papertrail.tools.arxiv.asyncio.sleep", _fast_sleep)
