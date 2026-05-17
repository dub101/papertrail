"""Tests for ``arxiv_search`` — happy path, empty result, validation, 429 retry.

All HTTP is mocked via ``pytest-httpx``; **no test in this file hits the live
arXiv API**. Marker ``integration`` is reserved for any future tests that do.

The 429-retry tests are the most important ones: they freeze the exact arXiv
protocol behaviour we promised to encode at the tool layer. If a future
refactor breaks "exactly one retry on 429 only", these tests catch it.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
from pytest_httpx import HTTPXMock

from papertrail.tools.arxiv import (
    ARXIV_API_BASE,
    MAX_RESULTS_CEILING,
    USER_AGENT,
    arxiv_search,
)

FIXTURES = Path(__file__).parent.parent / "fixtures"


def _xml(name: str) -> str:
    """Load a fixture XML file as a string."""
    return (FIXTURES / name).read_text()


# Pacing bypass fixture lives in ``tests/tools/conftest.py`` so it
# applies to ``test_arxiv_fetch.py`` and ``test_surface_smoke.py`` too.
# Tests that inspect specific sleeps re-monkeypatch ``asyncio.sleep``
# inside the test body; the local override wins over the conftest one.


# ───── Happy path ───────────────────────────────────────────────────────


async def test_search_returns_parsed_papers(httpx_mock: HTTPXMock) -> None:
    """Two-entry fixture parses to two ``ArxivPaper`` instances with the
    expected fields populated."""
    httpx_mock.add_response(
        url=httpx.URL(
            ARXIV_API_BASE,
            params={
                "search_query": "all:attention",
                "start": "0",
                "max_results": "2",
                "sortBy": "relevance",
                "sortOrder": "descending",
            },
        ),
        text=_xml("arxiv_search_sample.xml"),
    )

    papers = await arxiv_search("attention", max_results=2)

    assert len(papers) == 2
    first, second = papers

    assert first.arxiv_id == "1706.03762v5"
    assert first.title == "Attention Is All You Need"
    assert first.authors == ["Ashish Vaswani", "Noam Shazeer", "Niki Parmar"]
    assert "cs.CL" in first.categories and "cs.LG" in first.categories
    # ``HttpUrl`` normalisation adds a trailing slash — assert with str() to
    # cover that, not with a raw equality on the input string.
    assert str(first.entry_url).startswith("http://arxiv.org/abs/1706.03762v5")
    assert str(first.pdf_url).startswith("http://arxiv.org/pdf/1706.03762v5")
    # Whitespace from ``<summary>`` should already be collapsed by ``_text``.
    assert first.abstract.startswith("The dominant sequence transduction models")
    assert "  " not in first.abstract  # no double-spaces

    assert second.arxiv_id == "2205.14135v2"
    assert second.title.startswith("FlashAttention")
    assert second.authors == ["Tri Dao"]


async def test_search_sends_user_agent_header(httpx_mock: HTTPXMock) -> None:
    """The polite-client User-Agent must be on the wire — arXiv uses it as
    their primary contact channel if our traffic ever misbehaves."""
    httpx_mock.add_response(text=_xml("arxiv_search_sample.xml"))
    await arxiv_search("attention", max_results=2)
    request = httpx_mock.get_request()
    assert request is not None
    assert request.headers["user-agent"] == USER_AGENT


async def test_empty_result_returns_empty_list(httpx_mock: HTTPXMock) -> None:
    """A zero-entry feed is a legitimate outcome, not an error."""
    httpx_mock.add_response(text=_xml("arxiv_search_empty.xml"))
    papers = await arxiv_search("zzznoresults", max_results=5)
    assert papers == []


async def test_missing_pdf_link_is_synthesised_from_entry_url(httpx_mock: HTTPXMock) -> None:
    """Older arXiv responses occasionally omit the explicit PDF ``<link>``.
    We synthesise the PDF URL by swapping ``/abs/`` for ``/pdf/`` in the
    entry URL, so the model can still construct successfully."""
    # Hand-rolled inline fixture: one entry with only the ``rel="alternate"``
    # link, no ``title="pdf"`` link. Smaller than a separate fixture file.
    inline_xml = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/0001.0001v1</id>
    <updated>2024-01-01T00:00:00Z</updated>
    <published>2024-01-01T00:00:00Z</published>
    <title>Synthetic PDF Test</title>
    <summary>Single sentence abstract.</summary>
    <author><name>Test Author</name></author>
    <link href="http://arxiv.org/abs/0001.0001v1" rel="alternate" type="text/html"/>
  </entry>
</feed>"""
    httpx_mock.add_response(text=inline_xml)
    papers = await arxiv_search("anything", max_results=1)
    assert len(papers) == 1
    assert str(papers[0].pdf_url).startswith("http://arxiv.org/pdf/0001.0001v1")


async def test_sort_by_parameter_is_passed_through(httpx_mock: HTTPXMock) -> None:
    """``sort_by`` ends up in the outgoing query string verbatim."""
    httpx_mock.add_response(text=_xml("arxiv_search_empty.xml"))
    await arxiv_search("anything", max_results=1, sort_by="submittedDate")
    request = httpx_mock.get_request()
    assert request is not None
    assert b"sortBy=submittedDate" in request.url.query


# ───── max_results validation ───────────────────────────────────────────


@pytest.mark.parametrize("bad_value", [0, -1, MAX_RESULTS_CEILING + 1, 10_000])
async def test_max_results_outside_band_is_rejected(bad_value: int) -> None:
    """Below 1 or above the ceiling is rejected at the boundary, no HTTP
    call made."""
    with pytest.raises(ValueError, match="max_results must be between 1"):
        await arxiv_search("anything", max_results=bad_value)


# ───── 429 retry behaviour (the protocol-level guarantee) ───────────────


async def test_429_then_success_uses_one_retry(httpx_mock: HTTPXMock) -> None:
    """One 429 followed by success: two requests sent total."""
    httpx_mock.add_response(status_code=429)
    httpx_mock.add_response(text=_xml("arxiv_search_sample.xml"))

    papers = await arxiv_search("attention", max_results=2)
    assert len(papers) == 2
    assert len(httpx_mock.get_requests()) == 2


async def test_four_consecutive_429s_raise(
    httpx_mock: HTTPXMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If all 4 attempts (initial + 3 retries) get 429, re-raise."""

    # Monkeypatch sleep so the test doesn't wait the real 65 seconds.
    async def fake_sleep(seconds: float) -> None:
        _ = seconds

    monkeypatch.setattr("papertrail.tools.arxiv.asyncio.sleep", fake_sleep)

    for _ in range(4):
        httpx_mock.add_response(status_code=429)

    with pytest.raises(httpx.HTTPStatusError):
        await arxiv_search("attention", max_results=2)
    # Exactly 4 attempts (initial + the three backoff retries), not 5.
    assert len(httpx_mock.get_requests()) == 4


async def test_500_does_not_retry(httpx_mock: HTTPXMock) -> None:
    """5xx is a server fault, not a polite-wait scenario. Re-raise
    immediately so the architecture's resilience layer decides what to do."""
    httpx_mock.add_response(status_code=500)

    with pytest.raises(httpx.HTTPStatusError):
        await arxiv_search("attention", max_results=2)
    # Single attempt — no implicit retry on 5xx.
    assert len(httpx_mock.get_requests()) == 1


async def test_429_uses_exponential_backoff_schedule(
    httpx_mock: HTTPXMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When all 4 attempts 429, the backoff sleeps match the documented
    schedule (5s, 15s, 45s). Pacing is monkeypatched to a no-op so the
    assertion captures only backoff waits — pacing has its own test."""
    from papertrail.tools.arxiv import _RATE_LIMIT_BACKOFF_SECONDS

    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    async def fake_pace() -> None:
        return None

    monkeypatch.setattr("papertrail.tools.arxiv.asyncio.sleep", fake_sleep)
    monkeypatch.setattr("papertrail.tools.arxiv._pace_request", fake_pace)

    for _ in range(4):
        httpx_mock.add_response(status_code=429)

    with pytest.raises(httpx.HTTPStatusError):
        await arxiv_search("attention", max_results=2)
    assert sleeps == list(_RATE_LIMIT_BACKOFF_SECONDS)


async def test_inter_request_pacing_sleeps_at_least_min_seconds(
    httpx_mock: HTTPXMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two consecutive arxiv_search calls sleep at least
    MIN_INTER_REQUEST_SECONDS between them (subject to elapsed time
    being smaller than the gap)."""
    from papertrail.tools.arxiv import MIN_INTER_REQUEST_SECONDS

    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr("papertrail.tools.arxiv.asyncio.sleep", fake_sleep)
    # Two successful responses, back to back.
    httpx_mock.add_response(text=_xml("arxiv_search_sample.xml"))
    httpx_mock.add_response(text=_xml("arxiv_search_sample.xml"))

    await arxiv_search("attention", max_results=2)
    await arxiv_search("attention", max_results=2)

    # First call: pacing sees a huge elapsed (last=0.0), no sleep needed.
    # Second call: pacing sees a tiny elapsed (just after first GET),
    # sleeps roughly MIN_INTER_REQUEST_SECONDS.
    assert len(sleeps) == 1
    # Tolerance accounts for the parse + httpx setup time between calls
    # (tens of ms in CI / a few ms locally). The sleep is computed as
    # MIN - elapsed, so a slightly faster path produces a slightly
    # smaller sleep but never more than MIN.
    assert MIN_INTER_REQUEST_SECONDS - 0.5 < sleeps[0] <= MIN_INTER_REQUEST_SECONDS
