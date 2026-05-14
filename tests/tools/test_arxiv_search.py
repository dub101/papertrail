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


async def test_429_retries_exactly_once_then_succeeds(httpx_mock: HTTPXMock) -> None:
    """The arXiv protocol: 429 -> wait 3s -> retry -> success."""
    httpx_mock.add_response(status_code=429)
    httpx_mock.add_response(text=_xml("arxiv_search_sample.xml"))

    papers = await arxiv_search("attention", max_results=2)
    assert len(papers) == 2
    # Two requests were sent: the initial that 429'd, and the retry that
    # succeeded. ``pytest-httpx`` exposes both via ``get_requests``.
    assert len(httpx_mock.get_requests()) == 2


async def test_double_429_raises_status_error(httpx_mock: HTTPXMock) -> None:
    """If the retry also gets 429, we re-raise — no second retry."""
    httpx_mock.add_response(status_code=429)
    httpx_mock.add_response(status_code=429)

    with pytest.raises(httpx.HTTPStatusError):
        await arxiv_search("attention", max_results=2)
    # Exactly two attempts, not three.
    assert len(httpx_mock.get_requests()) == 2


async def test_500_does_not_retry(httpx_mock: HTTPXMock) -> None:
    """5xx is a server fault, not a polite-wait scenario. Re-raise
    immediately so the architecture's resilience layer decides what to do."""
    httpx_mock.add_response(status_code=500)

    with pytest.raises(httpx.HTTPStatusError):
        await arxiv_search("attention", max_results=2)
    # Single attempt — no implicit retry on 5xx.
    assert len(httpx_mock.get_requests()) == 1


async def test_429_uses_arxiv_documented_wait(
    httpx_mock: HTTPXMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The wait between the two attempts is exactly the constant ``arxiv``
    documents (3.0s). We assert by monkeypatching ``asyncio.sleep`` inside
    the module under test so the test runs fast and we can inspect what
    wait was requested."""
    from papertrail.tools.arxiv import RATE_LIMIT_WAIT_SECONDS

    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    # String form of setattr lets mypy stay strict (no need to reach inside
    # the module for ``asyncio``).
    monkeypatch.setattr("papertrail.tools.arxiv.asyncio.sleep", fake_sleep)

    httpx_mock.add_response(status_code=429)
    httpx_mock.add_response(text=_xml("arxiv_search_sample.xml"))

    await arxiv_search("attention", max_results=2)
    assert sleeps == [RATE_LIMIT_WAIT_SECONDS]
