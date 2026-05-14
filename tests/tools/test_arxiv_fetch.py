"""Tests for ``arxiv_fetch`` — happy path, not-found, validation, retry inheritance.

The 429-retry tests aren't duplicated here: ``arxiv_fetch`` and ``arxiv_search``
both call ``_get_with_arxiv_retry``, which already has thorough coverage in
``test_arxiv_search.py``. We add **one** retry test here so a future refactor
that wires fetch to a different HTTP path is caught.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
from pytest_httpx import HTTPXMock

from papertrail.tools.arxiv import ARXIV_API_BASE, arxiv_fetch

FIXTURES = Path(__file__).parent.parent / "fixtures"


def _xml(name: str) -> str:
    return (FIXTURES / name).read_text()


# ───── Happy path ───────────────────────────────────────────────────────


async def test_fetch_returns_single_paper(httpx_mock: HTTPXMock) -> None:
    """One-entry fixture parses to one ``ArxivPaper`` with expected fields."""
    httpx_mock.add_response(
        url=httpx.URL(ARXIV_API_BASE, params={"id_list": "1706.03762"}),
        text=_xml("arxiv_fetch_sample.xml"),
    )

    paper = await arxiv_fetch("1706.03762")

    assert paper.arxiv_id == "1706.03762v5"
    assert paper.title == "Attention Is All You Need"
    assert paper.authors == ["Ashish Vaswani", "Noam Shazeer"]
    assert "cs.CL" in paper.categories


async def test_fetch_accepts_versioned_id(httpx_mock: HTTPXMock) -> None:
    """Both ``"1706.03762"`` and ``"1706.03762v5"`` should be acceptable
    inputs — arXiv handles both server-side."""
    httpx_mock.add_response(text=_xml("arxiv_fetch_sample.xml"))
    paper = await arxiv_fetch("1706.03762v5")
    assert paper.arxiv_id == "1706.03762v5"


async def test_fetch_passes_id_list_param(httpx_mock: HTTPXMock) -> None:
    """The outgoing request uses ``id_list=`` (not ``search_query=``) —
    this is what semantically distinguishes fetch from search on the wire."""
    httpx_mock.add_response(text=_xml("arxiv_fetch_sample.xml"))
    await arxiv_fetch("1706.03762")
    request = httpx_mock.get_request()
    assert request is not None
    assert b"id_list=1706.03762" in request.url.query
    assert b"search_query=" not in request.url.query


# ───── Not-found semantics (the contract asymmetry vs. search) ──────────


async def test_unknown_id_raises_valueerror(httpx_mock: HTTPXMock) -> None:
    """The defining contract: a fetch that finds nothing is a ValueError,
    *not* an empty return. Asymmetric with ``arxiv_search`` on purpose."""
    httpx_mock.add_response(text=_xml("arxiv_fetch_notfound.xml"))

    with pytest.raises(ValueError, match=r"arxiv_id not found: 9999\.99999"):
        await arxiv_fetch("9999.99999")


@pytest.mark.parametrize("bad_id", ["", "   ", "\t\n"])
async def test_empty_or_whitespace_id_is_rejected_preflight(bad_id: str) -> None:
    """Empty/whitespace ids are rejected before any HTTP call."""
    with pytest.raises(ValueError, match="non-empty"):
        await arxiv_fetch(bad_id)


# ───── HTTP error pass-through ──────────────────────────────────────────


async def test_500_is_re_raised(httpx_mock: HTTPXMock) -> None:
    """fetch shares the no-retry-on-5xx contract with search."""
    httpx_mock.add_response(status_code=500)

    with pytest.raises(httpx.HTTPStatusError):
        await arxiv_fetch("1706.03762")
    assert len(httpx_mock.get_requests()) == 1


async def test_429_retries_once_then_succeeds(httpx_mock: HTTPXMock) -> None:
    """fetch inherits the arXiv 429-only retry policy from the shared helper.
    One assertion here is enough — full coverage lives in test_arxiv_search."""
    httpx_mock.add_response(status_code=429)
    httpx_mock.add_response(text=_xml("arxiv_fetch_sample.xml"))

    paper = await arxiv_fetch("1706.03762")
    assert paper.arxiv_id == "1706.03762v5"
    assert len(httpx_mock.get_requests()) == 2
