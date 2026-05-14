"""Smoke tests for the public tool-layer surface.

Purpose: prove the three tools compose end-to-end through their declared
contracts, and that the package's re-exports actually work. These are NOT
unit tests for individual tools (those live in ``test_arxiv_search.py``
etc.) — they exist to catch the class of regression where each tool is
fine on its own but the surface as a whole has drifted.

All HTTP is mocked; no live network.
"""

from __future__ import annotations

from pathlib import Path

from pytest_httpx import HTTPXMock

# Important: import via the package's public surface, not via the submodules.
# If a future refactor breaks the re-exports, these imports break and the
# tests catch it.
from papertrail.tools import (
    ArxivPaper,
    arxiv_fetch,
    arxiv_search,
    format_citation,
)

FIXTURES = Path(__file__).parent.parent / "fixtures"


def _xml(name: str) -> str:
    return (FIXTURES / name).read_text()


# ───── Surface composition ──────────────────────────────────────────────


async def test_search_then_cite_composes(httpx_mock: HTTPXMock) -> None:
    """arxiv_search -> format_citation: a fan-out over the result list.

    The architecture pattern this smoke-tests: "search for candidate papers,
    cite each one in the deliverable." If the types don't align, we catch
    it here at the surface boundary rather than in an architecture later.
    """
    httpx_mock.add_response(text=_xml("arxiv_search_sample.xml"))

    papers = await arxiv_search("attention", max_results=2)
    citations = [format_citation(p) for p in papers]

    # Both fixture papers cite cleanly.
    assert len(citations) == 2
    assert citations[0] == ("Vaswani et al. (2017). Attention Is All You Need. arXiv:1706.03762v5")
    assert citations[1].startswith("Dao (2022).")
    assert citations[1].endswith("arXiv:2205.14135v2")


async def test_fetch_then_cite_composes(httpx_mock: HTTPXMock) -> None:
    """arxiv_fetch -> format_citation: a targeted lookup followed by cite.

    The architecture pattern this smoke-tests: "I have an arxiv_id from a
    citation context; produce its formatted citation line." This is the
    fetch counterpart to the search-then-cite chain above.
    """
    httpx_mock.add_response(text=_xml("arxiv_fetch_sample.xml"))

    paper = await arxiv_fetch("1706.03762")
    citation = format_citation(paper)

    assert citation == ("Vaswani and Shazeer (2017). Attention Is All You Need. arXiv:1706.03762v5")


def test_re_exports_are_correct_types() -> None:
    """The package's re-exports point at the right things — not stale aliases
    or accidental shadowing during a future refactor."""
    assert ArxivPaper.__name__ == "ArxivPaper"
    assert arxiv_search.__name__ == "arxiv_search"
    assert arxiv_fetch.__name__ == "arxiv_fetch"
    assert format_citation.__name__ == "format_citation"
