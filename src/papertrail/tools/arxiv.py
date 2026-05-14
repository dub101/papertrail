"""arXiv tool primitives.

Public surface (this commit adds ``arxiv_search``; ``arxiv_fetch`` follows):

- ``ArxivPaper``    — pydantic model echoing one Atom entry
- ``arxiv_search``  — async free-text search, returns up to ``max_results`` papers
- ``SortBy``        — Literal of accepted sort modes

Design note — why ``ArxivPaper`` and not ``PaperEntry`` from ``benchmark.py``:

  ``PaperEntry`` carries fields that only an LLM (or an LLM-driven pipeline)
  can fill in — ``era_id``, the five labelled summaries, ``confidence``. Those
  are *deliverable* concerns. ``ArxivPaper`` carries only what arXiv itself
  knows about a paper. Keeping the two types distinct enforces the dependency
  arrow ``architectures/ -> tools/``: an architecture *enriches* an
  ``ArxivPaper`` into a ``PaperEntry``, never the reverse, and the tool layer
  has no awareness of LLM-side concepts.

Boundary policy decisions (D2 task statements):

  * **Sort order**: ``sort_by`` parameter; default ``"relevance"`` to match the
    legacy user-facing search behaviour. ``sort_order`` is *not* exposed —
    descending is always the useful direction for both relevance and recency.
  * **Result ceiling**: ``max_results`` is capped at 100. The deliverable
    target is 8-12 papers; 100 gives ~10x headroom for filtering. Anything
    larger is almost always a bug or overfetch.
  * **Retry policy**: exactly one retry, only on HTTP 429, with arXiv's
    documented 3-second wait. No retries on 5xx, no retries on network
    errors, no exponential backoff. This encodes the *arXiv protocol*, not a
    generic resilience policy — resilience belongs at the architecture layer
    (see ADR notes in the README) so different architectures can pick their
    own retry/circuit-breaker behaviour.

Cert mapping: D2 (Tool Design) — narrow signature, honest side effects,
explicit error semantics at the boundary.
"""

from __future__ import annotations

import asyncio
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field, HttpUrl

# ───── Constants ────────────────────────────────────────────────────────

ARXIV_API_BASE = "https://export.arxiv.org/api/query"

# Be a polite arXiv client: identify ourselves and link to the project so
# arXiv admins can reach us if our traffic ever looks abusive. The User-Agent
# string is arXiv's primary mechanism for client identification.
USER_AGENT = "papertrail/0.1 (+https://github.com/dub101/papertrail)"

# arXiv asks clients to wait 3 seconds after a 429. This is *their* number,
# not ours — encoded as a named constant so it's findable from the README.
RATE_LIMIT_WAIT_SECONDS = 3.0

# Whole-request timeout. Generous because arXiv occasionally pauses for
# several seconds under load. A network failure should surface as
# ``httpx.TimeoutException`` and propagate to the architecture.
HTTP_TIMEOUT_SECONDS = 30.0

# Hard upper bound on a single search call. Above this the call is almost
# certainly a bug — the deliverable target is only 8-12 papers.
MAX_RESULTS_CEILING = 100

# Atom XML namespaces. The arXiv-specific namespace is used for
# ``primary_category``; everything else lives in the standard Atom namespace.
_ATOM_NS = "http://www.w3.org/2005/Atom"
_ARXIV_NS = "http://arxiv.org/schemas/atom"


# ``Literal`` so callers get IDE autocomplete and mypy catches typos.
# ``relevance`` is first because it's the default in ``arxiv_search``.
SortBy = Literal["relevance", "lastUpdatedDate", "submittedDate"]


# ───── Pydantic base + model ────────────────────────────────────────────


class _StrictModel(BaseModel):
    """Tool-layer pydantic base with ``extra='forbid'``.

    Deliberately *not* imported from ``papertrail.benchmark`` even though the
    definition is identical. The tool layer must stay free of upward imports
    into LLM-aware modules; a four-line duplication is cheaper than a
    coupling that would have to be unwound later.
    """

    model_config = ConfigDict(extra="forbid")


class ArxivPaper(_StrictModel):
    """The faithful representation of one arXiv entry.

    Fields mirror what the arXiv Atom feed actually returns. We do **not**
    constrain ``arxiv_id`` to the post-2007 numeric scheme here — that policy
    belongs at the deliverable layer (see ``PaperEntry.arxiv_id`` in
    ``benchmark.py``). The tool layer's job is faithful echo; filtering is
    the architecture's call.

    ``published`` and ``updated`` are both kept because they carry distinct
    information: ``published`` is the original submission (used for citation
    year), ``updated`` is the last revision (used to detect whether the
    abstract a downstream summarizer sees is still the latest).
    """

    arxiv_id: str = Field(min_length=1, description="arXiv identifier as returned by the API")
    title: str = Field(min_length=1)

    # ``min_length=1`` because an entry with zero authors is almost certainly
    # a parse failure rather than a legitimate result. Failing loudly here
    # surfaces XML-parsing bugs at the tool boundary instead of letting
    # downstream code render "by  (2023)" cites.
    authors: list[str] = Field(min_length=1)

    # arXiv categories (e.g. "cs.LG", "stat.ML"). May be empty for very old
    # papers that predate the category taxonomy, so no min_length.
    categories: list[str] = Field(default_factory=list)

    abstract: str = Field(min_length=1)

    # The HTML landing page (entry_url) and the PDF URL are both stable
    # arXiv-issued links. Validated as HttpUrl so a malformed link fails at
    # construction rather than at render time.
    entry_url: HttpUrl
    pdf_url: HttpUrl

    published: datetime
    updated: datetime


# ───── Internal helpers ─────────────────────────────────────────────────


def _text(elem: ET.Element | None) -> str:
    """Return whitespace-collapsed text of an optional XML element, or empty.

    arXiv's response often wraps text content with leading whitespace
    (especially ``<summary>``). One whitespace pass at the parser boundary
    means downstream code never has to think about it.
    """
    if elem is None or elem.text is None:
        return ""
    return " ".join(elem.text.split())


def _parse_entry(entry: ET.Element) -> ArxivPaper:
    """Convert one Atom ``<entry>`` into an ``ArxivPaper``.

    Private helper shared by ``arxiv_search`` (loops over many entries) and
    later ``arxiv_fetch`` (asserts exactly one entry). Keeping the XML
    knowledge in one place means an arXiv API change only requires editing
    this function.
    """
    # arXiv's ``<id>`` is a URL like ``http://arxiv.org/abs/1706.03762v5``.
    # The arxiv_id is the bit after ``/abs/``; we keep the version suffix
    # ("v5") because it's part of the canonical identifier the API returns.
    id_url = _text(entry.find(f"{{{_ATOM_NS}}}id"))
    arxiv_id = id_url.rsplit("/abs/", 1)[-1] if "/abs/" in id_url else id_url

    title = _text(entry.find(f"{{{_ATOM_NS}}}title"))
    abstract = _text(entry.find(f"{{{_ATOM_NS}}}summary"))

    # ``<author><name>...</name></author>`` — we want the names, in order.
    authors = [
        _text(name)
        for name in entry.findall(f"{{{_ATOM_NS}}}author/{{{_ATOM_NS}}}name")
        if _text(name)
    ]

    # ``<category term="cs.LG"/>`` — the ``term`` attribute is the category.
    categories = [
        cat.get("term", "") for cat in entry.findall(f"{{{_ATOM_NS}}}category") if cat.get("term")
    ]

    # Two link variants in the response: ``rel="alternate"`` is the HTML page,
    # ``title="pdf"`` is the PDF. If the PDF link is missing we synthesise it
    # from the entry URL (arXiv's PDF route mirrors the abs route).
    entry_url = ""
    pdf_url = ""
    for link in entry.findall(f"{{{_ATOM_NS}}}link"):
        if link.get("rel") == "alternate":
            entry_url = link.get("href", "")
        elif link.get("title") == "pdf":
            pdf_url = link.get("href", "")
    if not pdf_url and entry_url:
        pdf_url = entry_url.replace("/abs/", "/pdf/")

    # ``datetime.fromisoformat`` handles arXiv's ``...Z`` and ``...-05:00``
    # suffixes natively in 3.11+. No need for a third-party date parser.
    published = datetime.fromisoformat(_text(entry.find(f"{{{_ATOM_NS}}}published")))
    updated = datetime.fromisoformat(_text(entry.find(f"{{{_ATOM_NS}}}updated")))

    return ArxivPaper(
        arxiv_id=arxiv_id,
        title=title,
        authors=authors,
        categories=categories,
        abstract=abstract,
        entry_url=HttpUrl(entry_url),
        pdf_url=HttpUrl(pdf_url),
        published=published,
        updated=updated,
    )


async def _get_with_arxiv_retry(
    client: httpx.AsyncClient, params: dict[str, str | int]
) -> httpx.Response:
    """Single GET against the arXiv API with the 429-only retry.

    Encodes arXiv's protocol-level politeness:
      * 429 once  -> wait ``RATE_LIMIT_WAIT_SECONDS``, retry exactly once.
      * 429 twice -> raise ``HTTPStatusError`` (caller decides).
      * Any other 4xx/5xx -> raise immediately, no retry.
      * Network errors -> propagate (``httpx.HTTPError`` family).

    The architecture layer is responsible for generic resilience (retry on
    5xx, circuit breakers, etc). This function only knows arXiv-specific
    rules.
    """
    response = await client.get(ARXIV_API_BASE, params=params)
    if response.status_code == 429:
        # One polite wait, then one more try. If that also returns 429 we
        # fall through to ``raise_for_status`` and the architecture handles
        # the failure on its own terms.
        await asyncio.sleep(RATE_LIMIT_WAIT_SECONDS)
        response = await client.get(ARXIV_API_BASE, params=params)
    response.raise_for_status()
    return response


# ───── Public API ───────────────────────────────────────────────────────


async def arxiv_search(
    query: str,
    max_results: int = 5,
    sort_by: SortBy = "relevance",
) -> list[ArxivPaper]:
    """Free-text search over the arXiv corpus.

    Parameters
    ----------
    query:
        Free-text query string. Passed verbatim to arXiv's ``search_query=all:``
        prefix, so quoting and boolean operators behave as documented at
        https://info.arxiv.org/help/api/user-manual.html#query_details.
    max_results:
        Upper bound on the number of papers returned. Capped at
        ``MAX_RESULTS_CEILING`` (100) — anything higher raises ``ValueError``
        because that's almost always a bug at this scale.
    sort_by:
        Sort axis. ``"relevance"`` mimics a user-facing search bar;
        ``"submittedDate"`` and ``"lastUpdatedDate"`` are useful for
        "state of the art in X" queries. Sort order is always descending.

    Returns
    -------
    list[ArxivPaper]
        Zero or more papers, in the order arXiv returned them. An empty list
        is a legitimate outcome (no matches), not an error.

    Raises
    ------
    ValueError
        If ``max_results`` is outside ``1..MAX_RESULTS_CEILING``.
    httpx.HTTPError
        On any network failure or non-recoverable HTTP error. Architectures
        decide whether to retry; this function does not.
    """
    if not 1 <= max_results <= MAX_RESULTS_CEILING:
        raise ValueError(
            f"max_results must be between 1 and {MAX_RESULTS_CEILING}, got {max_results}"
        )

    params: dict[str, str | int] = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
        "sortBy": sort_by,
        "sortOrder": "descending",
    }

    # One client per call. Connection-pool sharing is a benchmark-runner
    # concern; doing it here would couple the tool to that runner.
    async with httpx.AsyncClient(
        timeout=HTTP_TIMEOUT_SECONDS,
        headers={"User-Agent": USER_AGENT},
    ) as client:
        response = await _get_with_arxiv_retry(client, params)

    root = ET.fromstring(response.text)
    entries = root.findall(f"{{{_ATOM_NS}}}entry")
    return [_parse_entry(entry) for entry in entries]
