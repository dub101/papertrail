"""arXiv tool primitives: the ``ArxivPaper`` data shape and (later) the
``arxiv_search`` / ``arxiv_fetch`` async functions that produce it.

This commit lands the data shape only. The two async functions arrive in the
next two commits so each can be reviewed in isolation.

Design note — why ``ArxivPaper`` and not ``PaperEntry`` from ``benchmark.py``:

  ``PaperEntry`` carries fields that only an LLM (or an LLM-driven pipeline)
  can fill in — ``era_id``, the five labelled summaries, ``confidence``. Those
  are *deliverable* concerns. ``ArxivPaper`` carries only what arXiv itself
  knows about a paper. Keeping the two types distinct enforces the dependency
  arrow ``architectures/ -> tools/``: an architecture *enriches* an
  ``ArxivPaper`` into a ``PaperEntry``, never the reverse, and the tool layer
  has no awareness of LLM-side concepts.

Cert mapping: D2 (Tool Design) — the model is the tool's output contract.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, HttpUrl


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
