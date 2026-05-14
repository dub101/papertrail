"""Shared tool layer used by every architecture (arch_00 through arch_06).

This package holds the three primitive operations all architectures share:
``arxiv_search``, ``arxiv_fetch``, and ``format_citation``. Each is a plain
async (or pure sync) Python function — no Anthropic imports, no tool-use JSON
schemas, no agent loop. The architectures wrap these primitives in their own
orchestration; the MCP server in ``arch_06`` exposes the same primitives as
MCP tools. Same logic, different envelopes.

Cert mapping: foundation for D2 (Tool Design & MCP Integration) — these
primitives become literal D2 territory when wrapped as LLM-callable tools
in arch_01 onward. The narrow-vs-generic design choices baked in here
(three separate tools rather than one ``render_paper(format=...)``) are
the cert-tested principle from TS 2.1.

Public surface — re-exported below. Anything NOT re-exported (constants,
private helpers, the strict-model base) is implementation detail; treat as
unstable.
"""

from __future__ import annotations

from papertrail.tools.arxiv import ArxivPaper, SortBy, arxiv_fetch, arxiv_search
from papertrail.tools.citation import format_citation

__all__ = [
    "ArxivPaper",
    "SortBy",
    "arxiv_fetch",
    "arxiv_search",
    "format_citation",
]
