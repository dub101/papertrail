"""Shared tool layer used by every architecture (arch_00 through arch_06).

This package holds the three primitive operations all architectures share:
``arxiv_search``, ``arxiv_fetch``, and ``format_citation``. Each is a plain
async (or pure sync) Python function — no Anthropic imports, no tool-use JSON
schemas, no agent loop. The architectures wrap these primitives in their own
orchestration; the MCP server in ``arch_06`` exposes the same primitives as
MCP tools. Same logic, different envelopes.

Cert mapping: this package is the D2 (Tool Design) backbone — narrow,
unambiguous, side-effect-honest function signatures that any architecture
or transport can re-skin without changing semantics.
"""

from __future__ import annotations
