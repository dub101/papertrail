"""arch_00 — deterministic, no-LLM baseline.

This package holds the floor of the benchmark: one arxiv search, no Claude
calls, results dressed in the BenchmarkResult schema. See ``README.md`` in
this directory for the heuristic-vs-disclaimer policy that decides what goes
into each summary slot.

Public surface: ``BaselineArchitecture`` (the class) and
``BaselineTooFewResultsError`` (raised when arxiv returns < 8 usable papers,
the schema's hard floor on deliverable size).
"""

from __future__ import annotations

from papertrail.architectures.arch_00_baseline.baseline import (
    BaselineArchitecture,
    BaselineTooFewResultsError,
)

__all__ = ["BaselineArchitecture", "BaselineTooFewResultsError"]
