"""arch_01 — sequential pipeline architecture.

Six-stage pipeline: search → triage → per-paper synthesis → era partition →
executive summary → assembly. See ``docs/decisions/0005-arch-01-design.md``
for the full design rationale and cert mappings.

Implementation is being built stage-by-stage. Currently only stage 1
(``SearchAgent``) is implemented; the orchestrating ``SequentialArchitecture``
class lands once the remaining stages exist.

Cert mappings (full architecture):
    - **D1 TS 1.1** — stage 1 agentic loop with stop_reason inspection
    - **D1 TS 1.6** — overall pipeline as prompt chaining
    - **D4 TS 4.3** — forced tool_use with JSON schemas on every LLM stage
    - **D5 TS 5.1** — structured handoff between stages; compact tool_result
      hand-back inside stage 1
    - **D5 TS 5.3** — partial-results recovery on stage 1 with
      ``ErrorRecord(recovered=True)``; stage-typed exceptions across the pipeline
"""

from __future__ import annotations

from papertrail.architectures.arch_01_sequential.search import (
    MAX_SEARCH_ITERATIONS,
    MIN_PAPERS_TO_PROCEED,
    SearchAgent,
    SearchInsufficientResultsError,
    SearchResult,
    SearchTelemetry,
)

__all__ = [
    "MAX_SEARCH_ITERATIONS",
    "MIN_PAPERS_TO_PROCEED",
    "SearchAgent",
    "SearchInsufficientResultsError",
    "SearchResult",
    "SearchTelemetry",
]
