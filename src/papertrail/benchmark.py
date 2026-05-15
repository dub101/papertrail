"""Benchmark schema for PaperTrail — the unified output contract.

Every architecture (arch_00 through arch_06) returns a ``BenchmarkResult`` from
its ``run()`` method. Because the shape is identical across architectures, the
benchmark runner is generic and the Evaluator Agent reads the same fields
regardless of which orchestration pattern produced them.

The schema is split into three top-level sections:

- ``deliverable``  what a human reads (papers, timeline, summaries)
- ``telemetry``    how the run went (tokens, cost, the trace, candidates, errors)
- ``provenance``   identity and reproducibility metadata

Cert mapping: this module is the D4 (Structured Outputs) backbone of the
project — every agent's terminal output is validated against fields defined
here, and ``model_json_schema()`` can re-emit these as Anthropic tool schemas
when an architecture needs structured tool output.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    model_validator,
)

# arXiv identifier regex for the post-2007 numeric scheme (e.g. "1706.03762").
# Older identifiers like "hep-th/9901001" are intentionally excluded — modern
# papers are the target. If a foundational pre-2007 paper ever needs inclusion,
# relax this pattern then rather than now (YAGNI).
ARXIV_ID_RE = r"^\d{4}\.\d{4,5}(v\d+)?$"

# Current semver of the BenchmarkResult schema itself. Bump on any breaking
# shape change so historical results stay interpretable.
SCHEMA_VERSION = "1.0.0"


class _StrictModel(BaseModel):
    """Project-wide pydantic base.

    ``extra="forbid"`` makes a typo in calling code (e.g. ``confidance=`` for
    ``confidence=``) fail loudly at construction time instead of being
    silently dropped. This is the discipline Anthropic recommends for
    structured tool output too — a misnamed field is almost always a bug.
    """

    model_config = ConfigDict(extra="forbid")


# ───── Run-config (used by Telemetry) ───────────────────────────────────


class Modes(_StrictModel):
    """Architecture-internal knobs for a single run.

    Distinct from harness-level toggles like ``--no-evaluator`` and ``--mock``:
    those decide what *wraps* the architecture; ``Modes`` decides how the
    architecture *behaves*. Keeping them separate means the architecture's
    contract stays clean and ablation flags stay typed.
    """

    # Tools on/off lets us run the same architecture as a pure-prompt pipeline
    # and as a tool-using pipeline, then compare BenchmarkResults to isolate
    # what tools actually contribute (D2 ablation pattern).
    tools: bool = True

    # The deliverable must contain 8-12 papers (per the project brief). Some
    # topics have more landmark papers than others; the band is intentional.
    target_paper_count: int = Field(default=10, ge=8, le=12)


# ───── Deliverable side ─────────────────────────────────────────────────


class PaperEntry(_StrictModel):
    """A single paper in the deliverable, with the five labelled summaries.

    The five summary fields are typed as separate string fields rather than
    one ``summary: dict`` so pydantic validates each independently and
    downstream consumers (markdown formatter, Evaluator) can address them by
    name. This is the D4 discipline: narrow, named fields beat free-form text.
    """

    arxiv_id: str = Field(pattern=ARXIV_ID_RE, description="arXiv identifier, e.g. '1706.03762'")
    title: str = Field(min_length=1)
    authors: list[str] = Field(min_length=1)
    published_date: date
    url: HttpUrl

    # External citation lookup is best-effort. ``None`` means the lookup
    # failed or was skipped (e.g. ``tools=off``). The two fields move
    # together — knowing the count without knowing the source is useless
    # for provenance, so a validator enforces the pair invariant below.
    citation_count: int | None = Field(default=None, ge=0)
    citation_source: Literal["semantic_scholar", "openalex"] | None = None

    # Foreign key into Deliverable.timeline[*].era_id. The PaperEntry itself
    # can't know which eras exist; the cross-list integrity check runs at the
    # Deliverable level where both lists are visible.
    era_id: str = Field(min_length=1)

    # The five labelled summaries - each ~50-75 words ("2-3 lines" in the
    # spec). We deliberately do NOT word-count-validate here: going outside
    # the target is a quality issue the Evaluator should see and penalize,
    # not a crash that hides the failure.
    summary_about: str = Field(min_length=1)
    summary_relation_to_topic: str = Field(min_length=1)
    summary_problem: str = Field(min_length=1)
    summary_approach: str = Field(min_length=1)
    summary_impact: str = Field(min_length=1)

    # Architecture's self-assessed confidence that this paper belongs.
    # Lets the Evaluator spot overconfident architectures (all 1.0s) or
    # waffly ones (all 0.5s).
    confidence: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _citation_count_and_source_paired(self) -> PaperEntry:
        """``citation_count`` and ``citation_source`` must both be set or both ``None``.

        Knowing the count without the source means we can't audit it later
        (which API returned this number?), and knowing the source with no
        count is a contradiction. So they move as a pair.
        """
        if (self.citation_count is None) != (self.citation_source is None):
            raise ValueError("citation_count and citation_source must both be set or both None")
        return self


class TimelineEra(_StrictModel):
    """A grouping of papers within a phase of the topic's development.

    Eras may overlap in time. Parallel workstreams — e.g. RoPE and ALiBi as
    contemporaneous extensions of positional embeddings — belong in distinct
    eras even though they were developed simultaneously. Eras are about
    *coherent themes*, not strict chronological slices.
    """

    era_id: str = Field(min_length=1, description="Slug, e.g. 'foundations'")
    name: str = Field(min_length=1, description="Human-readable era name")
    date_range_start: date
    # ``None`` = the era is still considered current (frontier work ongoing).
    date_range_end: date | None = None
    narrative: str = Field(min_length=1, description="~80-120 words on this era")
    paper_ids: list[str] = Field(min_length=1, description="arxiv_ids of papers in this era")

    @model_validator(mode="after")
    def _end_after_start(self) -> TimelineEra:
        """``date_range_end``, if present, must be on or after ``date_range_start``."""
        if self.date_range_end is not None and self.date_range_end < self.date_range_start:
            raise ValueError("date_range_end must be on or after date_range_start")
        return self


class Deliverable(_StrictModel):
    """What a human reads: papers + a timeline of eras + an executive summary."""

    topic: str = Field(min_length=1)
    overall_summary: str = Field(min_length=1, description="~150-200 word executive summary")
    # 8-12 papers per the project brief; pydantic enforces the band.
    # Lower bound loosened from 8 to 4 by ADR-0006. The 4 is the smallest
    # count at which a meaningful era partition + per-paper synthesis +
    # executive summary can still be produced. arch_00 keeps its own strict
    # 8-floor via BaselineTooFewResultsError; it is no longer the schema's
    # contract.
    papers: list[PaperEntry] = Field(min_length=4, max_length=12)
    timeline: list[TimelineEra] = Field(min_length=1)

    @model_validator(mode="after")
    def _era_paper_consistency(self) -> Deliverable:
        """Cross-list integrity between ``papers`` and ``timeline``.

        Four invariants enforced:

        1. Every ``paper.era_id`` must point at a real era.
        2. Every ``era.paper_ids`` entry must point at a real paper.
        3. No paper may appear in more than one era's ``paper_ids``.
        4. Every paper must be claimed by some era (no orphans).

        These four together mean the era↔paper relationship is a strict
        partition of the papers list. Reading the timeline is a complete
        tour of the deliverable.
        """
        era_ids = {era.era_id for era in self.timeline}
        paper_ids = {p.arxiv_id for p in self.papers}

        # (1) Each paper's era_id must resolve.
        for p in self.papers:
            if p.era_id not in era_ids:
                raise ValueError(f"Paper {p.arxiv_id} references unknown era_id '{p.era_id}'")

        # (2) + (3): walk eras, build a paper→era map, fail on duplicates.
        seen: dict[str, str] = {}
        for era in self.timeline:
            for pid in era.paper_ids:
                if pid not in paper_ids:
                    raise ValueError(f"Era '{era.era_id}' references unknown paper_id '{pid}'")
                if pid in seen:
                    raise ValueError(
                        f"Paper {pid} appears in multiple eras: '{seen[pid]}' and '{era.era_id}'"
                    )
                seen[pid] = era.era_id

        # (4) Every paper must have been claimed by some era.
        missing = paper_ids - seen.keys()
        if missing:
            raise ValueError(f"Papers not assigned to any era: {sorted(missing)}")

        return self


# ───── Telemetry side ───────────────────────────────────────────────────


class ToolCallRecord(_StrictModel):
    """A single tool invocation made during an agent step."""

    tool_name: str = Field(min_length=1)
    # tool_input is structured but heterogeneous across tools, so we leave it
    # as a free-form dict. Each individual tool is responsible for typing its
    # own arguments at the call site.
    tool_input: dict[str, Any]
    tool_output_summary: str
    duration_ms: int = Field(ge=0)
    error: str | None = None


class TraceStep(_StrictModel):
    """One agent invocation.

    The list of TraceSteps on ``Telemetry`` is the spine of prompt debugging.
    When the deliverable looks wrong, you read the trace and ask "which step
    produced this and what prompt drove it?". That's why ``prompt_name`` and
    ``prompt_version`` are explicit fields — you can locate the exact prompt
    file to edit (D5 observability).
    """

    step_index: int = Field(ge=0)
    agent_name: str = Field(min_length=1)
    prompt_name: str = Field(min_length=1)
    prompt_version: str = Field(min_length=1)
    model: str = Field(min_length=1)
    started_at: datetime
    duration_ms: int = Field(ge=0)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    cost_usd: float = Field(ge=0.0)
    # Compact summaries only — full prompts/responses go to disk via the trace
    # logger to keep BenchmarkResult JSON readable. The disk path can be added
    # to ``decision`` or a future field if we need linking.
    input_summary: str
    output_summary: str
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    # Free-form: "selected 11 papers from 47 candidates", "routed to synthesizer", etc.
    decision: str | None = None


class ErrorRecord(_StrictModel):
    """Something went wrong during a run.

    Distinct from ``UnresolvedLookup``: an error is a *bug or fault*
    (API rejected the request, validator crashed). An unresolved lookup is
    a *legitimate absence* (no results for that query). Confusing the two
    leads to noisy error dashboards.
    """

    step_index: int | None = None
    category: Literal["api", "tool", "validation", "schema", "unknown"]
    message: str = Field(min_length=1)
    # Did the architecture keep running after this? If True, the result is
    # still trustworthy but degraded; if False, treat the run as a failure.
    recovered: bool


class CandidateRecord(_StrictModel):
    """A paper the architecture considered, with the verdict and why.

    For verdict='included', the full write-up lives in ``Deliverable.papers``,
    cross-referenced by ``arxiv_id``. This record is the lightweight
    breadcrumb that lets us answer "what got dropped and why?" later.
    """

    arxiv_id: str = Field(pattern=ARXIV_ID_RE)
    title: str = Field(min_length=1)
    verdict: Literal["included", "rejected"]
    reason: str = Field(min_length=1)


class UnresolvedLookup(_StrictModel):
    """A search or fetch that came back empty or soft-failed.

    Reads like an error but isn't one. "We asked arxiv for papers on X and
    got zero results" is the world being unhelpful, not the code being
    broken — handle it visibly but not alarmingly.
    """

    kind: Literal["arxiv_search", "arxiv_fetch", "citation_lookup"]
    query: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    attempted_at: datetime


class Telemetry(_StrictModel):
    """How the run went. The Evaluator and human debuggers read this side."""

    # Architecture identity — both name and version, because the wiring of a
    # given architecture changes more often than its name.
    architecture_name: str = Field(min_length=1)
    architecture_version: str = Field(min_length=1)

    # Maps prompt_name → version string. Lets us reproduce a run by checking
    # out the exact prompt files used.
    prompt_versions: dict[str, str]
    modes: Modes

    started_at: datetime
    finished_at: datetime
    duration_seconds: float = Field(ge=0.0)

    total_input_tokens: int = Field(ge=0)
    total_output_tokens: int = Field(ge=0)
    total_cost_usd: float = Field(ge=0.0)

    agent_call_count: int = Field(ge=0)
    tool_call_count: int = Field(ge=0)

    # Debugging surfaces. ``default_factory=list`` so an architecture can
    # produce a valid Telemetry even with zero events of a given kind.
    trace: list[TraceStep] = Field(default_factory=list)
    errors: list[ErrorRecord] = Field(default_factory=list)
    candidates: list[CandidateRecord] = Field(default_factory=list)
    unresolved_lookups: list[UnresolvedLookup] = Field(default_factory=list)

    @model_validator(mode="after")
    def _finished_after_started(self) -> Telemetry:
        """``finished_at`` must be on or after ``started_at``."""
        if self.finished_at < self.started_at:
            raise ValueError("finished_at must be on or after started_at")
        return self


# ───── Provenance ───────────────────────────────────────────────────────


class Provenance(_StrictModel):
    """Identity & reproducibility metadata.

    Three version axes on purpose:

    - ``schema_version``        the shape of BenchmarkResult itself
    - ``papertrail_version``    the package release
    - ``git_sha``               the exact commit that produced this run

    Old results stay interpretable because we can replay or migrate them
    if any of the three change.
    """

    run_id: UUID = Field(default_factory=uuid4)
    schema_version: str = Field(min_length=1)
    papertrail_version: str = Field(min_length=1)
    git_sha: str = Field(min_length=1)
    created_at: datetime


# ───── Top-level ────────────────────────────────────────────────────────


class BenchmarkResult(_StrictModel):
    """The unified output every architecture's ``run()`` returns.

    Three sections by deliberate design:

    - ``deliverable``  the research output (what a user reads)
    - ``telemetry``    run-time observability (what we read to debug)
    - ``provenance``   identity and reproducibility metadata

    Grouping over flat layout because: (i) it's easier to spot which fields
    are user-facing vs internal, (ii) the Evaluator can score the deliverable
    in isolation while logging the telemetry, (iii) we can serialize just the
    deliverable for human reports without leaking internal token counts.
    """

    deliverable: Deliverable
    telemetry: Telemetry
    provenance: Provenance
