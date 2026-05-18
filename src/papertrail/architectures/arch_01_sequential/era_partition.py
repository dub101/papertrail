"""arch_01 stage 4 — era partition + narrative.

What this module does (in one paragraph):
    Implements ``EraPartitionAgent``, the fourth stage of the arch_01
    pipeline. It takes the 4-12 synthesised papers from stage 3 and
    partitions them into 2-4 content-driven eras, each with a year
    range, a slug, a short name, a concept-anchored narrative
    (~80-120 words), and the list of arxiv_ids in that era. A single
    forced ``tool_use`` call; same machinery as triage and synthesis.
    The cross-field invariants (every paper assigned exactly once, no
    fabricated ids, unique era_ids) are validated post-parse — JSON
    Schema cannot express the input-set membership check.

Why content-driven not date-driven:
    arch_00 already does date-bucket partition; arch_01 is supposed to
    clear that floor by letting the model use intellectual content to
    decide where eras break. Years are descriptive labels for the era
    you've chosen, not the determinant of the boundary. Era date ranges
    may overlap across eras when papers' methodological lineage spans
    chronological boundaries.

Cert mappings:
    - **D4 TS 4.3** (primary) — forced ``tool_use`` with a JSON schema
      generated from ``EraPartition``.
    - **D1 TS 1.6** (primary) — first cross-paper integration pass in
      arch_01 ("analyze each file individually, then run a cross-file
      integration pass" in the cert's framing). Stage 3 was the local
      pass; this is the integration.
    - **D5 TS 5.6** (secondary) — the era narrative must preserve
      information provenance: per-paper claim attribution lives in
      ``paper_ids`` (the audit trail), and the narrative speaks at the
      concept level rather than name-dropping papers.

Libraries: anthropic SDK (raw), pydantic v2 (schema-from-model + regex),
papertrail.pricing (``compute_cost_usd``), papertrail.prompts
(``load_prompt``). Note: arxiv_id regex compliance for paper_ids is
guaranteed transitively — every emitted paper_id must be in the input
set (cross-field validator), and every input id is already regex-valid
upstream, so a separate ``Field(pattern=ARXIV_ID_RE)`` isn't needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Final, cast

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from papertrail.pricing import compute_cost_usd
from papertrail.prompts import load_prompt

if TYPE_CHECKING:
    from anthropic import AsyncAnthropic
    from anthropic.types import (
        Message,
        MessageParam,
        ToolChoiceToolParam,
        ToolParam,
    )

    from papertrail.architectures.arch_01_sequential.synthesis import PaperSynthesis
    from papertrail.tools.arxiv import ArxivPaper


# ───── Bounds constants ─────────────────────────────────────────────────


# Era-count bounds per ADR-0005 / user direction: hard floor 2, hard
# ceiling 4. The floor avoids the degenerate "everything in one era"
# output that loses all of stage 4's value; the ceiling avoids
# over-fragmentation (5 eras over 10 papers reduces signal).
ERAS_MIN: Final[int] = 2
ERAS_MAX: Final[int] = 4

# Narrative character bounds (a soft 80-120 word target, ~500-700 chars
# in English prose). 200 is generous on the low end to permit terse
# eras; 1500 is generous on the high end to permit verbose ones without
# tolerating runaway paragraphs. The first real run on 2026-05-17 hit
# the previous 900 cap with a substantive narrative just slightly over,
# so 1500 absorbs that variance.
_NARRATIVE_MIN_CHARS: Final[int] = 200
_NARRATIVE_MAX_CHARS: Final[int] = 1500

# Year-range bounds. arXiv started in 1991; we allow 1990 as a margin.
# 2100 is a future-proofing ceiling that will only ever bite a typo.
_YEAR_MIN: Final[int] = 1990
_YEAR_MAX: Final[int] = 2100

# era_id slug constraints — must match TimelineEra.era_id at the
# Deliverable level (which is just min_length=1). We tighten here to
# enforce a readable, lower-case slug; assembly translates one-to-one.
_ERA_ID_PATTERN: Final[str] = r"^[a-z0-9_-]+$"
_ERA_ID_MAX_CHARS: Final[int] = 40

# Output budget. 4 eras x ~120-word narratives x ~1.3 tokens/word plus
# the surrounding JSON envelope is ~700 tokens; 2048 is comfortable.
_MAX_TOKENS: Final[int] = 2048


# ───── Pydantic schema the model fills in ───────────────────────────────


class _StrictModel(BaseModel):
    """Local strict base — same shape as ``benchmark._StrictModel``."""

    model_config = ConfigDict(extra="forbid")


class EraEntry(_StrictModel):
    """One era's metadata + the paper_ids assigned to it.

    Year fields are integers rather than ``date`` objects: the user
    cares about year-level granularity, and integers serialise cleanly
    through the tool's JSON Schema. At assembly time (stage 6) these
    integers get lifted into ``date(year, 1, 1)`` and
    ``date(year, 12, 31)`` to satisfy the existing ``TimelineEra``
    schema in ``benchmark.py``.

    ``date_range_end_year`` is nullable on purpose: the "frontier" /
    "ongoing" era often has no defined end, and ``None`` is the honest
    representation. A nullable field also gives the model an explicit
    "no end" exit rather than encouraging it to fabricate a year.
    """

    era_id: str = Field(
        min_length=1,
        max_length=_ERA_ID_MAX_CHARS,
        pattern=_ERA_ID_PATTERN,
    )
    name: str = Field(min_length=1, max_length=80)
    date_range_start_year: int = Field(ge=_YEAR_MIN, le=_YEAR_MAX)
    date_range_end_year: int | None = Field(default=None, ge=_YEAR_MIN, le=_YEAR_MAX)
    narrative: str = Field(min_length=_NARRATIVE_MIN_CHARS, max_length=_NARRATIVE_MAX_CHARS)
    paper_ids: list[str] = Field(min_length=1)


class EraPartition(_StrictModel):
    """The shape the model fills in via the forced ``submit_era_partition`` call.

    Hard bounds on the era count are encoded at the schema level so the
    model gets immediate feedback when it tries to over- or under-emit.
    The cross-field invariants (one paper per era, no fabricated paper
    ids, unique era_ids, year ordering) cannot be expressed here — they
    are enforced post-parse in ``_validate_cross_field``.
    """

    eras: list[EraEntry] = Field(min_length=ERAS_MIN, max_length=ERAS_MAX)


# ───── Exceptions — fail-loud per ADR-0005 ──────────────────────────────


class EraPartitionError(RuntimeError):
    """Base class for any era-partition-side failure."""


class EraPartitionRefusedError(EraPartitionError):
    """Model returned without calling ``submit_era_partition``.

    Carries the response's ``stop_reason`` and a content summary so a
    refusal is debuggable from the exception alone.
    """

    def __init__(self, stop_reason: str, content_summary: str) -> None:
        self.stop_reason = stop_reason
        self.content_summary = content_summary
        super().__init__(
            f"Era partition refused (no tool_use block); "
            f"stop_reason={stop_reason!r}; "
            f"content={content_summary[:200]!r}"
        )


class EraPartitionInvalidOutputError(EraPartitionError):
    """The model called the tool but the output was invalid.

    Covers three classes of failure, distinguished by the message text:
        - Pydantic schema validation (wrong types, regex miss, length
          out of bounds, era count out of [2, 4]).
        - Cross-field violation: a paper assigned to multiple eras, a
          paper missing from all eras, a fabricated paper_id, a
          duplicate era_id, end year before start year.
        - Tool_use input that the SDK couldn't parse into
          ``EraPartition`` at all.

    No retry — ADR-0005 fail-loud stance for stages 2/4/5.
    """


# ───── Return shapes ────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class EraPartitionUsage:
    """Token and cost accounting for the one tool-use call."""

    input_tokens: int
    output_tokens: int
    cost_usd: float


@dataclass(frozen=True, slots=True)
class EraPartitionResult:
    """What stage 4 returns to the pipeline orchestrator.

    ``eras`` is the parsed pydantic list, in model-emitted order. The
    orchestrator decides whether to re-sort by start_year at assembly
    time (stage 6 may want chronological ordering even when the model
    emitted in conceptual order).

    ``papers_by_era`` is a convenience map ``era_id -> tuple[arxiv_id]``
    so downstream code doesn't have to re-walk the era list. Equivalent
    information; pre-computed for ergonomics.
    """

    eras: tuple[EraEntry, ...]
    papers_by_era: dict[str, tuple[str, ...]]
    usage: EraPartitionUsage


# ───── The agent ────────────────────────────────────────────────────────


class EraPartitionAgent:
    """Stage 4 of arch_01: forced ``tool_use`` partition of synthesised papers.

    Wire-up:
        1. Render a user message containing the topic and one block per
           paper carrying the five-field synthesis from stage 3 plus
           ``arxiv_id`` and publication year.
        2. Define one tool ``submit_era_partition`` whose ``input_schema``
           is ``EraPartition.model_json_schema()``.
        3. Call ``client.messages.create`` with
           ``tool_choice={"type":"tool","name":"submit_era_partition"}``
           to force the call.
        4. Validate the tool_use input into ``EraPartition``; run the
           four cross-field invariants; build ``papers_by_era``; return.

    Cert mappings: see module docstring.
    """

    DEFAULT_MODEL: ClassVar[str] = "claude-haiku-4-5"
    PROMPT_NAME: ClassVar[str] = "arch_01_era_partition"
    PROMPT_VERSION: ClassVar[str] = "v1"
    TOOL_NAME: ClassVar[str] = "submit_era_partition"
    TOOL_DESCRIPTION: ClassVar[str] = (
        "Submit the era partition of the synthesised papers. Call this "
        "exactly once, with between 2 and 4 eras and every input "
        "arxiv_id assigned to exactly one era."
    )

    def __init__(
        self,
        client: AsyncAnthropic,
        *,
        model: str | None = None,
    ) -> None:
        """Inject the Anthropic client.

        ``model`` defaults to ``DEFAULT_MODEL`` (Haiku per ADR-0003).
        """
        self._client = client
        self._model = model or self.DEFAULT_MODEL
        self._system_prompt: str | None = None

    @property
    def model(self) -> str:
        """The model id this agent was constructed with."""
        return self._model

    @property
    def system_prompt(self) -> str:
        """Lazy-loaded prompt body. Cached after first read."""
        if self._system_prompt is None:
            self._system_prompt = load_prompt(self.PROMPT_NAME, self.PROMPT_VERSION)
        return self._system_prompt

    async def partition(
        self,
        *,
        topic: str,
        papers: list[ArxivPaper],
        syntheses: list[PaperSynthesis],
    ) -> EraPartitionResult:
        """Partition ``papers`` into 2-4 content-driven eras.

        Args:
            topic: The research topic (paragraph or keyword).
            papers: The chosen papers (post-triage). Used for year and
                arxiv_id; the synthesis comes from the parallel
                ``syntheses`` list.
            syntheses: Stage 3's per-paper synthesis, one entry per
                paper. The lists must be aligned by ``arxiv_id``.

        Raises:
            EraPartitionRefusedError: Model returned no tool_use block.
            EraPartitionInvalidOutputError: Schema or cross-field check
                failed.
        """
        if not papers:
            raise EraPartitionInvalidOutputError(
                "partition() called with no papers; stage 3 should have raised"
            )

        synth_by_id = {s.arxiv_id: s for s in syntheses}
        # Defensive: every paper must have a matching synthesis. A
        # disclaimer synthesis still counts (stage 3 writes one when a
        # paper couldn't be synthesised); only a *missing* entry is a bug.
        missing_synth = [p.arxiv_id for p in papers if p.arxiv_id not in synth_by_id]
        if missing_synth:
            raise EraPartitionInvalidOutputError(
                f"partition() input mismatch: papers without synthesis: {sorted(missing_synth)}"
            )

        user_content = self._render(topic=topic, papers=papers, synth_by_id=synth_by_id)

        tool_definition: ToolParam = {
            "name": self.TOOL_NAME,
            "description": self.TOOL_DESCRIPTION,
            "input_schema": cast("Any", EraPartition.model_json_schema()),
        }
        tool_choice: ToolChoiceToolParam = {"type": "tool", "name": self.TOOL_NAME}
        messages: list[MessageParam] = [{"role": "user", "content": user_content}]

        response: Message = await self._client.messages.create(
            model=self._model,
            max_tokens=_MAX_TOKENS,
            system=self.system_prompt,
            messages=messages,
            tools=[tool_definition],
            tool_choice=tool_choice,
        )

        partition = self._extract_partition(response)

        input_ids = [p.arxiv_id for p in papers]
        self._validate_cross_field(partition=partition, input_ids=input_ids)

        papers_by_era = {era.era_id: tuple(era.paper_ids) for era in partition.eras}

        input_tokens = int(getattr(response.usage, "input_tokens", 0) or 0)
        output_tokens = int(getattr(response.usage, "output_tokens", 0) or 0)
        usage = EraPartitionUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=compute_cost_usd(self._model, input_tokens, output_tokens),
        )

        return EraPartitionResult(
            eras=tuple(partition.eras),
            papers_by_era=papers_by_era,
            usage=usage,
        )

    # ─── Helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _render(
        *,
        topic: str,
        papers: list[ArxivPaper],
        synth_by_id: dict[str, PaperSynthesis],
    ) -> str:
        """Render the user message: topic + per-paper block with full synthesis.

        Per ADR-0005 / user direction, stage 4 receives the full
        five-field synthesis from stage 3 — not the abstract. The
        synthesis is the model's "view" into the paper at this stage.
        """
        lines: list[str] = [
            "# Topic",
            "",
            topic,
            "",
            f"# Papers ({len(papers)})",
            "",
            "Each paper is presented with its arxiv_id, publication year, "
            "and the five-field synthesis produced by the previous stage.",
            "",
        ]
        for paper in papers:
            synth = synth_by_id[paper.arxiv_id]
            lines.extend(
                [
                    f"## arxiv_id={paper.arxiv_id}",
                    f"- year: {paper.published.year}",
                    f"- about: {synth.summary_about}",
                    f"- relation_to_topic: {synth.summary_relation_to_topic}",
                    f"- problem: {synth.summary_problem}",
                    f"- approach: {synth.summary_approach}",
                    f"- impact: {synth.summary_impact}",
                    "",
                ]
            )
        return "\n".join(lines)

    def _extract_partition(self, response: Message) -> EraPartition:
        """Pull the ``submit_era_partition`` tool_use block; validate.

        Raises:
            EraPartitionRefusedError: No matching tool_use block found.
            EraPartitionInvalidOutputError: Tool_use block present but
                its ``input`` failed ``EraPartition`` validation.
        """
        text_summary = ""
        for block in response.content:
            if block.type == "tool_use" and block.name == self.TOOL_NAME:
                try:
                    return EraPartition.model_validate(block.input)
                except ValidationError as e:
                    input_repr = repr(block.input)[:500]
                    raise EraPartitionInvalidOutputError(
                        f"submit_era_partition input failed schema validation. "
                        f"stop_reason={response.stop_reason!r}; "
                        f"tool_use_input={input_repr}; "
                        f"text_blocks={text_summary[:300]!r}; "
                        f"pydantic={e}"
                    ) from e
            if block.type == "text":
                text_summary += block.text

        raise EraPartitionRefusedError(
            stop_reason=response.stop_reason or "unknown",
            content_summary=text_summary,
        )

    @staticmethod
    def _validate_cross_field(*, partition: EraPartition, input_ids: list[str]) -> None:
        """Run the four invariants JSON Schema cannot express.

        Order: fabricated > duplicate-paper > duplicate-era > missing
        > date-range. Fabricated and duplicate-paper are the most
        diagnostic ("model is hallucinating ids" vs "model is
        confused"), so we surface them before the missing check which
        is just "model didn't cover everything."
        """
        input_set = set(input_ids)
        all_assigned: list[str] = []
        for era in partition.eras:
            all_assigned.extend(era.paper_ids)
        assigned_set = set(all_assigned)

        fabricated = assigned_set - input_set
        if fabricated:
            raise EraPartitionInvalidOutputError(
                f"Era partition references arxiv_ids not in the input: {sorted(fabricated)}"
            )

        # A paper appears in multiple eras iff it appears more than once
        # across all paper_ids lists. Option A from the design
        # discussion: each paper assigned to exactly one era.
        if len(all_assigned) != len(assigned_set):
            seen: set[str] = set()
            duplicates: set[str] = set()
            for aid in all_assigned:
                if aid in seen:
                    duplicates.add(aid)
                seen.add(aid)
            raise EraPartitionInvalidOutputError(
                f"Papers assigned to multiple eras: {sorted(duplicates)}. "
                f"Each arxiv_id must appear in exactly one era's paper_ids."
            )

        era_ids = [era.era_id for era in partition.eras]
        if len(era_ids) != len(set(era_ids)):
            era_seen: set[str] = set()
            era_dupes: set[str] = set()
            for eid in era_ids:
                if eid in era_seen:
                    era_dupes.add(eid)
                era_seen.add(eid)
            raise EraPartitionInvalidOutputError(
                f"Duplicate era_id values: {sorted(era_dupes)}. "
                f"Each era_id must be unique within the partition."
            )

        missing = input_set - assigned_set
        if missing:
            raise EraPartitionInvalidOutputError(
                f"{len(missing)} input paper(s) not assigned to any era: {sorted(missing)}"
            )

        for era in partition.eras:
            if (
                era.date_range_end_year is not None
                and era.date_range_end_year < era.date_range_start_year
            ):
                raise EraPartitionInvalidOutputError(
                    f"Era {era.era_id!r}: date_range_end_year "
                    f"({era.date_range_end_year}) is before "
                    f"date_range_start_year ({era.date_range_start_year})."
                )
