"""arch_01 stage 3 — per-paper synthesis with parallel batches and one retry.

What this module does (in one paragraph):
    Implements ``SynthesisAgent``, the third stage of the arch_01 pipeline.
    It takes the 4-12 papers that triage selected and produces, for each
    one, the five summary fields that ``PaperEntry`` requires. To keep
    each call below the "lost in the middle" attention dilution threshold,
    papers are split into balanced batches of ~5 and processed
    **concurrently** via ``asyncio.gather``. Any paper a batch misses (or
    any batch that fails entirely) is retried exactly once in a second
    parallel round; papers still missing after retry become disclaimer
    entries with an accompanying ``ErrorRecord(recovered=True)``.

Why batched + parallel + one-retry:
    Batching addresses D5 TS 5.1 (lost in the middle) and D1 TS 1.6
    (task decomposition into focused passes). Wrapping each batch in a
    ``_safe_batch`` coroutine that catches exceptions lets us
    ``asyncio.gather`` all batches without one failure cancelling the
    others — the structured concurrent-failure handling the cert calls
    out implicitly. One retry hits D4 TS 4.4 (retry-with-feedback),
    un-deferring the policy ADR-0005 had left as "revisit on real
    failures." The cert is explicit that retries are bounded: *"retries
    are ineffective when the required information is simply absent from
    the source document."*

Cert mappings:
    - **D1 TS 1.6** (primary) — chunked sequential passes ("analyze each
      file individually, then run a cross-file integration pass" in the
      cert's verbiage); stage 3 is the per-paper local pass.
    - **D4 TS 4.3** (primary) — forced ``tool_use`` per batch with a
      pydantic-generated ``input_schema``.
    - **D4 TS 4.4** (primary, first use in the project) — single retry
      pass on missing papers, with the "retry once, then accept partial"
      bound.
    - **D5 TS 5.1** (secondary) — batching to bound per-call context.
    - **D5 TS 5.3** (secondary) — partial-results recovery: failed
      papers get ``ErrorRecord(recovered=True)`` and the stage proceeds.

Libraries: asyncio, anthropic SDK (raw), pydantic v2,
papertrail.benchmark (``ARXIV_ID_RE``, ``ErrorRecord``),
papertrail.pricing, papertrail.prompts, papertrail.tools.arxiv.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Final, cast

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from papertrail.benchmark import ARXIV_ID_RE, ErrorRecord, SynthesisNote
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

    from papertrail.tools.arxiv import ArxivPaper


# ───── Batching + budget constants ──────────────────────────────────────


# Target papers per batch. ADR-0005 specifies 5. Larger risks "lost in
# the middle" (D5 TS 5.1); smaller multiplies per-call overhead.
TARGET_BATCH_SIZE: Final[int] = 5

# Output budget per batch call. 5 papers * 5 fields * ~80 words * ~1.3
# tokens/word ~= 2600 tokens; 4096 is comfortable headroom.
_MAX_TOKENS: Final[int] = 4096

_FIELD_MAX_CHARS: Final[int] = 800
_NOTES_MAX_CHARS: Final[int] = 500


# ───── Disclaimer constants for papers we couldn't synthesise ───────────


# Per-dimension distinct text so the Evaluator's per-field scoring can
# attribute partial failures to stage 3 rather than collapsing five
# dimensions into one blanket low score (same pattern arch_00 uses).
_SYNTH_FAIL_ABOUT: Final[str] = (
    "Stage 3 synthesis failed for this paper; no 'about' summary available. "
    "See telemetry.errors for the failure mode."
)
_SYNTH_FAIL_RELATION: Final[str] = (
    "Stage 3 synthesis failed for this paper; no topic-relation analysis "
    "available. See telemetry.errors for the failure mode."
)
_SYNTH_FAIL_PROBLEM: Final[str] = (
    "Stage 3 synthesis failed for this paper; no problem analysis available. "
    "See telemetry.errors for the failure mode."
)
_SYNTH_FAIL_APPROACH: Final[str] = (
    "Stage 3 synthesis failed for this paper; no approach analysis available. "
    "See telemetry.errors for the failure mode."
)
_SYNTH_FAIL_IMPACT: Final[str] = (
    "Stage 3 synthesis failed for this paper; no impact analysis available. "
    "See telemetry.errors for the failure mode."
)


# ───── Pydantic schema the model fills in ───────────────────────────────


class _StrictModel(BaseModel):
    """Local strict base — same shape as ``benchmark._StrictModel``."""

    model_config = ConfigDict(extra="forbid")


class PaperSynthesis(_StrictModel):
    """One paper's five-field synthesis, plus a per-paper confidence and
    an optional notes channel.

    Field constraints mirror ``PaperEntry``'s requirements (``min_length=1``)
    so a successful synthesis flows directly into the deliverable. The
    upper bound is generous but bounded — runaway paragraphs are caught
    before they reach JSON persistence.

    ``confidence`` is the model's self-assessment of how well this
    per-paper synthesis captures the contribution given the abstract it
    received. Stage 6 propagates it directly into ``PaperEntry.confidence``
    (replacing the earlier heuristic). Disclaimer-filled syntheses (stage
    3 failed for this paper) carry ``confidence=0.0`` as a visible
    "code-emitted floor case" signal — the model-emitted range is then
    the meaningful one.

    ``notes`` is the user-requested "channel for surprises": optional,
    bounded, telemetry-bound. Not part of ``PaperEntry``; routed to
    ``Telemetry.synthesis_notes`` by the orchestrator.
    """

    arxiv_id: str = Field(pattern=ARXIV_ID_RE)
    summary_about: str = Field(min_length=1, max_length=_FIELD_MAX_CHARS)
    summary_relation_to_topic: str = Field(min_length=1, max_length=_FIELD_MAX_CHARS)
    summary_problem: str = Field(min_length=1, max_length=_FIELD_MAX_CHARS)
    summary_approach: str = Field(min_length=1, max_length=_FIELD_MAX_CHARS)
    summary_impact: str = Field(min_length=1, max_length=_FIELD_MAX_CHARS)
    confidence: float = Field(ge=0.0, le=1.0)
    notes: str | None = Field(default=None, max_length=_NOTES_MAX_CHARS)


class SynthesisBatchOutput(_StrictModel):
    """What the model emits per ``submit_synthesis`` call — one batch's worth."""

    syntheses: list[PaperSynthesis] = Field(min_length=1)


# ───── Exception — fail-loud per ADR-0005 ───────────────────────────────


class SynthesisError(RuntimeError):
    """Stage 3 produced zero successfully-synthesised papers.

    Distinct from per-paper / per-batch failures (those degrade
    gracefully via partial-results recovery). Reaching this error means
    not a single paper could be synthesised after retry — the
    deliverable cannot be assembled.
    """


# ───── Return shapes ────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class SynthesisUsage:
    """Token + cost totals across all batch calls (round 1 + retry round).

    ``batches_attempted`` counts the total number of ``messages.create``
    calls the stage made (round 1 + round 2), so the orchestrator can
    fold it into ``Telemetry.agent_call_count`` without recomputing.
    """

    input_tokens: int
    output_tokens: int
    cost_usd: float
    batches_attempted: int = 0


# ``SynthesisNote`` lives in ``papertrail.benchmark`` so ``Telemetry`` can
# reference it without inverting the layering (benchmark.py is the schema
# layer; this file is an architecture-specific stage). Imported above.


@dataclass(frozen=True, slots=True)
class SynthesisResult:
    """What stage 3 returns to the pipeline orchestrator.

    ``syntheses`` is ordered to match the input papers (i.e. triage's
    selection order). Permanently-missing papers appear with disclaimer
    fields and a matching ``ErrorRecord`` in ``error_records``.
    """

    syntheses: tuple[PaperSynthesis, ...]
    error_records: tuple[ErrorRecord, ...]
    notes: tuple[SynthesisNote, ...]
    usage: SynthesisUsage


# ───── Internal per-batch outcome ───────────────────────────────────────


@dataclass(frozen=True, slots=True)
class _BatchOutcome:
    """Wraps one ``_safe_batch`` result, packaging usage with the verdict.

    Either ``output`` is set (success) or ``error`` is (failure); never
    both. ``input_tokens`` and ``output_tokens`` are reported even on
    failure when the API call did happen — Anthropic charges for failed
    schema validations too.
    """

    output: SynthesisBatchOutput | None
    error: Exception | None
    input_tokens: int
    output_tokens: int


# ───── Batch sizing ─────────────────────────────────────────────────────


def _split_balanced(n: int, target: int = TARGET_BATCH_SIZE) -> list[int]:
    """Return batch sizes that split ``n`` into balanced groups near ``target``.

    Rule: ``ceil(n / target)`` batches, sizes differ by at most 1. So
    n=6 -> (3, 3) not (5, 1); n=11 -> (6, 5); n=4 -> (4,). Encodes the
    "balanced beats fill-to-target-first" preference.
    """
    if n <= 0:
        return []
    if n <= target:
        return [n]
    num_batches = (n + target - 1) // target  # ceiling division
    base = n // num_batches
    remainder = n % num_batches
    # Larger batches come first so n=11 -> (6, 5) not (5, 6).
    return [base + 1] * remainder + [base] * (num_batches - remainder)


def _partition(papers: list[ArxivPaper]) -> list[list[ArxivPaper]]:
    """Apply ``_split_balanced`` to slice the paper list."""
    sizes = _split_balanced(len(papers))
    out: list[list[ArxivPaper]] = []
    cursor = 0
    for size in sizes:
        out.append(papers[cursor : cursor + size])
        cursor += size
    return out


# ───── The agent ────────────────────────────────────────────────────────


class SynthesisAgent:
    """Stage 3 of arch_01: parallel batched synthesis with one retry round.

    Wire-up at a high level:
        1. ``_partition`` splits selected papers into balanced batches.
        2. Each batch is wrapped in ``_safe_batch`` (catches exceptions,
           packages usage with verdict) and ``asyncio.gather``ed.
        3. Walk the per-batch outcomes: successful syntheses go into
           the by-id dict, exceptions go into error_records, and any
           paper not present in any output stays "missing."
        4. If anything is missing, re-partition the missing set and
           gather a second round. Same processing.
        5. Papers still missing after retry get a disclaimer-filled
           ``PaperSynthesis`` plus ``ErrorRecord(step_index=3,
           recovered=True)``.
        6. If zero papers succeeded across both rounds, raise
           ``SynthesisError`` — the run cannot proceed.

    Cert mappings: see module docstring.
    """

    DEFAULT_MODEL: ClassVar[str] = "claude-haiku-4-5"
    PROMPT_NAME: ClassVar[str] = "arch_01_synthesis"
    PROMPT_VERSION: ClassVar[str] = "v1"
    TOOL_NAME: ClassVar[str] = "submit_synthesis"
    TOOL_DESCRIPTION: ClassVar[str] = (
        "Submit the per-paper synthesis for every paper in this batch. "
        "Call this exactly once, with one entry per input arxiv_id."
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

    async def synthesize(
        self,
        *,
        topic: str,
        papers: list[ArxivPaper],
    ) -> SynthesisResult:
        """Synthesise the five summary fields for each paper.

        Args:
            topic: The research topic, repeated in each batch's user
                message for context.
            papers: Papers chosen by triage (4-12 expected). Order is
                preserved in the returned ``syntheses`` tuple.

        Raises:
            SynthesisError: Zero papers were successfully synthesised
                across both rounds.

        Returns:
            ``SynthesisResult`` with one ``PaperSynthesis`` per input
            paper (disclaimer-filled for any that failed after retry),
            the accumulated ``ErrorRecord`` list, any non-null notes,
            and usage totals across all calls.
        """
        if not papers:
            # Defensive: empty paper list is a stage-2 bug; surface
            # without an API call.
            raise SynthesisError("synthesize() called with no papers")

        synthesised: dict[str, PaperSynthesis] = {}
        error_records: list[ErrorRecord] = []
        notes: list[SynthesisNote] = []
        total_in = 0
        total_out = 0

        # ─── Round 1: parallel synthesis over balanced batches ───
        round1_batches = _partition(papers)
        round1_outcomes = await self._gather_batches(topic=topic, batches=round1_batches)
        round1_in, round1_out = self._absorb(
            batches=round1_batches,
            outcomes=round1_outcomes,
            into_synthesised=synthesised,
            into_errors=error_records,
            into_notes=notes,
        )
        total_in += round1_in
        total_out += round1_out
        batches_attempted = len(round1_batches)

        # ─── Round 2: retry only the still-missing papers ───
        all_ids = [p.arxiv_id for p in papers]
        missing_ids = [aid for aid in all_ids if aid not in synthesised]
        if missing_ids:
            by_id = {p.arxiv_id: p for p in papers}
            missing_papers = [by_id[aid] for aid in missing_ids]
            round2_batches = _partition(missing_papers)
            round2_outcomes = await self._gather_batches(topic=topic, batches=round2_batches)
            round2_in, round2_out = self._absorb(
                batches=round2_batches,
                outcomes=round2_outcomes,
                into_synthesised=synthesised,
                into_errors=error_records,
                into_notes=notes,
            )
            total_in += round2_in
            total_out += round2_out
            batches_attempted += len(round2_batches)

        # ─── Disclaimer fill for papers still missing after retry ───
        still_missing = [aid for aid in all_ids if aid not in synthesised]
        for aid in still_missing:
            synthesised[aid] = self._build_disclaimer_synthesis(aid)
            error_records.append(
                ErrorRecord(
                    step_index=3,
                    category="api",
                    message=(
                        f"Stage 3 could not synthesise arxiv_id={aid!r} "
                        f"after one retry; disclaimer entry written."
                    ),
                    recovered=True,
                )
            )

        # ─── Hard fail if literally nothing succeeded ───
        if len(still_missing) == len(papers):
            raise SynthesisError(
                f"Stage 3 synthesised 0 of {len(papers)} papers after one "
                f"retry; the run cannot proceed. See ErrorRecord entries "
                f"for per-paper failure causes."
            )

        # Preserve input order in the returned tuple.
        ordered = tuple(synthesised[p.arxiv_id] for p in papers)

        return SynthesisResult(
            syntheses=ordered,
            error_records=tuple(error_records),
            notes=tuple(notes),
            usage=SynthesisUsage(
                batches_attempted=batches_attempted,
                input_tokens=total_in,
                output_tokens=total_out,
                cost_usd=compute_cost_usd(self._model, total_in, total_out),
            ),
        )

    # ─── Concurrency primitive ──────────────────────────────────────────

    async def _gather_batches(
        self,
        *,
        topic: str,
        batches: list[list[ArxivPaper]],
    ) -> list[_BatchOutcome]:
        """Run all batches concurrently via ``asyncio.gather``.

        Each batch is wrapped in ``_safe_batch``, which catches
        exceptions and packages usage with verdict; this is a cleaner
        contract than ``return_exceptions=True`` because it lets us
        track tokens spent on failed batches too.
        """
        outcomes = await asyncio.gather(*[self._safe_batch(topic=topic, batch=b) for b in batches])
        return list(outcomes)

    async def _safe_batch(
        self,
        *,
        topic: str,
        batch: list[ArxivPaper],
    ) -> _BatchOutcome:
        """One batch call wrapped to never raise.

        Any exception (refusal, schema validation, cross-field, network)
        becomes a ``_BatchOutcome(output=None, error=e)``. The token
        counts are best-effort: if the API call returned before raising
        (e.g. schema validation fails locally), tokens are populated;
        if the call itself raised (e.g. network), tokens default to 0.
        """
        captured_in = 0
        captured_out = 0
        try:
            output, captured_in, captured_out = await self._synthesize_batch(
                topic=topic, batch=batch
            )
        except Exception as e:
            return _BatchOutcome(
                output=None,
                error=e,
                input_tokens=captured_in,
                output_tokens=captured_out,
            )
        return _BatchOutcome(
            output=output,
            error=None,
            input_tokens=captured_in,
            output_tokens=captured_out,
        )

    # ─── Single-batch API call ──────────────────────────────────────────

    async def _synthesize_batch(
        self,
        *,
        topic: str,
        batch: list[ArxivPaper],
    ) -> tuple[SynthesisBatchOutput, int, int]:
        """One forced ``tool_use`` call producing one batch's output.

        Returns ``(output, input_tokens, output_tokens)``. Raises
        ``RuntimeError`` on refusal, schema validation, or cross-field
        violation; ``_safe_batch`` catches and packages those.
        """
        user_content = self._render_batch(topic=topic, batch=batch)

        tool_definition: ToolParam = {
            "name": self.TOOL_NAME,
            "description": self.TOOL_DESCRIPTION,
            "input_schema": cast("Any", SynthesisBatchOutput.model_json_schema()),
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

        in_tokens = int(getattr(response.usage, "input_tokens", 0) or 0)
        out_tokens = int(getattr(response.usage, "output_tokens", 0) or 0)

        for block in response.content:
            if block.type == "tool_use" and block.name == self.TOOL_NAME:
                try:
                    output = SynthesisBatchOutput.model_validate(block.input)
                except ValidationError as e:
                    raise RuntimeError(
                        f"submit_synthesis input failed schema validation: {e}"
                    ) from e
                self._validate_batch_cross_field(output=output, batch=batch)
                return output, in_tokens, out_tokens

        raise RuntimeError(
            f"Synthesis batch returned no tool_use block; stop_reason={response.stop_reason!r}"
        )

    @staticmethod
    def _validate_batch_cross_field(
        *, output: SynthesisBatchOutput, batch: list[ArxivPaper]
    ) -> None:
        """Reject fabricated ids and duplicates within a single batch's output.

        Missing-paper detection is **not** done here — that's the retry
        trigger and is handled at the round-absorption level (papers
        not present in any successful output go to the missing set).
        Raising on missing here would lose those papers to the
        exception path instead of letting round 2 pick them up.
        """
        emitted = [s.arxiv_id for s in output.syntheses]
        input_ids = {p.arxiv_id for p in batch}
        fabricated = set(emitted) - input_ids
        if fabricated:
            raise RuntimeError(
                f"Synthesis batch fabricated arxiv_ids not in the input: {sorted(fabricated)}"
            )
        if len(emitted) != len(set(emitted)):
            raise RuntimeError(
                f"Synthesis batch produced duplicate entries for arxiv_ids: {sorted(emitted)}"
            )

    # ─── Round absorption ───────────────────────────────────────────────

    @staticmethod
    def _absorb(
        *,
        batches: list[list[ArxivPaper]],
        outcomes: list[_BatchOutcome],
        into_synthesised: dict[str, PaperSynthesis],
        into_errors: list[ErrorRecord],
        into_notes: list[SynthesisNote],
    ) -> tuple[int, int]:
        """Walk one round's outcomes; classify, accumulate, return token totals.

        For each (batch, outcome) pair:
            - ``error`` set -> record a batch-level ErrorRecord; the
              batch's papers stay missing.
            - ``output`` set -> record each emitted synthesis by
              arxiv_id (first-write-wins so round 2 doesn't overwrite
              round 1); collect non-null notes.

        Returns ``(total_input_tokens, total_output_tokens)`` for this
        round so the caller can fold them into the agent-level rollup.
        """
        round_in = 0
        round_out = 0
        for batch, outcome in zip(batches, outcomes, strict=True):
            round_in += outcome.input_tokens
            round_out += outcome.output_tokens
            if outcome.error is not None:
                into_errors.append(
                    ErrorRecord(
                        step_index=3,
                        category="api",
                        message=(
                            f"Stage 3 batch of {len(batch)} papers failed: "
                            f"{type(outcome.error).__name__}: {outcome.error}"
                        ),
                        recovered=True,
                    )
                )
                continue
            assert outcome.output is not None  # type narrowing for mypy
            for paper_synth in outcome.output.syntheses:
                # First-write-wins: round 1 successes are not overwritten
                # by round 2 retries (which only target the misses).
                if paper_synth.arxiv_id not in into_synthesised:
                    into_synthesised[paper_synth.arxiv_id] = paper_synth
                    if paper_synth.notes:
                        into_notes.append(
                            SynthesisNote(
                                arxiv_id=paper_synth.arxiv_id,
                                note=paper_synth.notes,
                            )
                        )
        return round_in, round_out

    # ─── Disclaimer construction for permanently-missing papers ─────────

    @staticmethod
    def _build_disclaimer_synthesis(arxiv_id: str) -> PaperSynthesis:
        """Build a per-paper synthesis with distinct disclaimers per dimension.

        Distinct text per field so the Evaluator's per-dimension scoring
        can attribute the floor to "stage 3 failed on this paper" rather
        than collapsing five dimensions into one blanket low score.

        ``confidence=0.0`` is deliberate: the model never wrote a real
        synthesis for this paper, so an "honest" model-emitted confidence
        is unavailable. Setting it to the absolute floor makes the
        disclaimer case loud in downstream tooling — any
        ``PaperEntry.confidence == 0.0`` is a code-emitted floor entry,
        not a model judgement.
        """
        return PaperSynthesis(
            arxiv_id=arxiv_id,
            summary_about=_SYNTH_FAIL_ABOUT,
            summary_relation_to_topic=_SYNTH_FAIL_RELATION,
            summary_problem=_SYNTH_FAIL_PROBLEM,
            summary_approach=_SYNTH_FAIL_APPROACH,
            summary_impact=_SYNTH_FAIL_IMPACT,
            confidence=0.0,
            notes=None,
        )

    # ─── User-message rendering for one batch ───────────────────────────

    @staticmethod
    def _render_batch(*, topic: str, batch: list[ArxivPaper]) -> str:
        """Render the user message: topic + per-paper structured block."""
        lines: list[str] = [
            "# Topic",
            "",
            topic,
            "",
            f"# Papers in this batch ({len(batch)})",
            "",
        ]
        for paper in batch:
            primary_category = paper.categories[0] if paper.categories else "unknown"
            year = paper.published.year
            lines.extend(
                [
                    f"## arxiv_id={paper.arxiv_id}",
                    f"- year: {year}",
                    f"- primary_category: {primary_category}",
                    f"- title: {paper.title}",
                    "- abstract:",
                    "",
                    paper.abstract,
                    "",
                ]
            )
        return "\n".join(lines)
