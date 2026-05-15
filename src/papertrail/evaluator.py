"""Evaluator agent — the LLM-as-judge for BenchmarkResult quality.

What this module does (in one paragraph):
    Defines the schema (``EvaluatorScore`` and friends) that an independent
    Claude instance produces when grading the output of one of our research
    architectures. The Evaluator class itself (added in step 5b) consumes a
    ``BenchmarkResult.deliverable``, sends it to claude-sonnet-4-6 with a
    forced ``tool_use`` call, and validates the model's tool input back into
    this schema. The schema is the contract — once it's pinned, the prompt
    and the orchestration can both change without breaking persistence.

Why the dimensions look like this:
    arch_00 (the deterministic floor) populates *some* summary fields with
    honest heuristics (``summary_about``, ``summary_relation_to_topic``,
    ``overall_summary``) and others with distinct disclaimer constants
    (``summary_problem``, ``summary_approach``, ``summary_impact``,
    ``era.narrative``). For that design to pay off, the Evaluator must
    score each dimension *separately* so the floor scores attribute to the
    specific dimensions that didn't try. A single overall number would
    waste arch_00's deliberate asymmetry.

Cert mapping:
    - **D4 TS 4.6** (primary) — "Design multi-instance and multi-pass review
      architectures". The Evaluator is the canonical example: an independent
      Claude instance without the generator's reasoning context, self-reporting
      confidence alongside each finding for calibrated routing.
    - **D4 TS 4.3** (secondary) — "Enforce structured output using tool use and
      JSON schemas". The verdict shape below is exposed as a tool's
      ``input_schema`` and the model is forced to call that tool, eliminating
      JSON syntax errors as a failure mode.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, ClassVar, Final, cast

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from papertrail.benchmark import Deliverable
from papertrail.pricing import MODEL_PRICING_PER_MTOK, compute_cost_usd
from papertrail.prompts import load_prompt

# Backwards-compatible alias for the historical private name used inside
# this module. ``MODEL_PRICING_PER_MTOK`` is re-exported via ``__all__``
# below so callers that still ``from papertrail.evaluator import ...`` it
# keep working.
_compute_cost_usd = compute_cost_usd

if TYPE_CHECKING:
    from anthropic import AsyncAnthropic
    from anthropic.types import (
        Message,
        MessageParam,
        ToolChoiceToolParam,
        ToolParam,
    )

# Five labelled summary fields on ``PaperEntry``. Re-stated here as a typed
# constant so the SynthesisScores model and the prompt template can both
# reference the same canonical list. If a sixth field is ever added to
# PaperEntry, this constant and ``SynthesisScores`` change together — that
# coupling is the point.
SUMMARY_FIELDS: Final[tuple[str, str, str, str, str]] = (
    "summary_about",
    "summary_relation_to_topic",
    "summary_problem",
    "summary_approach",
    "summary_impact",
)

# Per-model pricing has moved to ``papertrail.pricing`` -- a third caller
# (arch_01 triage stage) appeared, which per the codebase rule triggers
# extraction from "duplicated in two places" to "shared module."
# Re-exported below for any importer that still references the old names.

class _StrictModel(BaseModel):
    """Local strict base — same shape as ``benchmark._StrictModel``.

    Duplicated rather than imported because the benchmark module marks its
    version private (leading underscore). If a third subsystem ever needs
    this shape we promote it to a shared base; until then, duplication is
    cheaper than premature extraction.
    """

    model_config = ConfigDict(extra="forbid")


# ───── The unit ─────────────────────────────────────────────────────────


class DimensionScore(_StrictModel):
    """One graded dimension: a number plus the reason for it.

    ``score`` is bounded to ``[0.0, 1.0]`` so the Evaluator can't accidentally
    emit a scale the harness doesn't understand. ``rationale`` is required and
    non-empty: a bare number with no justification is the failure mode this
    schema exists to prevent. If the evaluator has nothing to say, that's
    information — but it has to *say* it ("no rationale available").
    """

    score: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(min_length=1)


# ───── The per-summary-field rollup ─────────────────────────────────────


class SynthesisScores(_StrictModel):
    """One ``DimensionScore`` per per-paper summary field, aggregated across papers.

    Aggregated, not per-paper. Per-paper scoring would balloon output tokens
    (5 dimensions x 10 papers = 50 nested scores) and the Evaluator-stub
    contract is "fast, structured, useful" — the per-paper granularity can
    come later if calibration shows it matters.

    Named fields rather than a ``dict[str, DimensionScore]`` because the set
    is closed (mirrors PaperEntry exactly) and named fields produce a cleaner
    JSON Schema for ``tool_use`` — Claude sees explicit properties to fill
    in rather than an open-ended object.
    """

    about: DimensionScore
    relation_to_topic: DimensionScore
    problem: DimensionScore
    approach: DimensionScore
    impact: DimensionScore


# ───── What the LLM emits ───────────────────────────────────────────────


class EvaluatorVerdict(_StrictModel):
    """Exactly the shape the LLM fills in via its forced ``tool_use`` call.

    This is split from ``EvaluatorScore`` (below) deliberately: the model only
    knows about the grading content, not who graded or when. Provenance is
    stamped by the ``Evaluator`` class after the model returns. Exposing the
    smaller schema as the tool's ``input_schema`` means Claude isn't asked to
    fabricate its own model name or timestamp — both anti-patterns.

    Why two separate timeline dimensions:
        ``timeline_quality`` grades whether the era partition is structurally
        coherent (sensible number of buckets, sensible boundaries, narrative
        reads). ``timeline_veracity`` grades whether individual papers are
        placed in the correct era. arch_00 will likely pass quality but is
        shaky on veracity (date-bucket heuristic, no semantic check) —
        collapsing them would hide that.

    Why ``overall`` is self-reported, not derived:
        Letting the model weight dimensions holistically with its own
        judgment matches the LLM-as-judge pattern. If we ever see drift, we
        can add a derived field alongside this one — but not before we have
        data showing the self-report is miscalibrated.
    """

    # Top-line summary scores. The model produces both — we don't derive.
    overall: float = Field(
        ge=0.0,
        le=1.0,
        description="Holistic top-line score across all dimensions, model-judged.",
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Evaluator's self-reported certainty in its grades. Low values flag "
            "this verdict for human review per D4 TS 4.6 calibrated routing."
        ),
    )

    # Was the right set of papers chosen for the query?
    selection_relevance: DimensionScore = Field(
        description=(
            "How well the selected 8-12 papers match the topic. Judge each paper's "
            "relevance to the query and aggregate."
        ),
    )

    # Two-axis timeline grading. See class docstring above.
    timeline_quality: DimensionScore = Field(
        description=(
            "Structural coherence of the era partition: sensible bucket count, "
            "sensible date boundaries, narrative reads as a coherent theme."
        ),
    )
    timeline_veracity: DimensionScore = Field(
        description=(
            "Are individual papers placed in the era they actually belong to "
            "(by date AND by thematic fit)?"
        ),
    )

    # Per-field synthesis quality — the asymmetry arch_00 was designed around.
    synthesis: SynthesisScores = Field(
        description=(
            "One score per per-paper summary field (about / relation_to_topic / "
            "problem / approach / impact), aggregated across all papers."
        ),
    )

    # Executive summary — the ~150-200 word top-of-deliverable text.
    executive_summary: DimensionScore = Field(
        description=(
            "Quality of ``deliverable.overall_summary``: does it cohere, is it "
            "calibrated to the papers, is it free of fabrication?"
        ),
    )

    # Cross-dimensional narrative. Forces the model to articulate what's
    # going on across grades, not just emit numbers.
    critique: str = Field(
        min_length=1,
        description=(
            "Free-text critique tying the dimension scores together — what the "
            "deliverable did well, where it fell short, what to fix next."
        ),
    )


# ───── Token usage + cost estimate ──────────────────────────────────────


class EvaluatorUsage(_StrictModel):
    """Tokens (exact) and estimated USD cost for one Evaluator call.

    ``input_tokens`` and ``output_tokens`` come straight from the Anthropic
    ``Message.usage`` response — they are authoritative. ``cost_usd_estimated``
    is derived locally via ``_compute_cost_usd`` and the per-model rate
    table; treat it as a glance-friendly approximation, not a billed figure.
    The ``_estimated`` suffix on the field name keeps that distinction
    legible at every read site.

    DryRunEvaluator stamps all zeros: no real call, no tokens used, no cost.
    """

    input_tokens: int = Field(
        ge=0,
        description="Exact input-token count returned by the Anthropic API.",
    )
    output_tokens: int = Field(
        ge=0,
        description="Exact output-token count returned by the Anthropic API.",
    )
    cost_usd_estimated: float = Field(
        ge=0.0,
        description=(
            "Approximate USD cost computed from input/output tokens and the "
            "local per-model pricing table. Authoritative billing lives on "
            "Anthropic's side — this is a convenience estimate."
        ),
    )


# ───── What the harness persists ────────────────────────────────────────


class EvaluatorScore(_StrictModel):
    """The verdict plus enough provenance to interpret it later.

    Stored alongside the BenchmarkResult under ``benchmark_runs/``. The
    provenance and usage fields mirror the ``BenchmarkResult.provenance``
    pattern: you have to know *what model* graded *with which prompt* *when*
    and *at what cost* to replay a grade, audit drift between evaluator
    versions, or budget future runs.
    """

    verdict: EvaluatorVerdict
    evaluator_model: str = Field(
        min_length=1,
        description="Anthropic model id used as judge, e.g. 'claude-sonnet-4-6'.",
    )
    evaluator_prompt_version: str = Field(
        min_length=1,
        description="Version slug of the grading prompt, e.g. 'v1'.",
    )
    evaluated_at: datetime = Field(
        description="UTC timestamp the verdict was produced (wall clock).",
    )
    usage: EvaluatorUsage = Field(
        description=(
            "Exact token counts and an estimated USD cost for this call. "
            "Zero across the board for DryRunEvaluator."
        ),
    )


# ───── Exceptions ───────────────────────────────────────────────────────


class EvaluatorError(RuntimeError):
    """Base class for Evaluator-side failures.

    Subclasses distinguish the *kind* of failure so callers (the runner,
    tests, future retry policies) can branch on type rather than parse
    messages. Per the no-retry decision, we raise the first time something
    breaks and let the user see the model's stop_reason / validation message
    so they can fix the prompt or the schema. No silent recovery.
    """


class EvaluatorRefusedError(EvaluatorError):
    """The model returned without making the required ``submit_evaluation`` call.

    Happens when Sonnet decides — despite ``tool_choice`` — that it can't or
    won't comply: typical causes are a safety refusal, a malformed schema
    it can't satisfy, or a deliverable so empty there's nothing to grade.
    The ``stop_reason`` and a short content summary are preserved so the
    user can read the model's actual response and decide what to change.
    """

    def __init__(self, stop_reason: str | None, content_summary: str) -> None:
        self.stop_reason = stop_reason
        self.content_summary = content_summary
        super().__init__(
            f"Evaluator did not call submit_evaluation "
            f"(stop_reason={stop_reason!r}); response content: {content_summary[:200]!r}"
        )


class EvaluatorInvalidOutputError(EvaluatorError):
    """The model called the tool but its input failed ``EvaluatorVerdict`` validation.

    ``tool_use`` with a forced ``tool_choice`` and a JSON schema *should*
    eliminate this. When it doesn't, the validation error message is the
    actionable artifact — it tells you which field violated which constraint.
    """

    def __init__(self, validation_message: str) -> None:
        self.validation_message = validation_message
        super().__init__(f"Evaluator tool input failed validation: {validation_message}")


# ───── Deliverable → markdown for the user message ──────────────────────


def _format_deliverable_markdown(deliverable: Deliverable, *, topic: str) -> str:
    """Render the deliverable as markdown for the Evaluator's user message.

    Markdown over JSON for two reasons:

    1. Token economy. Markdown is ~30% lighter than equivalent JSON for the
       same prose-heavy content (no quote/comma/brace overhead). At Sonnet
       prices, that's the difference between ~$0.018 and ~$0.025 per call's
       input side — small per call, real over hundreds of runs.
    2. Comprehension. Sonnet reads research-paper-like markdown more
       naturally than nested JSON; the model is closer to its training
       distribution.

    We deliberately do NOT include ``BenchmarkResult.telemetry`` here. The
    Evaluator should grade the deliverable on its own merits, not be biased
    by token counts, prompt names, or which architecture produced it. This
    is D5 TS 5.1 in spirit — trim verbose tool outputs to only relevant
    fields before they accumulate in downstream context.
    """
    lines: list[str] = [f"# Topic\n\n{topic}\n"]

    lines.append(f"# Executive summary\n\n{deliverable.overall_summary}\n")

    lines.append("# Timeline\n")
    for era in deliverable.timeline:
        # ``date_range_end`` can be None to mean "still current"; surface that
        # explicitly so the model doesn't have to guess what missing means.
        end = era.date_range_end.isoformat() if era.date_range_end else "present"
        lines.append(
            f"## Era `{era.era_id}` — {era.name} "
            f"({era.date_range_start.isoformat()} to {end})"
        )
        lines.append(f"\n{era.narrative}")
        lines.append(f"\n**Papers in era:** {', '.join(era.paper_ids)}\n")

    lines.append(f"# Papers ({len(deliverable.papers)})\n")
    for idx, p in enumerate(deliverable.papers, start=1):
        lines.append(f"## {idx}. {p.title}")
        lines.append(f"\n- **arxiv_id:** {p.arxiv_id}")
        lines.append(f"- **authors:** {', '.join(p.authors)}")
        lines.append(f"- **published:** {p.published_date.isoformat()}")
        lines.append(f"- **era_id:** {p.era_id}")
        # The architecture's self-reported confidence is data the Evaluator
        # uses to spot overconfident or under-confident architectures.
        lines.append(f"- **architecture self-confidence:** {p.confidence:.2f}")
        lines.append(f"\n**summary_about:** {p.summary_about}")
        lines.append(f"\n**summary_relation_to_topic:** {p.summary_relation_to_topic}")
        lines.append(f"\n**summary_problem:** {p.summary_problem}")
        lines.append(f"\n**summary_approach:** {p.summary_approach}")
        lines.append(f"\n**summary_impact:** {p.summary_impact}\n")

    return "\n".join(lines)


# ───── The real evaluator ───────────────────────────────────────────────


class Evaluator:
    """LLM-as-judge for ``BenchmarkResult.deliverable``.

    Wire-up:
        1. Build the user message by rendering the deliverable as markdown.
        2. Define one tool, ``submit_evaluation``, whose ``input_schema`` is
           ``EvaluatorVerdict.model_json_schema()``.
        3. Call ``client.messages.create`` with
           ``tool_choice={"type":"tool","name":"submit_evaluation"}`` to force
           the model to fill that schema.
        4. Pull the ``tool_use`` content block, validate its ``input`` back
           into ``EvaluatorVerdict``, wrap with provenance, return.

    Why no retry on failure:
        Per the step-5 decision: raise the first time the model misbehaves
        and let the user read the error to fix the cause (prompt, schema,
        or deliverable). Retry-with-feedback (D4 TS 4.4) is a real pattern,
        but adding it now would mask which failure mode we actually hit and
        double the cost ceiling unpredictably.

    Cert mappings:
        - **D4 TS 4.3** — forced tool_use with strict JSON schema.
        - **D4 TS 4.6** — independent review instance (Sonnet judging Haiku
          per ADR-0003), self-reported confidence for calibrated routing.
    """

    DEFAULT_MODEL: ClassVar[str] = "claude-sonnet-4-6"
    PROMPT_NAME: ClassVar[str] = "evaluator"
    PROMPT_VERSION: ClassVar[str] = "v1"
    TOOL_NAME: ClassVar[str] = "submit_evaluation"
    TOOL_DESCRIPTION: ClassVar[str] = (
        "Submit the structured evaluation of the research deliverable. "
        "You must call this exactly once with all required fields filled in."
    )
    # 2048 covers ~9 dimensions worth of rationale + a 6-sentence critique
    # with comfortable headroom. If verdicts start getting truncated, this is
    # the knob — but the prompt asks for concise rationales precisely so we
    # don't have to inflate it.
    MAX_TOKENS: ClassVar[int] = 2048

    def __init__(
        self,
        client: AsyncAnthropic,
        *,
        model: str | None = None,
    ) -> None:
        """Inject the Anthropic client.

        ``client`` is injected (not constructed inside) so tests can pass a
        fake without touching the network and so a future runner can share
        one client across multiple evaluators. ``model`` defaults to
        ``DEFAULT_MODEL`` but can be overridden for ad-hoc experiments
        (e.g. prompt iteration on Haiku). The runner is expected to pass
        the default to keep ADR-0003 comparability intact.
        """
        self._client = client
        self._model = model or self.DEFAULT_MODEL
        # System prompt is loaded lazily on first ``evaluate`` so constructing
        # an Evaluator never touches the filesystem.
        self._system_prompt: str | None = None

    @property
    def model(self) -> str:
        """The model id this evaluator was constructed with."""
        return self._model

    @property
    def system_prompt(self) -> str:
        """Lazy-loaded prompt body. Cached after first read."""
        if self._system_prompt is None:
            self._system_prompt = load_prompt(self.PROMPT_NAME, self.PROMPT_VERSION)
        return self._system_prompt

    async def evaluate(
        self,
        deliverable: Deliverable,
        *,
        topic: str,
    ) -> EvaluatorScore:
        """Grade a deliverable and return a fully-populated ``EvaluatorScore``.

        Raises:
            EvaluatorRefusedError: model returned without calling the tool.
            EvaluatorInvalidOutputError: tool input failed schema validation.
        """
        user_content = _format_deliverable_markdown(deliverable, topic=topic)

        # The tool schema is auto-derived from the pydantic model. This is
        # the canonical D4 TS 4.3 pattern: one source of truth for the shape,
        # consumed both by the model (via input_schema) and by our parser
        # (via model_validate). Any future schema change propagates to both.
        tool_definition: ToolParam = {
            "name": self.TOOL_NAME,
            "description": self.TOOL_DESCRIPTION,
            # cast: pydantic's model_json_schema() returns dict[str, Any];
            # the SDK's InputSchema TypedDict has a stricter declared shape
            # but structurally accepts anything that begins with type=object.
            "input_schema": cast("Any", EvaluatorVerdict.model_json_schema()),
        }

        # Typed locals so mypy validates the SDK call against the official
        # overloads. tool_choice forces the call; without it, "auto" would
        # let the model respond with text and skip the tool entirely.
        messages: list[MessageParam] = [{"role": "user", "content": user_content}]
        tool_choice: ToolChoiceToolParam = {"type": "tool", "name": self.TOOL_NAME}

        response = await self._client.messages.create(
            model=self._model,
            max_tokens=self.MAX_TOKENS,
            system=self.system_prompt,
            messages=messages,
            tools=[tool_definition],
            tool_choice=tool_choice,
        )

        verdict = self._extract_verdict(response)

        # Token counts come directly from the API response — these are the
        # authoritative usage numbers. The dollar figure is estimated via
        # the local pricing table and is named accordingly downstream.
        input_tokens = int(getattr(response.usage, "input_tokens", 0) or 0)
        output_tokens = int(getattr(response.usage, "output_tokens", 0) or 0)
        usage = EvaluatorUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd_estimated=_compute_cost_usd(
                model=self._model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            ),
        )

        return EvaluatorScore(
            verdict=verdict,
            evaluator_model=self._model,
            evaluator_prompt_version=self.PROMPT_VERSION,
            evaluated_at=datetime.now(UTC),
            usage=usage,
        )

    def _extract_verdict(self, response: Message) -> EvaluatorVerdict:
        """Pull the tool_use input block out of the response and validate it.

        The response may contain a leading ``TextBlock`` (the model's prose
        before the tool call); that's not an error — we just look past it
        for the ``tool_use`` block with our tool name.
        """
        for block in response.content:
            # Duck-typed access on purpose: anthropic SDK content blocks are
            # distinct pydantic classes (ToolUseBlock, TextBlock, ...), and
            # importing each adds coupling for no gain.
            if getattr(block, "type", None) == "tool_use" and (
                getattr(block, "name", None) == self.TOOL_NAME
            ):
                tool_input = getattr(block, "input", None)
                try:
                    return EvaluatorVerdict.model_validate(tool_input)
                except ValidationError as e:
                    raise EvaluatorInvalidOutputError(str(e)) from e

        # No matching tool_use block. Build a short summary of what we got
        # so the user can read the actual response when debugging.
        first = response.content[0] if response.content else None
        if first is None:
            summary = "<empty content list>"
        else:
            text = getattr(first, "text", None)
            summary = text if text is not None else f"<{type(first).__name__}>"
        raise EvaluatorRefusedError(
            stop_reason=response.stop_reason,
            content_summary=summary,
        )


# ───── The dry-run evaluator ────────────────────────────────────────────


class DryRunEvaluator:
    """No-network evaluator: emits a sentinel verdict for wiring tests.

    Same ``evaluate`` signature as ``Evaluator`` so the runner can swap one
    for the other without branching on type. Score values are intentionally
    unrealistic (``0.123`` everywhere) so a glance at any persisted verdict
    immediately reveals "this didn't come from a real LLM call". Don't read
    meaning into these numbers — they are sentinels, not grades. The
    provenance fields are similarly labeled (``evaluator_model="dry-run-no-llm"``)
    so JSON dumps are self-identifying.
    """

    SENTINEL_SCORE: ClassVar[float] = 0.123
    EVALUATOR_MODEL: ClassVar[str] = "dry-run-no-llm"
    PROMPT_VERSION: ClassVar[str] = "dry-run"

    async def evaluate(
        self,
        deliverable: Deliverable,
        *,
        topic: str,
    ) -> EvaluatorScore:
        """Return a sentinel ``EvaluatorScore`` without calling any model.

        Takes the same arguments as ``Evaluator.evaluate`` so the call site
        is identical. The arguments are inspected (``len`` on papers, topic
        echoed in the rationale) only to ensure a typo at the call site
        raises rather than silently passing.
        """
        rationale = (
            f"DRY RUN — no LLM call was made. Sentinel score for topic "
            f"{topic!r} with {len(deliverable.papers)} papers."
        )
        sentinel = DimensionScore(score=self.SENTINEL_SCORE, rationale=rationale)
        verdict = EvaluatorVerdict(
            overall=self.SENTINEL_SCORE,
            confidence=self.SENTINEL_SCORE,
            selection_relevance=sentinel,
            timeline_quality=sentinel,
            timeline_veracity=sentinel,
            synthesis=SynthesisScores(
                about=sentinel,
                relation_to_topic=sentinel,
                problem=sentinel,
                approach=sentinel,
                impact=sentinel,
            ),
            executive_summary=sentinel,
            critique=rationale,
        )
        # Dry-run: no real call happened, so all usage counts are honestly zero.
        zero_usage = EvaluatorUsage(
            input_tokens=0,
            output_tokens=0,
            cost_usd_estimated=0.0,
        )
        return EvaluatorScore(
            verdict=verdict,
            evaluator_model=self.EVALUATOR_MODEL,
            evaluator_prompt_version=self.PROMPT_VERSION,
            evaluated_at=datetime.now(UTC),
            usage=zero_usage,
        )


__all__ = [
    "MODEL_PRICING_PER_MTOK",
    "SUMMARY_FIELDS",
    "DimensionScore",
    "DryRunEvaluator",
    "Evaluator",
    "EvaluatorError",
    "EvaluatorInvalidOutputError",
    "EvaluatorRefusedError",
    "EvaluatorScore",
    "EvaluatorUsage",
    "EvaluatorVerdict",
    "SynthesisScores",
]
