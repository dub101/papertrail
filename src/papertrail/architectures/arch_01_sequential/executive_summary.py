"""arch_01 stage 5 — executive summary.

What this module does (in one paragraph):
    Implements ``ExecutiveSummaryAgent``, the fifth and final LLM stage
    of the arch_01 pipeline. It produces the user-facing prose that
    opens the deliverable: a ~150-200 word executive summary plus a
    self-reported confidence score. Single forced ``tool_use`` call,
    same machinery as triage / synthesis / era partition. The model
    sees the topic, the era partition (high-level structure), and the
    per-paper syntheses (supporting detail) — in that order — so the
    aggregated input follows the "key findings summaries at the
    beginning" pattern for combating attention dilution.

Why the eras-first / syntheses-second ordering:
    Stage 5 has the broadest input of any agent in the pipeline (~5000
    tokens when 12 papers are in scope). The cert's D5 TS 5.1 guidance
    for combating "lost in the middle" is: place key findings
    summaries at the beginning of aggregated inputs and organize
    detailed results with explicit section headers. The eras are the
    key findings; the per-paper syntheses are the detail. Inverting
    that ordering would force the model to derive the structure from
    the bottom up at every call.

Cert mappings:
    - **D4 TS 4.3** (primary) — forced ``tool_use`` with a JSON schema
      generated from ``ExecutiveSummary``.
    - **D1 TS 1.6** (primary) — top-level cross-paper integration; the
      final compression pass over the pipeline's output.
    - **D5 TS 5.1** (secondary) — eras-first / syntheses-grouped-by-era
      message ordering to mitigate position effects on the largest
      input in the pipeline.
    - **D5 TS 5.6** (secondary) — the ``confidence`` field gives
      downstream tooling a calibration signal for routing human review.

Libraries: anthropic SDK (raw), pydantic v2, papertrail.pricing
(``compute_cost_usd``), papertrail.prompts (``load_prompt``).
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

    from papertrail.architectures.arch_01_sequential.era_partition import EraEntry
    from papertrail.architectures.arch_01_sequential.synthesis import PaperSynthesis


# ───── Bounds constants ─────────────────────────────────────────────────


# Summary character bounds. 150-200 words at ~5-6 chars/word + spacing
# is roughly 750-1200 chars; 600-1500 gives generous headroom on both
# sides so the model isn't rejected for landing slightly outside the
# target word count. The soft 150-200 word target lives in the prompt;
# the schema bounds are the structural guard.
_SUMMARY_MIN_CHARS: Final[int] = 600
_SUMMARY_MAX_CHARS: Final[int] = 1500

# Output budget. ~1500 chars of summary plus a confidence float plus
# the tool_use envelope is ~600 tokens; 2048 is comfortable headroom.
_MAX_TOKENS: Final[int] = 2048


# ───── Pydantic schema the model fills in ───────────────────────────────


class _StrictModel(BaseModel):
    """Local strict base — same shape as ``benchmark._StrictModel``."""

    model_config = ConfigDict(extra="forbid")


class ExecutiveSummary(_StrictModel):
    """The shape the model fills in via the forced ``submit_executive_summary`` call.

    Two fields by design:
        - ``summary``: the user-facing prose that becomes
          ``Deliverable.overall_summary`` at assembly time. Bounded at
          the schema level to prevent stub or runaway outputs; the
          actual word-count target lives in the prompt.
        - ``confidence``: the model's self-assessment of how well the
          summary captures the topic's evolution given the inputs.
          Pure float, same shape as ``EvaluatorVerdict.confidence`` and
          ``PaperEntry.confidence`` (no rationale subfield).

    No cross-field validation — there is no input-set membership check
    at this stage.
    """

    summary: str = Field(min_length=_SUMMARY_MIN_CHARS, max_length=_SUMMARY_MAX_CHARS)
    confidence: float = Field(ge=0.0, le=1.0)


# ───── Exceptions — fail-loud per ADR-0005 ──────────────────────────────


class ExecutiveSummaryError(RuntimeError):
    """Base class for any executive-summary-side failure."""


class ExecutiveSummaryRefusedError(ExecutiveSummaryError):
    """Model returned without calling ``submit_executive_summary``.

    Carries the response's ``stop_reason`` and a content summary so a
    refusal is debuggable from the exception alone.
    """

    def __init__(self, stop_reason: str, content_summary: str) -> None:
        self.stop_reason = stop_reason
        self.content_summary = content_summary
        super().__init__(
            f"Executive summary refused (no tool_use block); "
            f"stop_reason={stop_reason!r}; "
            f"content={content_summary[:200]!r}"
        )


class ExecutiveSummaryInvalidOutputError(ExecutiveSummaryError):
    """The model called the tool but the output failed validation.

    Covers schema-level failures (summary too short, summary too long,
    confidence out of `[0.0, 1.0]`, missing required field). No retry
    per ADR-0005's fail-loud stance for stages 2/4/5.
    """


# ───── Return shapes ────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class ExecutiveSummaryUsage:
    """Token and cost accounting for the one tool-use call."""

    input_tokens: int
    output_tokens: int
    cost_usd: float


@dataclass(frozen=True, slots=True)
class ExecutiveSummaryResult:
    """What stage 5 returns to the pipeline orchestrator.

    The orchestrator drops ``summary`` straight into
    ``Deliverable.overall_summary`` and records ``confidence`` on the
    deliverable-level telemetry (or on a future
    ``Deliverable.executive_summary_confidence`` field — open at
    time of writing).
    """

    summary: str
    confidence: float
    usage: ExecutiveSummaryUsage


# ───── The agent ────────────────────────────────────────────────────────


class ExecutiveSummaryAgent:
    """Stage 5 of arch_01: forced ``tool_use`` producing the top-level summary.

    Wire-up:
        1. Defensive: validate non-empty inputs and that every
           synthesis's arxiv_id appears in at least one era's
           paper_ids list (alignment check across stage 3 and stage 4).
        2. Render the user message: topic first, then the era
           partition (high-level structure), then the per-paper
           syntheses grouped by era. This is the D5 TS 5.1 mitigation
           against position effects on large inputs.
        3. Define one tool ``submit_executive_summary`` whose
           ``input_schema`` is ``ExecutiveSummary.model_json_schema()``.
        4. Call ``client.messages.create`` with
           ``tool_choice={"type":"tool","name":"submit_executive_summary"}``
           to force the call.
        5. Validate the tool_use input into ``ExecutiveSummary``; build
           the result; return.

    Cert mappings: see module docstring.
    """

    DEFAULT_MODEL: ClassVar[str] = "claude-haiku-4-5"
    PROMPT_NAME: ClassVar[str] = "arch_01_executive_summary"
    PROMPT_VERSION: ClassVar[str] = "v1"
    TOOL_NAME: ClassVar[str] = "submit_executive_summary"
    TOOL_DESCRIPTION: ClassVar[str] = (
        "Submit the executive summary of the topic's evolution plus a "
        "self-assessed confidence score. Call this exactly once."
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

    async def summarize(
        self,
        *,
        topic: str,
        eras: list[EraEntry],
        syntheses: list[PaperSynthesis],
    ) -> ExecutiveSummaryResult:
        """Produce the executive summary + confidence.

        Args:
            topic: The research topic the pipeline ran against.
            eras: The era partition from stage 4 (2-4 EraEntry objects).
            syntheses: The per-paper synthesis from stage 3, one entry
                per chosen paper.

        Raises:
            ExecutiveSummaryError: Empty inputs or alignment mismatch
                between syntheses and eras (a stage-3 / stage-4 bug).
            ExecutiveSummaryRefusedError: Model returned no tool_use block.
            ExecutiveSummaryInvalidOutputError: Schema validation failed.
        """
        if not eras:
            raise ExecutiveSummaryError(
                "summarize() called with no eras; stage 4 should have raised"
            )
        if not syntheses:
            raise ExecutiveSummaryError(
                "summarize() called with no syntheses; stage 3 should have raised"
            )

        # Build an arxiv_id -> era_id index from the era partition; used
        # both for alignment checking and for rendering the per-paper
        # block "grouped by era" in the user message.
        era_id_by_paper: dict[str, str] = {}
        for era in eras:
            for paper_id in era.paper_ids:
                era_id_by_paper[paper_id] = era.era_id

        orphan = [s.arxiv_id for s in syntheses if s.arxiv_id not in era_id_by_paper]
        if orphan:
            raise ExecutiveSummaryError(
                f"summarize() alignment mismatch: syntheses without an era: "
                f"{sorted(orphan)} — stage 4's cross-field validators should "
                f"have caught this."
            )

        user_content = self._render(
            topic=topic,
            eras=eras,
            syntheses=syntheses,
            era_id_by_paper=era_id_by_paper,
        )

        tool_definition: ToolParam = {
            "name": self.TOOL_NAME,
            "description": self.TOOL_DESCRIPTION,
            "input_schema": cast("Any", ExecutiveSummary.model_json_schema()),
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

        summary_output = self._extract_summary(response)

        input_tokens = int(getattr(response.usage, "input_tokens", 0) or 0)
        output_tokens = int(getattr(response.usage, "output_tokens", 0) or 0)
        usage = ExecutiveSummaryUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=compute_cost_usd(self._model, input_tokens, output_tokens),
        )

        return ExecutiveSummaryResult(
            summary=summary_output.summary,
            confidence=summary_output.confidence,
            usage=usage,
        )

    # ─── Helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _render(
        *,
        topic: str,
        eras: list[EraEntry],
        syntheses: list[PaperSynthesis],
        era_id_by_paper: dict[str, str],
    ) -> str:
        """Render the user message in the eras-first / syntheses-grouped ordering.

        The structure is deliberately:
            # Topic
            # Era partition (high-level)
              ## Era 1: ...
              ## Era 2: ...
            # Per-paper syntheses (supporting detail, grouped by era)
              ## arxiv_id=... — era: <era_id>
                - 5 synthesis fields
              ## arxiv_id=... — era: <era_id>

        Per-paper blocks are grouped by their era assignment so the
        model can trace each paper back to its place in the high-level
        structure without re-deriving the membership.
        """
        synth_by_id = {s.arxiv_id: s for s in syntheses}

        lines: list[str] = [
            "# Topic",
            "",
            topic,
            "",
            f"# Era partition ({len(eras)} eras — high-level structure)",
            "",
        ]
        for era in eras:
            end_year_repr = (
                str(era.date_range_end_year)
                if era.date_range_end_year is not None
                else "ongoing"
            )
            lines.extend(
                [
                    f"## Era: {era.era_id} — {era.name}",
                    f"- year range: {era.date_range_start_year}-{end_year_repr}",
                    f"- papers: {', '.join(era.paper_ids)}",
                    "- narrative:",
                    "",
                    era.narrative,
                    "",
                ]
            )

        lines.extend(
            [
                "# Per-paper syntheses (supporting detail, grouped by era)",
                "",
            ]
        )
        # Walk eras in their emitted order so the per-paper blocks
        # appear in the same era ordering the high-level section uses.
        for era in eras:
            for paper_id in era.paper_ids:
                # An era can reference paper_ids whose synthesis is
                # disclaimer-only (stage 3 fail). Render whatever we have.
                synth = synth_by_id.get(paper_id)
                if synth is None:
                    # This case is screened out by the orphan check
                    # earlier; defensive fallthrough.
                    continue
                lines.extend(
                    [
                        f"## arxiv_id={paper_id} — era: {era_id_by_paper[paper_id]}",
                        f"- about: {synth.summary_about}",
                        f"- relation_to_topic: {synth.summary_relation_to_topic}",
                        f"- problem: {synth.summary_problem}",
                        f"- approach: {synth.summary_approach}",
                        f"- impact: {synth.summary_impact}",
                        "",
                    ]
                )

        return "\n".join(lines)

    def _extract_summary(self, response: Message) -> ExecutiveSummary:
        """Pull the ``submit_executive_summary`` tool_use block; validate.

        Raises:
            ExecutiveSummaryRefusedError: No matching tool_use block found.
            ExecutiveSummaryInvalidOutputError: Tool_use block present
                but its ``input`` failed ``ExecutiveSummary`` validation.
        """
        text_summary = ""
        for block in response.content:
            if block.type == "tool_use" and block.name == self.TOOL_NAME:
                try:
                    return ExecutiveSummary.model_validate(block.input)
                except ValidationError as e:
                    raise ExecutiveSummaryInvalidOutputError(
                        f"submit_executive_summary input failed schema "
                        f"validation: {e}"
                    ) from e
            if block.type == "text":
                text_summary += block.text

        raise ExecutiveSummaryRefusedError(
            stop_reason=response.stop_reason or "unknown",
            content_summary=text_summary,
        )
