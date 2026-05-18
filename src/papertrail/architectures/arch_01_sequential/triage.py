"""arch_01 stage 2 — triage and select 8-12 papers from the candidate pool.

What this module does (in one paragraph):
    Implements ``TriageAgent``, the second stage of the arch_01 sequential
    pipeline. It takes the ~25-35 candidate papers that stage 1 collected
    and emits a structured selection of 4-12 included papers (target 8-12)
    plus one rejection reason per excluded paper. The selection is made by
    Claude via a single forced ``tool_use`` call — the same pattern the
    Evaluator uses (ADR-0004), reused here per ADR-0005.

Why not an agentic loop here:
    Triage is a single, well-defined decision: read all candidates, return
    one verdict each. An iterative loop adds nothing — the model isn't
    discovering anything, it's judging a fixed input set. Forced
    ``tool_use`` with a strict schema is the right pattern for "structured
    decision over a fixed input."

Cert mappings:
    - **D4 TS 4.3** (primary) — forced ``tool_use`` with a JSON schema
      generated from ``TriageSelection``. Eliminates JSON syntax errors.
    - **D4 TS 4.1** (secondary) — the system prompt anchors on explicit
      categorical criteria (contribution / novelty / step-forward /
      self-contained / influence-signal-not-fabricated-count) rather than
      vague "pick the best" guidance. See ``arch_01_triage_v1.md``.
    - **D5 TS 5.6** (tertiary) — every candidate's verdict and reason flow
      into ``Telemetry.candidates`` as ``CandidateRecord`` entries,
      preserving the audit trail from candidate set to chosen set.

Libraries: anthropic SDK (raw), pydantic v2 (schema-from-model), papertrail
.benchmark (``ARXIV_ID_RE``, ``CandidateRecord``), papertrail.pricing
(``compute_cost_usd``), papertrail.prompts (``load_prompt``),
papertrail.tools.arxiv (``ArxivPaper``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Final, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from papertrail.benchmark import ARXIV_ID_RE, CandidateRecord
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


# ───── Quality gate constants ───────────────────────────────────────────


# Lower bound on ``included`` count below which triage refuses to proceed.
# Matches the ``Deliverable.papers`` schema floor set by ADR-0006. The
# distinction between this floor and the *target* (8-12, lives in the
# prompt) is intentional: target is a soft preference for the model;
# floor is a hard architectural guarantee for the downstream stages.
INCLUDED_MIN: Final[int] = 4

# Upper bound — structural, matches ``Deliverable.papers`` max_length.
INCLUDED_MAX: Final[int] = 12

# Per-call output cap. The agentic loop is nondeterministic — different
# runs can return 25-50+ candidate papers, and the model emits one
# decision per candidate (~80-100 output tokens each). 4096 was hit
# immediately; 8192 was still hit on a second run with a larger pool.
# 16384 gives comfortable headroom for the largest realistic candidate
# pool (~80 decisions). If we keep hitting it, the right fix is capping
# the candidate count at orchestration time, not raising the budget
# further.
_MAX_TOKENS: Final[int] = 16384


# ───── Pydantic schemas the model fills in ──────────────────────────────


class _StrictModel(BaseModel):
    """Triage-local strict base — same shape as ``benchmark._StrictModel``.

    Duplicated rather than imported because the benchmark module marks its
    version private (leading underscore). The duplication is consistent
    with the codebase rule already followed by ``evaluator.py``.
    """

    model_config = ConfigDict(extra="forbid")


def _arxiv_id_root(arxiv_id: str) -> str:
    """Strip the optional ``vN`` version suffix from an arxiv_id.

    The model sometimes emits ``2206.05457v2`` when the input had
    ``2206.05457`` (or vice versa). Same paper, same selection
    significance; the version difference is not a fabrication.
    """
    if "v" in arxiv_id:
        idx = arxiv_id.rfind("v")
        if arxiv_id[idx + 1 :].isdigit():
            return arxiv_id[:idx]
    return arxiv_id


class TriageDecision(_StrictModel):
    """One per-paper verdict.

    ``arxiv_id`` carries the same regex constraint as ``PaperEntry.arxiv_id``
    so a fabricated id (e.g. ``"unknown"``, ``"hep-th/9901001"``) fails at
    pydantic-validate time rather than slipping into ``CandidateRecord``.

    ``reason`` is bounded to 300 characters: long enough for "introduces X,
    foundational" or "near-duplicate of arxiv_id=Y" but short enough that a
    50-paper triage doesn't generate paragraph-long justifications. The
    prompt explicitly asks for triage justifications, not summaries.
    """

    arxiv_id: str = Field(pattern=ARXIV_ID_RE)
    verdict: Literal["included", "rejected"]
    reason: str = Field(min_length=1, max_length=300)


class TriageSelection(_StrictModel):
    """The shape the model fills in via the forced ``submit_triage`` tool call.

    A single ``decisions`` list — one entry per *input* candidate. The
    one-to-one constraint (every input id appears once, no fabricated ids,
    no duplicates) cannot be expressed in JSON Schema because it requires
    knowing the input set; it is enforced post-parse in
    ``TriageAgent._validate_cross_field``. ``min_length=1`` here only
    guards against the degenerate empty-list case.
    """

    decisions: list[TriageDecision] = Field(min_length=1)


# ───── Exceptions — fail-loud per ADR-0005 ──────────────────────────────


class TriageError(RuntimeError):
    """Base class for any triage-side failure."""


class TriageRefusedError(TriageError):
    """Model returned without calling ``submit_triage``.

    Distinguishes "Claude refused" (a safety / policy outcome) from
    "Claude tried but produced invalid output" (a schema / criteria issue).
    Carries the response's ``stop_reason`` and a content summary so the
    failure is debuggable from the exception alone.
    """

    def __init__(self, stop_reason: str, content_summary: str) -> None:
        self.stop_reason = stop_reason
        self.content_summary = content_summary
        super().__init__(
            f"Triage refused (no tool_use block); stop_reason={stop_reason!r}; "
            f"content={content_summary[:200]!r}"
        )


class TriageInvalidOutputError(TriageError):
    """The model called ``submit_triage`` but the output was invalid.

    Covers three classes of failure, distinguished by the message text:
        - Pydantic schema validation (wrong types, regex miss, missing field).
        - Cross-field violation (input id missing from decisions, duplicate
          decision for the same id, fabricated id not in the input set,
          ``included`` count above ``INCLUDED_MAX``).
        - Tool_use input that the SDK couldn't parse into ``TriageSelection``
          at all.

    No automatic retry — per ADR-0005's fail-loud stance. Retry-with-feedback
    (D4 TS 4.4) is deferred consistently with ADR-0004.
    """


class TriageInsufficientQualityError(TriageError):
    """``included`` count fell below ``INCLUDED_MIN`` (the quality floor).

    Distinct from ``TriageInvalidOutputError`` because the model's output
    is *valid* — it just reflects an editorial judgment that fewer than
    ``INCLUDED_MIN`` candidates pass the quality gate. The architecture
    cannot proceed (downstream ``Deliverable.papers`` requires ≥ 4), but
    the failure mode is "not enough quality papers" rather than "model
    misbehaved" — different cause, different fix, different message.
    """

    def __init__(self, topic: str, included: int, total: int) -> None:
        self.topic = topic
        self.included = included
        self.total = total
        super().__init__(
            f"Triage included only {included} of {total} candidates "
            f"(minimum {INCLUDED_MIN}); topic={topic!r}. The model judged "
            f"too few candidates worthy. Consider widening the search."
        )


# ───── Return shape ─────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class TriageUsage:
    """Token and cost accounting for one triage call."""

    input_tokens: int
    output_tokens: int
    cost_usd: float


@dataclass(frozen=True, slots=True)
class TriageResult:
    """What stage 2 returns to the pipeline orchestrator.

    ``selected`` is the chosen papers in the order the model emitted them
    in its ``decisions`` list (filtered to verdict=="included"). The
    pipeline preserves the model's ordering rather than re-sorting; if
    later stages need a different order they re-sort themselves.

    ``candidate_records`` is the full audit trail — one
    ``CandidateRecord`` per input candidate, regardless of verdict — for
    direct insertion into ``Telemetry.candidates``.
    """

    selected: tuple[ArxivPaper, ...]
    candidate_records: tuple[CandidateRecord, ...]
    usage: TriageUsage


# ───── The agent ────────────────────────────────────────────────────────


class TriageAgent:
    """Stage 2 of arch_01: forced ``tool_use`` selection over a fixed candidate set.

    Wire-up:
        1. Render the topic + candidates as a structured markdown user
           message (one block per candidate, full abstract included).
        2. Define one tool, ``submit_triage``, whose ``input_schema`` is
           ``TriageSelection.model_json_schema()``.
        3. Call ``client.messages.create`` with
           ``tool_choice={"type":"tool","name":"submit_triage"}`` to force
           the call.
        4. Pull the tool_use block, validate input into ``TriageSelection``,
           run cross-field validators against the original candidate set,
           apply the ``INCLUDED_MIN`` quality gate, build the
           ``CandidateRecord`` list, return.

    Cert mappings: see module docstring.
    """

    DEFAULT_MODEL: ClassVar[str] = "claude-haiku-4-5"
    PROMPT_NAME: ClassVar[str] = "arch_01_triage"
    PROMPT_VERSION: ClassVar[str] = "v1"
    TOOL_NAME: ClassVar[str] = "submit_triage"
    TOOL_DESCRIPTION: ClassVar[str] = (
        "Submit the triage decisions for every candidate paper. You must "
        "call this exactly once, with one decision per input arxiv_id."
    )

    def __init__(
        self,
        client: AsyncAnthropic,
        *,
        model: str | None = None,
    ) -> None:
        """Inject the Anthropic client.

        ``model`` defaults to ``DEFAULT_MODEL`` (Haiku per ADR-0003) but
        can be overridden for ad-hoc experiments. The pipeline orchestrator
        is expected to pass the default to keep ADR-0003 comparability.
        """
        self._client = client
        self._model = model or self.DEFAULT_MODEL
        # Lazy-loaded so construction never touches the filesystem.
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

    async def triage(
        self,
        *,
        topic: str,
        candidates: list[ArxivPaper],
    ) -> TriageResult:
        """Run triage against ``candidates`` for the given ``topic``.

        Args:
            topic: The research topic (paragraph or keyword) the candidates
                were gathered for. Echoed into the user message verbatim.
            candidates: The deduplicated paper set from stage 1. Order is
                preserved in the user message; the model's output may
                reorder.

        Raises:
            TriageRefusedError: Model returned with no ``tool_use`` block.
            TriageInvalidOutputError: ``tool_use`` block was present but
                input failed schema validation or cross-field validation
                (missing id, duplicate id, fabricated id, count > MAX).
            TriageInsufficientQualityError: Output was valid but ``included``
                count is below ``INCLUDED_MIN``.
        """
        if not candidates:
            # Defensive: an empty candidate list is a stage-1 bug (it should
            # have raised SearchInsufficientResultsError instead). Surface
            # immediately rather than calling the API for nothing.
            raise TriageInvalidOutputError(
                "triage() called with no candidates; stage 1 should have raised"
            )

        user_content = self._render_candidates(topic=topic, candidates=candidates)

        tool_definition: ToolParam = {
            "name": self.TOOL_NAME,
            "description": self.TOOL_DESCRIPTION,
            # cast: pydantic's model_json_schema() returns dict[str, Any];
            # the SDK's InputSchema TypedDict is structural — same trick
            # the Evaluator uses.
            "input_schema": cast("Any", TriageSelection.model_json_schema()),
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

        selection = self._extract_selection(response)

        input_ids = [p.arxiv_id for p in candidates]
        self._validate_cross_field(selection=selection, input_ids=input_ids)

        included_ids = [d.arxiv_id for d in selection.decisions if d.verdict == "included"]
        if len(included_ids) < INCLUDED_MIN:
            raise TriageInsufficientQualityError(
                topic=topic, included=len(included_ids), total=len(candidates)
            )

        # Build the chosen-paper list in the model's emitted order.
        by_id: dict[str, ArxivPaper] = {p.arxiv_id: p for p in candidates}
        title_by_id: dict[str, str] = {p.arxiv_id: p.title for p in candidates}
        selected = tuple(by_id[aid] for aid in included_ids)

        candidate_records = tuple(
            CandidateRecord(
                arxiv_id=d.arxiv_id,
                title=title_by_id[d.arxiv_id],
                verdict=d.verdict,
                reason=d.reason,
            )
            for d in selection.decisions
        )

        input_tokens = int(getattr(response.usage, "input_tokens", 0) or 0)
        output_tokens = int(getattr(response.usage, "output_tokens", 0) or 0)
        usage = TriageUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=compute_cost_usd(self._model, input_tokens, output_tokens),
        )

        return TriageResult(
            selected=selected,
            candidate_records=candidate_records,
            usage=usage,
        )

    # ─── Helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _render_candidates(*, topic: str, candidates: list[ArxivPaper]) -> str:
        """Render the user message: topic at top, then one block per candidate.

        Per-candidate block carries arxiv_id, year, primary_category, title,
        and **full abstract**. The stage-1 first-sentence-only compaction is
        appropriate for an iterative loop where token cost accumulates; here
        the model gets one shot to judge each paper, so the full abstract
        is essential signal (D4 TS 4.1: precision requires sufficient info).
        """
        lines: list[str] = [
            "# Topic",
            "",
            topic,
            "",
            f"# Candidates ({len(candidates)} papers)",
            "",
        ]
        for paper in candidates:
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

    def _extract_selection(self, response: Message) -> TriageSelection:
        """Pull the ``submit_triage`` tool_use block; validate into the schema.

        Raises:
            TriageRefusedError: No matching tool_use block was found
                (model returned text only, or called a different tool).
            TriageInvalidOutputError: Tool_use block was present but its
                ``input`` failed ``TriageSelection`` validation. The
                error carries ``stop_reason``, the raw tool input, and
                any text content from the response so a degraded
                response (e.g. truncation at max_tokens) is diagnosable
                from the exception alone, not from process logs.
        """
        text_summary = ""
        for block in response.content:
            if block.type == "tool_use" and block.name == self.TOOL_NAME:
                try:
                    return TriageSelection.model_validate(block.input)
                except ValidationError as e:
                    input_repr = repr(block.input)[:500]
                    raise TriageInvalidOutputError(
                        f"submit_triage input failed schema validation. "
                        f"stop_reason={response.stop_reason!r}; "
                        f"tool_use_input={input_repr}; "
                        f"text_blocks={text_summary[:300]!r}; "
                        f"pydantic={e}"
                    ) from e
            if block.type == "text":
                # Capture text for the refusal error message — debugging
                # a refusal is much easier when the model's own words are
                # in the exception.
                text_summary += block.text

        raise TriageRefusedError(
            stop_reason=response.stop_reason or "unknown",
            content_summary=text_summary,
        )

    @staticmethod
    def _validate_cross_field(
        *,
        selection: TriageSelection,
        input_ids: list[str],
    ) -> None:
        """Enforce one-decision-per-input-id and count ceilings.

        These three checks cannot be expressed in JSON Schema (they need
        the input set), so they run here after pydantic validation. Each
        failure raises ``TriageInvalidOutputError`` with a precise message
        so the operator can tell *which* invariant the model violated.

        Order matters: check fabricated ids before missing ids, because a
        fabricated id is a stronger signal of model misbehavior than a
        missing one (which could just be an under-eager truncation).

        arxiv version suffixes (``v1``, ``v2``, ...) are normalised before
        comparison — the model occasionally adds or strips a version even
        though we render exactly what arxiv returned. The same paper at
        a different version is the same paper for selection purposes.
        """
        decision_ids = [d.arxiv_id for d in selection.decisions]
        # Normalise both sides by stripping the optional vN suffix so
        # "2206.05457" and "2206.05457v2" compare equal.
        input_root_set = {_arxiv_id_root(a) for a in input_ids}
        decision_roots = [_arxiv_id_root(a) for a in decision_ids]
        decision_root_set = set(decision_roots)

        fabricated = decision_root_set - input_root_set
        if fabricated:
            raise TriageInvalidOutputError(
                f"Triage produced decisions for arxiv_ids not in the input: {sorted(fabricated)}"
            )

        if len(decision_roots) != len(decision_root_set):
            seen: set[str] = set()
            duplicates: set[str] = set()
            for aid in decision_roots:
                if aid in seen:
                    duplicates.add(aid)
                seen.add(aid)
            raise TriageInvalidOutputError(
                f"Triage produced duplicate decisions for arxiv_ids: {sorted(duplicates)}"
            )

        missing = input_root_set - decision_root_set
        if missing:
            raise TriageInvalidOutputError(
                f"Triage missed {len(missing)} input candidate(s): {sorted(missing)}"
            )

        included_count = sum(1 for d in selection.decisions if d.verdict == "included")
        if included_count > INCLUDED_MAX:
            raise TriageInvalidOutputError(
                f"Triage included {included_count} papers; ceiling is "
                f"{INCLUDED_MAX} ({INCLUDED_MAX} is the Deliverable.papers "
                f"max_length)."
            )
