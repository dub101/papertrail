"""arch_01 stage 1 — agentic search loop over ``arxiv_search``.

What this module does (in one paragraph):
    Implements ``SearchAgent``, the first stage of the arch_01 sequential
    pipeline. It accepts a topic (paragraph or keyword), gives Claude one
    tool (``arxiv_search``), and lets the model issue as many search calls
    as it judges necessary to assemble ~25-35 unique candidate papers for
    downstream triage. The loop terminates primarily on
    ``stop_reason == "end_turn"`` — i.e. the model's own coverage judgment;
    a hard ``MAX_SEARCH_ITERATIONS`` cap is a safety circuit, not the
    intended exit. Below 4 unique papers a ``SearchInsufficientResultsError``
    fires (matches the ``Deliverable.papers`` schema floor set by ADR-0006).

Why an agentic loop here (and only here):
    The user's input is a paragraph that names several distinct sub-topics.
    Programmatic decomposition into keywords would be brittle; pre-tokenizing
    into one query forfeits coverage. The model is genuinely better at
    decomposing the paragraph, picking diverse queries, and judging when
    coverage is adequate. Every later stage in arch_01 is a single forced
    ``tool_use`` call — the loop is deliberate and isolated to stage 1.

Cert mappings:
    - **D1 TS 1.1** (primary) — agentic loop with ``stop_reason`` inspection
      as the primary stopping mechanism. The iteration cap is a circuit
      breaker per the TS 1.1 anti-pattern guidance.
    - **D5 TS 5.1** (secondary) — compact ``tool_result`` hand-back: only
      arxiv_id / title / first sentence / year / primary_category per paper,
      plus running unique-paper and duplicate counters. ~10x reduction in
      per-iteration token cost vs handing back the full ``ArxivPaper``.
    - **D5 TS 5.3** (secondary) — partial-results recovery: if an unexpected
      ``stop_reason`` fires after ≥4 papers were accumulated, the loop
      records an ``ErrorRecord(recovered=True)`` and returns; below 4 it
      raises ``SearchInsufficientResultsError``.

Libraries: anthropic SDK (raw), papertrail.tools.arxiv (async search),
papertrail.benchmark (``ErrorRecord``), papertrail.prompts (``load_prompt``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Final, cast

from papertrail.benchmark import ErrorRecord
from papertrail.prompts import load_prompt
from papertrail.tools.arxiv import ArxivPaper, arxiv_search

if TYPE_CHECKING:
    from anthropic import AsyncAnthropic
    from anthropic.types import Message, MessageParam, ToolParam


# ───── Loop and cost constants ──────────────────────────────────────────


# A circuit breaker, not the primary stop. The TS 1.1 anti-pattern is using
# "an arbitrary iteration cap as the primary stopping mechanism" — so the
# intended exit is always ``stop_reason == "end_turn"``. This cap only
# bounds blast radius if the model gets stuck issuing repetitive queries.
# Sized at 8: enough headroom for a paragraph topic that legitimately
# decomposes into 5-6 angles plus a couple of follow-ups, well short of a
# pathological budget burn.
MAX_SEARCH_ITERATIONS: Final[int] = 8

# Matches the ``Deliverable.papers`` schema floor set by ADR-0006. Below
# this, the architecture cannot produce a valid deliverable downstream, so
# stage 1 fails fast with ``SearchInsufficientResultsError`` rather than
# letting stages 2-5 burn tokens on a doomed run.
MIN_PAPERS_TO_PROCEED: Final[int] = 4

# Per-turn output cap. The model's text turns are short (queries plus brief
# reasoning); 2048 is generous headroom and matches what the Evaluator uses.
_MAX_TOKENS_PER_TURN: Final[int] = 2048

# Pricing dict is duplicated from ``papertrail.evaluator`` deliberately —
# two callers today (evaluator + this module). The codebase rule (see
# evaluator.py docstring on ``_StrictModel`` duplication) is: extract to a
# shared module once a third caller appears. Arch_01 stages 2-5 will be
# that third caller — at that point ``MODEL_PRICING_PER_MTOK`` and
# ``_compute_cost_usd`` move to ``papertrail/pricing.py``. Until then,
# duplication is cheaper than premature extraction.
#
# Tuple shape: ``(input_per_mtok_usd, output_per_mtok_usd)``.
# Verified 2026-05-14. Bump the date in the comment when updating rates.
_MODEL_PRICING_PER_MTOK: Final[dict[str, tuple[float, float]]] = {
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
    "claude-haiku-4-5-20251001": (1.0, 5.0),
}


def _compute_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate USD cost from token counts using ``_MODEL_PRICING_PER_MTOK``.

    Unknown models fall back to ``0.0`` — a visibly-wrong zero against a real
    model is a louder "table is stale, fix it" signal than a silent miss.
    """
    pricing = _MODEL_PRICING_PER_MTOK.get(model)
    if pricing is None:
        return 0.0
    input_rate, output_rate = pricing
    return (
        (input_tokens / 1_000_000) * input_rate
        + (output_tokens / 1_000_000) * output_rate
    )


# ───── Tool definition exposed to the model ─────────────────────────────


# The model-facing ``max_results`` ceiling is intentionally tighter than the
# underlying tool's ``MAX_RESULTS_CEILING`` of 100. 20 is enough for the
# model to batch when it's confident about an angle while preventing
# "burn 100 results x 8 iterations of token cost" pathologies. The hard
# minimum of 1 prevents the model from issuing zero-result calls; the
# default of 10 nudges toward the comfortable middle.
_ARXIV_SEARCH_TOOL_DEF: Final[dict[str, Any]] = {
    "name": "arxiv_search",
    "description": (
        "Search arXiv for papers matching a free-text query. Returns up to "
        "``max_results`` papers, filtered to remove duplicates of papers "
        "already returned in this session. Use multiple calls with different "
        "queries to cover distinct sub-topics. Stop calling once new queries "
        "are returning mostly already-seen papers."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "The arXiv full-text query string. Vary terminology "
                    "between calls — 'self-attention', 'attention mechanism', "
                    "and 'transformer' are not equivalent to arXiv search."
                ),
                "minLength": 1,
            },
            "max_results": {
                "type": "integer",
                "description": "Number of papers to request (1-20).",
                "minimum": 1,
                "maximum": 20,
                "default": 10,
            },
        },
        "required": ["query"],
        "additionalProperties": False,
    },
}


# ───── Public exceptions ────────────────────────────────────────────────


class SearchInsufficientResultsError(RuntimeError):
    """Stage 1 ended with fewer than ``MIN_PAPERS_TO_PROCEED`` unique papers.

    Carries the count, the topic, and the final ``stop_reason`` so callers
    (and humans reading logs) can tell apart "model gave up cleanly with too
    few results" (``end_turn``) from "cap or refusal hit too early"
    (``max_tokens`` / ``refusal`` / etc.).
    """

    def __init__(self, topic: str, found: int, stop_reason: str) -> None:
        self.topic = topic
        self.found = found
        self.stop_reason = stop_reason
        super().__init__(
            f"arch_01 search ended with {found} unique papers "
            f"(minimum {MIN_PAPERS_TO_PROCEED}); stop_reason={stop_reason!r}; "
            f"topic={topic!r}"
        )


# ───── Return shape ─────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class SearchTelemetry:
    """Stage-local accounting that the orchestrator folds into ``Telemetry``.

    All counters are observed, not predicted: tokens come straight from the
    Anthropic API's ``Message.usage`` field; ``cost_usd`` is derived via the
    pricing table. ``final_stop_reason`` is exposed so the orchestrator can
    log it without re-parsing.
    """

    iterations_used: int
    queries_issued: tuple[str, ...]
    input_tokens: int
    output_tokens: int
    cost_usd: float
    final_stop_reason: str
    # True iff stage 1 exited on a non-``end_turn`` stop_reason but still met
    # the ``MIN_PAPERS_TO_PROCEED`` floor. False on clean end_turn and on
    # any case that raises ``SearchInsufficientResultsError``.
    recovered: bool


@dataclass(frozen=True, slots=True)
class SearchResult:
    """What stage 1 returns to the pipeline orchestrator.

    ``papers`` is deduplicated by ``arxiv_id`` in discovery order — i.e. the
    order the model first saw each paper across all of its queries.
    ``error_records`` carries at most one record today (the partial-results
    case); the plural shape matches ``Telemetry.errors`` for trivial
    concatenation at the orchestrator level.
    """

    papers: tuple[ArxivPaper, ...]
    telemetry: SearchTelemetry
    error_records: tuple[ErrorRecord, ...]


# ───── The agent ────────────────────────────────────────────────────────


class SearchAgent:
    """Stage 1 of arch_01: agentic search loop over ``arxiv_search``.

    Wire-up:
        1. Build a single ``user`` message containing the topic.
        2. Loop: call ``client.messages.create`` with the agent's system
           prompt, one tool, and the running message history.
        3. On each response, append the assistant content and inspect
           ``stop_reason``:
              - ``end_turn`` → primary exit, leave the loop.
              - ``tool_use`` → execute every ``arxiv_search`` block, append
                a compact ``tool_result`` for each, loop.
              - anything else → break with the partial-results decision.
        4. After the loop, decide: enough papers to recover, or raise.

    Why client injection: lets tests pass a mocked ``AsyncAnthropic`` and
    lets the future pipeline orchestrator share one client across all
    stages of one run.

    Cert mappings: see module docstring.
    """

    DEFAULT_MODEL: ClassVar[str] = "claude-haiku-4-5"
    PROMPT_NAME: ClassVar[str] = "arch_01_search"
    PROMPT_VERSION: ClassVar[str] = "v1"

    def __init__(
        self,
        client: AsyncAnthropic,
        *,
        model: str | None = None,
        max_iterations: int = MAX_SEARCH_ITERATIONS,
    ) -> None:
        """Inject the Anthropic client.

        ``model`` defaults to ``DEFAULT_MODEL`` (Haiku per ADR-0003) and is
        overridable for ad-hoc experiments. ``max_iterations`` is overridable
        primarily so tests can force the cap to fire by setting it to 1.
        """
        self._client = client
        self._model = model or self.DEFAULT_MODEL
        self._max_iterations = max_iterations
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

    async def search(self, topic: str) -> SearchResult:
        """Run the agentic search loop for ``topic``.

        Returns:
            A populated ``SearchResult`` with the deduplicated papers,
            telemetry, and any error records (partial-results recovery only).

        Raises:
            SearchInsufficientResultsError: fewer than ``MIN_PAPERS_TO_PROCEED``
                unique papers were collected, regardless of how the loop ended.
        """
        # Dict, not list, so dedup-on-write is O(1) and insertion order is
        # preserved (Python 3.7+ language guarantee). Reading ``.values()``
        # at the end gives papers in the order the model first encountered
        # them across all its queries.
        unique_papers: dict[str, ArxivPaper] = {}
        queries_issued: list[str] = []
        messages: list[MessageParam] = [{"role": "user", "content": topic}]

        input_tokens = 0
        output_tokens = 0
        iterations = 0
        final_stop_reason = "uninitialized"

        tools: list[ToolParam] = [cast("ToolParam", _ARXIV_SEARCH_TOOL_DEF)]

        while iterations < self._max_iterations:
            iterations += 1

            response: Message = await self._client.messages.create(
                model=self._model,
                max_tokens=_MAX_TOKENS_PER_TURN,
                system=self.system_prompt,
                tools=tools,
                messages=messages,
            )

            # Token counts are exact (from the API), accumulated across the
            # whole loop. Cost is derived once at the end from the totals.
            input_tokens += int(getattr(response.usage, "input_tokens", 0) or 0)
            output_tokens += int(getattr(response.usage, "output_tokens", 0) or 0)
            final_stop_reason = response.stop_reason or "unknown"

            # Append the full assistant turn — including any text blocks and
            # all tool_use blocks — so the next iteration sees the model's
            # own reasoning trail. The SDK accepts content blocks directly.
            messages.append(
                {"role": "assistant", "content": cast("Any", response.content)}
            )

            if final_stop_reason == "end_turn":
                # Primary exit. The model decided coverage is adequate.
                break

            if final_stop_reason == "tool_use":
                # Process every tool_use block in this response. The API can
                # emit more than one tool_use in a single turn; we honor
                # them all before looping (and only one round-trip per turn,
                # which is the SDK's contract).
                tool_result_blocks: list[dict[str, Any]] = []
                for block in response.content:
                    if block.type != "tool_use":
                        continue
                    if block.name != _ARXIV_SEARCH_TOOL_DEF["name"]:
                        # Defensive: only one tool exists. Anything else is
                        # a schema violation and should surface immediately.
                        raise RuntimeError(
                            f"SearchAgent received unexpected tool call: "
                            f"{block.name!r}"
                        )
                    tool_input = cast("dict[str, Any]", block.input)
                    query = str(tool_input.get("query", ""))
                    max_results = int(tool_input.get("max_results", 10))
                    queries_issued.append(query)

                    papers = await arxiv_search(
                        query=query, max_results=max_results
                    )
                    new_papers, dup_count = _fold_into_unique(papers, unique_papers)
                    tool_result_blocks.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": _format_compact_result(
                                query=query,
                                new_papers=new_papers,
                                dup_count=dup_count,
                                total_unique=len(unique_papers),
                            ),
                        }
                    )

                messages.append(
                    {"role": "user", "content": cast("Any", tool_result_blocks)}
                )
                continue

            # Anything else (``max_tokens``, ``pause_turn``, ``refusal``, ...)
            # is an unexpected stop. Drop out and let the post-loop block
            # decide whether to recover or raise.
            break

        # ─── Post-loop: recover, raise, or return clean ──────────────────

        n_papers = len(unique_papers)

        if n_papers < MIN_PAPERS_TO_PROCEED:
            # Below the schema floor — no path forward, regardless of why we
            # exited. Includes the case where the model emitted ``end_turn``
            # cleanly but found too few papers: the schema would reject the
            # final deliverable anyway, so fail at stage 1 with a clearer
            # message than a deferred ``pydantic.ValidationError``.
            raise SearchInsufficientResultsError(
                topic=topic, found=n_papers, stop_reason=final_stop_reason
            )

        recovered = False
        error_records: list[ErrorRecord] = []
        if final_stop_reason != "end_turn":
            # We have enough papers, but the model didn't choose to stop —
            # the cap fired, or refusal, or max_tokens. This is the D5 TS 5.3
            # partial-results recovery branch: surface the degradation as a
            # structured ``ErrorRecord(recovered=True)`` rather than swallowing
            # it (silent suppression) or treating it as fatal (over-strict).
            recovered = True
            error_records.append(
                ErrorRecord(
                    step_index=1,
                    category="api",
                    message=(
                        f"Stage 1 exited stop_reason={final_stop_reason!r} "
                        f"after {iterations} iterations; recovered with "
                        f"{n_papers} unique papers."
                    ),
                    recovered=True,
                )
            )

        return SearchResult(
            papers=tuple(unique_papers.values()),
            telemetry=SearchTelemetry(
                iterations_used=iterations,
                queries_issued=tuple(queries_issued),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=_compute_cost_usd(self._model, input_tokens, output_tokens),
                final_stop_reason=final_stop_reason,
                recovered=recovered,
            ),
            error_records=tuple(error_records),
        )


# ───── Helpers ──────────────────────────────────────────────────────────


def _fold_into_unique(
    papers: list[ArxivPaper], unique: dict[str, ArxivPaper]
) -> tuple[list[ArxivPaper], int]:
    """Add ``papers`` to ``unique`` (mutating it), return (new_papers, dup_count).

    The mutation is intentional: ``unique`` is the running session state and
    the caller wants both the side effect and the per-call new/dup split.
    Splitting them into two functions would re-iterate the same list twice.
    """
    new_papers: list[ArxivPaper] = []
    dup_count = 0
    for paper in papers:
        if paper.arxiv_id in unique:
            dup_count += 1
            continue
        unique[paper.arxiv_id] = paper
        new_papers.append(paper)
    return new_papers, dup_count


def _format_compact_result(
    *,
    query: str,
    new_papers: list[ArxivPaper],
    dup_count: int,
    total_unique: int,
) -> str:
    """Render a compact ``tool_result`` body for one ``arxiv_search`` call.

    Fields per paper (per ADR-0005 line 43): ``arxiv_id``, ``title``, first
    sentence of abstract, year (from ``published``), ``primary_category``.
    Authors, full abstract, URLs, DOIs, secondary categories are stripped —
    the model doesn't use them at this stage and stripping cuts per-call
    token cost ~10x.

    The header carries the two stopping-signal numbers the system prompt
    points at: how many returned papers were new, and the running total
    unique. Without those, the model has no observable basis for judging
    coverage and would fall back to internal-knowledge heuristics (the
    explicit anti-pattern in the prompt).
    """
    total_returned = len(new_papers) + dup_count
    lines: list[str] = [
        f'Query: "{query}"',
        f"Returned {total_returned} results: {len(new_papers)} new, "
        f"{dup_count} already seen this session.",
        f"Total unique papers so far: {total_unique}.",
    ]
    if not new_papers:
        lines.append("")
        lines.append("(No new papers in this call.)")
        return "\n".join(lines)

    lines.append("")
    for idx, paper in enumerate(new_papers, start=1):
        primary_category = paper.categories[0] if paper.categories else "unknown"
        year = paper.published.year
        first_sentence = _first_sentence(paper.abstract)
        lines.append(
            f"{idx}. arxiv_id={paper.arxiv_id}  ({year}, {primary_category})"
        )
        lines.append(f'   "{paper.title}"')
        lines.append(f"   {first_sentence}")
    return "\n".join(lines)


def _first_sentence(text: str) -> str:
    """Return the first sentence of ``text``, naively.

    A period followed by whitespace is treated as a sentence terminator.
    This deliberately mishandles "e.g." and "Dr." — the same trade-off
    arch_00's heuristic summary makes. The downstream effect is a slightly
    long or slightly truncated first line, never a structural failure.
    Worth fixing if the model is observed misreading these; not before.
    """
    stripped = text.strip()
    for i, ch in enumerate(stripped):
        if ch == "." and (i + 1 == len(stripped) or stripped[i + 1].isspace()):
            return stripped[: i + 1]
    return stripped
