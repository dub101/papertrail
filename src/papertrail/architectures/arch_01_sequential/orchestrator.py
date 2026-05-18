"""arch_01 stage 6 — sequential pipeline orchestrator.

What this module does (in one paragraph):
    Implements ``SequentialArchitecture``, the concrete ``Architecture``
    subclass that wires the five LLM stages of arch_01 into one runnable
    pipeline. ``run(topic)`` instantiates each agent (search, triage,
    synthesis, era partition, executive summary), calls them in order
    threading outputs to inputs, then assembles the final
    ``BenchmarkResult``: the ``Deliverable`` (papers + timeline +
    overall_summary), aggregated ``Telemetry`` (tokens, cost, errors,
    candidates, notes summed across stages), and ``Provenance`` (via
    the shared ``papertrail.provenance`` helpers). Pure-code stage —
    no LLM calls of its own.

Why pure programmatic handoff and not prompt-based routing:
    Each stage's output flows to the next stage's input as a typed
    Python argument, not as text the model interprets. That's the
    cert's D1 TS 1.4 "programmatic enforcement vs prompt-based
    guidance" distinction: deterministic compliance for the workflow,
    probabilistic compliance for the per-stage content.

Cert mappings:
    - **D1 TS 1.6** (primary) — prompt chaining; the architecture is
      the chain. Five LLM stages composed sequentially.
    - **D1 TS 1.4** (secondary) — multi-step workflow with programmatic
      handoff. The orchestrator routes outputs to inputs in Python; the
      model doesn't decide what runs next.
    - **D5 TS 5.3** (secondary) — error propagation: stage exceptions
      surface with stage attribution; partial-results recovery from
      stages 1 and 3 is preserved via ``Telemetry.errors``.
    - **D5 TS 5.6** (tertiary) — full audit chain: candidate (stage 1)
      → selected (stage 2) → synthesised (stage 3) → assigned to era
      (stage 4) → represented in summary (stage 5).

Libraries: anthropic SDK (AsyncAnthropic for client injection),
papertrail.architecture (the ABC), papertrail.benchmark (every schema
type touched at assembly), papertrail.provenance (shared helpers).
"""

from __future__ import annotations

import time
from datetime import UTC, date, datetime
from typing import TYPE_CHECKING, ClassVar

from papertrail.architecture import Architecture
from papertrail.architectures.arch_01_sequential.era_partition import (
    EraPartitionAgent,
)
from papertrail.architectures.arch_01_sequential.executive_summary import (
    ExecutiveSummaryAgent,
)
from papertrail.architectures.arch_01_sequential.search import SearchAgent
from papertrail.architectures.arch_01_sequential.synthesis import SynthesisAgent
from papertrail.architectures.arch_01_sequential.triage import TriageAgent
from papertrail.benchmark import (
    BenchmarkResult,
    Deliverable,
    Modes,
    PaperEntry,
    Telemetry,
    TimelineEra,
)
from papertrail.provenance import build_provenance

if TYPE_CHECKING:
    from anthropic import AsyncAnthropic

    from papertrail.architectures.arch_01_sequential.era_partition import (
        EraEntry,
        EraPartitionResult,
    )
    from papertrail.architectures.arch_01_sequential.executive_summary import (
        ExecutiveSummaryResult,
    )
    from papertrail.architectures.arch_01_sequential.search import SearchResult
    from papertrail.architectures.arch_01_sequential.synthesis import (
        PaperSynthesis,
        SynthesisResult,
    )
    from papertrail.architectures.arch_01_sequential.triage import TriageResult
    from papertrail.tools.arxiv import ArxivPaper


# ───── The architecture ─────────────────────────────────────────────────


class SequentialArchitecture(Architecture):
    """arch_01: 5-stage sequential LLM pipeline.

    Stage flow (each is a separate agent class with its own prompt,
    schema, and unit tests):

        1. Search (agentic loop over arxiv_search)
        2. Triage (forced tool_use; pick 4-12 from 25-35 candidates)
        3. Synthesis (parallel batched + one retry; 5 fields per paper)
        4. Era partition (forced tool_use; 2-4 content-driven eras)
        5. Executive summary (forced tool_use; top-level prose + confidence)
        6. Assembly (this class; pure code, no LLM)

    The orchestrator owns the cross-stage concerns the individual agents
    can't see: year-int -> date lifting for TimelineEra, telemetry
    aggregation, provenance injection, PaperEntry.confidence direct
    passthrough from the per-paper confidence emitted by stage 3.
    """

    name: ClassVar[str] = "arch_01_sequential"
    version: ClassVar[str] = "0.1.0"
    description: ClassVar[str] = (
        "Five-stage sequential LLM pipeline: agentic search, structured "
        "triage, parallel batched synthesis with one retry, content-driven "
        "era partition, top-level executive summary. Each stage is a "
        "separate Anthropic call with a pydantic-validated output."
    )

    def __init__(
        self,
        client: AsyncAnthropic,
        *,
        model: str | None = None,
    ) -> None:
        """Inject the Anthropic client; instantiate the five agents.

        ``model`` is forwarded to every agent. The default
        (``claude-haiku-4-5`` per ADR-0003) keeps cost comparable
        across runs; the override exists for ad-hoc experimentation
        (e.g. running the whole pipeline on Sonnet for an A/B).
        """
        self._client = client
        self._model = model
        self._search = SearchAgent(client, model=model)
        self._triage = TriageAgent(client, model=model)
        self._synthesis = SynthesisAgent(client, model=model)
        self._era_partition = EraPartitionAgent(client, model=model)
        self._executive_summary = ExecutiveSummaryAgent(client, model=model)

    async def run(
        self,
        topic: str,
        *,
        prompt_versions: dict[str, str] | None = None,
        modes: Modes | None = None,
    ) -> BenchmarkResult:
        """Execute the five-stage pipeline against ``topic``.

        Stage failures (any of the stage-specific exceptions) propagate
        unchanged. The orchestrator does not catch them; the runner
        decides how to report.

        ``prompt_versions`` is accepted for the Architecture ABC's
        contract but not currently used as an override (each agent uses
        its own ``PROMPT_VERSION`` ClassVar); the supplied dict is
        recorded in ``Telemetry.prompt_versions`` for reproducibility.
        See open follow-up below.

        Raises:
            SearchInsufficientResultsError: <4 papers from search.
            TriageError (subclasses): triage refused / invalid / failed
                the quality gate.
            SynthesisError: zero papers synthesised across both rounds.
            EraPartitionError (subclasses): era partition refused /
                invalid output (schema or cross-field).
            ExecutiveSummaryError (subclasses): executive summary
                refused / invalid output.
        """
        modes = modes or Modes()
        started_at = datetime.now(UTC)
        t0 = time.perf_counter()

        # ─── Stage 1: agentic search loop ───
        search = await self._search.search(topic)

        # ─── Stage 2: triage and select 4-12 papers ───
        # Stage 1's agentic loop is nondeterministic and can return 80+
        # papers on a productive run. The triage prompt expects ~25-35
        # candidates; handing the model 80 forces it to emit 80 decisions
        # and blow past max_tokens. Cap at 40 in arxiv-relevance order so
        # triage always sees a manageable pool.
        _TRIAGE_INPUT_CAP = 40
        candidate_pool = list(search.papers[:_TRIAGE_INPUT_CAP])
        triage = await self._triage.triage(topic=topic, candidates=candidate_pool)

        # ─── Stage 3: per-paper synthesis (parallel batches + one retry) ───
        synth = await self._synthesis.synthesize(topic=topic, papers=list(triage.selected))

        # ─── Stage 4: content-driven era partition ───
        eras = await self._era_partition.partition(
            topic=topic,
            papers=list(triage.selected),
            syntheses=list(synth.syntheses),
        )

        # ─── Stage 5: executive summary + confidence ───
        exec_summary = await self._executive_summary.summarize(
            topic=topic,
            eras=list(eras.eras),
            syntheses=list(synth.syntheses),
        )

        finished_at = datetime.now(UTC)
        duration_seconds = time.perf_counter() - t0

        # ─── Stage 6: assembly (pure code, no LLM) ───
        return self._assemble(
            topic=topic,
            prompt_versions=prompt_versions,
            modes=modes,
            started_at=started_at,
            finished_at=finished_at,
            duration_seconds=duration_seconds,
            search=search,
            triage=triage,
            synth=synth,
            eras=eras,
            exec_summary=exec_summary,
        )

    # ─── Assembly ────────────────────────────────────────────────────────

    @staticmethod
    def _assemble(
        *,
        topic: str,
        prompt_versions: dict[str, str] | None,
        modes: Modes,
        started_at: datetime,
        finished_at: datetime,
        duration_seconds: float,
        search: SearchResult,
        triage: TriageResult,
        synth: SynthesisResult,
        eras: EraPartitionResult,
        exec_summary: ExecutiveSummaryResult,
    ) -> BenchmarkResult:
        """Compose the five stages' outputs into one ``BenchmarkResult``."""
        # Build a synthesis lookup so PaperEntry assembly is O(1) per paper.
        synth_by_id: dict[str, PaperSynthesis] = {s.arxiv_id: s for s in synth.syntheses}
        # Build a paper-to-era_id map from the era partition.
        era_id_by_paper: dict[str, str] = {}
        for era in eras.eras:
            for paper_id in era.paper_ids:
                era_id_by_paper[paper_id] = era.era_id

        # PaperEntries are assembled in the order triage chose (the user's
        # preferred ordering); confidence comes directly from the per-paper
        # synthesis (a disclaimer entry carries confidence=0.0).
        papers = [
            SequentialArchitecture._build_paper_entry(
                paper=paper,
                synthesis=synth_by_id[paper.arxiv_id],
                era_id=era_id_by_paper[paper.arxiv_id],
            )
            for paper in triage.selected
        ]

        # TimelineEras are built by lifting the year-ints from EraEntry
        # into proper ``date`` objects (Jan 1 / Dec 31 for the whole-year
        # convention). Order preserves the model's emitted order.
        timeline = [SequentialArchitecture._lift_era(era) for era in eras.eras]

        deliverable = Deliverable(
            topic=topic,
            overall_summary=exec_summary.summary,
            papers=papers,
            timeline=timeline,
        )

        # Per-stage usage rollup. Each stage exposes input/output tokens
        # and a cost. Notes flow into Telemetry.synthesis_notes.
        total_input_tokens = (
            search.telemetry.input_tokens
            + triage.usage.input_tokens
            + synth.usage.input_tokens
            + eras.usage.input_tokens
            + exec_summary.usage.input_tokens
        )
        total_output_tokens = (
            search.telemetry.output_tokens
            + triage.usage.output_tokens
            + synth.usage.output_tokens
            + eras.usage.output_tokens
            + exec_summary.usage.output_tokens
        )
        total_cost_usd = (
            search.telemetry.cost_usd
            + triage.usage.cost_usd
            + synth.usage.cost_usd
            + eras.usage.cost_usd
            + exec_summary.usage.cost_usd
        )

        # Agent calls = number of messages.create invocations across
        # all stages. Search makes one call per loop iteration; triage,
        # era, exec summary each make one; synthesis makes one per
        # batch attempted (round 1 + retry round). Tool calls track
        # the same in this architecture (each stage's call corresponds
        # to one tool invocation: arxiv_search for stage 1, the
        # submit_* tools for the rest).
        agent_call_count = (
            search.telemetry.iterations_used + 1 + synth.usage.batches_attempted + 1 + 1
        )
        # Stage 1 may emit multiple tool_use blocks per iteration if
        # the model decides to issue parallel queries — we tracked
        # those as ``queries_issued``. Other stages are 1:1.
        tool_call_count = (
            len(search.telemetry.queries_issued) + 1 + synth.usage.batches_attempted + 1 + 1
        )

        # All error records concatenated. Stage 1 may emit one
        # ErrorRecord(recovered=True) (partial-results recovery);
        # synthesis may emit several (one per failed batch + one per
        # permanently-missing paper). Other stages don't emit any.
        errors = [*search.error_records, *synth.error_records]

        telemetry = Telemetry(
            architecture_name=SequentialArchitecture.name,
            architecture_version=SequentialArchitecture.version,
            prompt_versions=prompt_versions or _default_prompt_versions(),
            modes=modes,
            started_at=started_at,
            finished_at=finished_at,
            duration_seconds=duration_seconds,
            total_input_tokens=total_input_tokens,
            total_output_tokens=total_output_tokens,
            total_cost_usd=total_cost_usd,
            agent_call_count=agent_call_count,
            tool_call_count=tool_call_count,
            trace=[],
            errors=errors,
            candidates=list(triage.candidate_records),
            unresolved_lookups=[],
            synthesis_notes=list(synth.notes),
        )

        provenance = build_provenance(created_at=finished_at)

        return BenchmarkResult(
            deliverable=deliverable,
            telemetry=telemetry,
            provenance=provenance,
        )

    # ─── Per-element assembly helpers ───────────────────────────────────

    @staticmethod
    def _build_paper_entry(
        *,
        paper: ArxivPaper,
        synthesis: PaperSynthesis,
        era_id: str,
    ) -> PaperEntry:
        """Compose one ``PaperEntry`` from the search paper + stage-3 synthesis.

        ``citation_count`` / ``citation_source`` stay ``None`` — ADR-0005
        defers citation lookup to arch_02. ``confidence`` is the
        per-paper value emitted by stage 3 (disclaimer entries carry
        ``0.0``; real syntheses carry the model's self-rated value).
        """
        return PaperEntry(
            arxiv_id=paper.arxiv_id,
            title=paper.title,
            authors=paper.authors,
            published_date=paper.published.date(),
            url=paper.entry_url,
            citation_count=None,
            citation_source=None,
            era_id=era_id,
            summary_about=synthesis.summary_about,
            summary_relation_to_topic=synthesis.summary_relation_to_topic,
            summary_problem=synthesis.summary_problem,
            summary_approach=synthesis.summary_approach,
            summary_impact=synthesis.summary_impact,
            confidence=synthesis.confidence,
        )

    @staticmethod
    def _lift_era(era: EraEntry) -> TimelineEra:
        """Lift an ``EraEntry`` (year ints) into a ``TimelineEra`` (date objects).

        Whole-year convention: ``date(year, 1, 1)`` for the start and
        ``date(year, 12, 31)`` for the end. ``None`` end stays ``None``
        (the "ongoing" / "frontier" era case).
        """
        end: date | None = None
        if era.date_range_end_year is not None:
            end = date(era.date_range_end_year, 12, 31)
        return TimelineEra(
            era_id=era.era_id,
            name=era.name,
            date_range_start=date(era.date_range_start_year, 1, 1),
            date_range_end=end,
            narrative=era.narrative,
            paper_ids=list(era.paper_ids),
        )


# ───── Module helpers ───────────────────────────────────────────────────


def _default_prompt_versions() -> dict[str, str]:
    """Map of prompt-slug to version for the default arch_01 configuration.

    Mirrors the ``PROMPT_NAME`` / ``PROMPT_VERSION`` ClassVars on each
    agent. Captured here as a small fact so ``Telemetry.prompt_versions``
    is faithful when the orchestrator runs with default prompts (the
    common case). Bumps when any of the five prompt versions bump.

    Open follow-up: when ``run(..., prompt_versions=...)`` is supplied,
    we currently record the supplied dict but the agents still use their
    own ClassVars. Wiring the override through requires letting each
    agent class accept a version at construction. Deferred until a
    second prompt version of any agent exists.
    """
    return {
        SearchAgent.PROMPT_NAME: SearchAgent.PROMPT_VERSION,
        TriageAgent.PROMPT_NAME: TriageAgent.PROMPT_VERSION,
        SynthesisAgent.PROMPT_NAME: SynthesisAgent.PROMPT_VERSION,
        EraPartitionAgent.PROMPT_NAME: EraPartitionAgent.PROMPT_VERSION,
        ExecutiveSummaryAgent.PROMPT_NAME: ExecutiveSummaryAgent.PROMPT_VERSION,
    }
