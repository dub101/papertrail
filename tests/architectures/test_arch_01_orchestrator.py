"""Tests for ``arch_01`` stage 6: ``SequentialArchitecture`` (the orchestrator).

Each agent has its own tests for its internal behavior; here we test only
the orchestrator's wiring — that it threads outputs to inputs, assembles
the ``BenchmarkResult`` correctly, lifts year-ints into ``date`` objects,
sums tokens/cost across stages, and propagates exceptions.

Strategy: patch each agent's stage method (``SearchAgent.search`` etc.)
to return canned ``*Result`` objects. The real agents and the
``AsyncAnthropic`` client never run.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import HttpUrl

from papertrail.architectures.arch_01_sequential.era_partition import (
    EraEntry,
    EraPartitionAgent,
    EraPartitionResult,
    EraPartitionUsage,
)
from papertrail.architectures.arch_01_sequential.executive_summary import (
    ExecutiveSummaryAgent,
    ExecutiveSummaryResult,
    ExecutiveSummaryUsage,
)
from papertrail.architectures.arch_01_sequential.orchestrator import (
    SequentialArchitecture,
)
from papertrail.architectures.arch_01_sequential.search import (
    SearchAgent,
    SearchInsufficientResultsError,
    SearchResult,
    SearchTelemetry,
)
from papertrail.architectures.arch_01_sequential.synthesis import (
    PaperSynthesis,
    SynthesisAgent,
    SynthesisResult,
    SynthesisUsage,
)
from papertrail.architectures.arch_01_sequential.triage import (
    TriageAgent,
    TriageResult,
    TriageUsage,
)
from papertrail.benchmark import (
    BenchmarkResult,
    CandidateRecord,
    ErrorRecord,
    SynthesisNote,
)
from papertrail.tools.arxiv import ArxivPaper

# ───── Builders for canned stage results ────────────────────────────────


def _arxiv_paper(idx: int, *, year: int = 2020) -> ArxivPaper:
    return ArxivPaper(
        arxiv_id=f"1706.{idx:05d}",
        title=f"Paper {idx}",
        authors=[f"Author {idx}"],
        categories=["cs.LG"],
        abstract=f"Abstract for paper {idx}.",
        entry_url=HttpUrl(f"https://arxiv.org/abs/1706.{idx:05d}"),
        pdf_url=HttpUrl(f"https://arxiv.org/pdf/1706.{idx:05d}"),
        published=datetime(year, 6, 1, tzinfo=UTC),
        updated=datetime(year, 6, 1, tzinfo=UTC),
    )


def _papers(n: int) -> list[ArxivPaper]:
    return [_arxiv_paper(i, year=2014 + i) for i in range(n)]


def _synthesis(arxiv_id: str, *, confidence: float = 0.85) -> PaperSynthesis:
    return PaperSynthesis(
        arxiv_id=arxiv_id,
        summary_about=f"What {arxiv_id} is about.",
        summary_relation_to_topic=f"How {arxiv_id} relates.",
        summary_problem=f"Problem of {arxiv_id}.",
        summary_approach=f"Approach of {arxiv_id}.",
        summary_impact=f"Impact of {arxiv_id}.",
        confidence=confidence,
    )


def _era(
    era_id: str, paper_ids: list[str], *, start: int = 2014, end: int | None = 2020
) -> EraEntry:
    narrative = (
        f"The {era_id} era marks a methodological shift. Researchers moved "
        "from one set of assumptions to a substantively different framing, "
        "producing a body of work that downstream papers treat as the "
        "canonical starting point. The shift reshaped the questions "
        "practitioners considered worth asking."
    )
    return EraEntry(
        era_id=era_id,
        name=era_id.title(),
        date_range_start_year=start,
        date_range_end_year=end,
        narrative=narrative,
        paper_ids=paper_ids,
    )


def _search_result(
    papers: list[ArxivPaper],
    *,
    iterations: int = 3,
    queries: tuple[str, ...] = ("q1", "q2", "q3"),
    input_tokens: int = 1500,
    output_tokens: int = 400,
    cost_usd: float = 0.0035,
    error_records: tuple[ErrorRecord, ...] = (),
    recovered: bool = False,
) -> SearchResult:
    return SearchResult(
        papers=tuple(papers),
        telemetry=SearchTelemetry(
            iterations_used=iterations,
            queries_issued=queries,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
            final_stop_reason="end_turn" if not recovered else "max_tokens",
            recovered=recovered,
        ),
        error_records=error_records,
    )


def _triage_result(
    selected: list[ArxivPaper],
    *,
    rejected_records: tuple[CandidateRecord, ...] = (),
    input_tokens: int = 8000,
    output_tokens: int = 1500,
    cost_usd: float = 0.0155,
) -> TriageResult:
    selected_records = tuple(
        CandidateRecord(
            arxiv_id=p.arxiv_id,
            title=p.title,
            verdict="included",
            reason=f"selected {p.arxiv_id}",
        )
        for p in selected
    )
    return TriageResult(
        selected=tuple(selected),
        candidate_records=selected_records + rejected_records,
        usage=TriageUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
        ),
    )


def _synthesis_result(
    papers: list[ArxivPaper],
    *,
    confidences: dict[str, float] | None = None,
    notes: tuple[SynthesisNote, ...] = (),
    error_records: tuple[ErrorRecord, ...] = (),
    batches_attempted: int = 2,
    input_tokens: int = 5000,
    output_tokens: int = 2000,
    cost_usd: float = 0.015,
) -> SynthesisResult:
    confs = confidences or {p.arxiv_id: 0.85 for p in papers}
    syntheses = tuple(_synthesis(p.arxiv_id, confidence=confs[p.arxiv_id]) for p in papers)
    return SynthesisResult(
        syntheses=syntheses,
        error_records=error_records,
        notes=notes,
        usage=SynthesisUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
            batches_attempted=batches_attempted,
        ),
    )


def _era_partition_result(
    papers: list[ArxivPaper],
    *,
    input_tokens: int = 4000,
    output_tokens: int = 800,
    cost_usd: float = 0.008,
) -> EraPartitionResult:
    """Build a two-era partition: first half foundations, second half extensions."""
    half = len(papers) // 2
    eras = [
        _era("foundations", [p.arxiv_id for p in papers[:half]], start=2014, end=2018),
        _era("extensions", [p.arxiv_id for p in papers[half:]], start=2017, end=2022),
    ]
    papers_by_era = {e.era_id: tuple(e.paper_ids) for e in eras}
    return EraPartitionResult(
        eras=tuple(eras),
        papers_by_era=papers_by_era,
        usage=EraPartitionUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
        ),
    )


def _executive_summary_result(
    *,
    confidence: float = 0.85,
    input_tokens: int = 5500,
    output_tokens: int = 300,
    cost_usd: float = 0.007,
) -> ExecutiveSummaryResult:
    summary = (
        "The field has evolved across several distinct eras. Early work "
        "established the foundational framing and the canonical problem "
        "definition; subsequent extensions reframed the methodological "
        "centre. The most recent generation synthesises both directions "
        "while addressing remaining limits. Throughout, the central tension "
        "has been between expressiveness and tractability, with each era "
        "negotiating along a different axis. The frontier focuses on "
        "bridging remaining gaps."
    )
    return ExecutiveSummaryResult(
        summary=summary,
        confidence=confidence,
        usage=ExecutiveSummaryUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
        ),
    )


def _fake_client() -> MagicMock:
    """A bare AsyncAnthropic-shaped mock. The orchestrator never calls it
    directly — its agents do, and we patch the agents' stage methods."""
    return MagicMock()


def _patch_all_agents(
    *,
    search: SearchResult,
    triage: TriageResult,
    synth: SynthesisResult,
    eras: EraPartitionResult,
    exec_summary: ExecutiveSummaryResult,
) -> tuple[object, ...]:
    """Return a tuple of context managers that patch each agent method."""
    return (
        patch.object(SearchAgent, "search", new=AsyncMock(return_value=search)),
        patch.object(TriageAgent, "triage", new=AsyncMock(return_value=triage)),
        patch.object(SynthesisAgent, "synthesize", new=AsyncMock(return_value=synth)),
        patch.object(EraPartitionAgent, "partition", new=AsyncMock(return_value=eras)),
        patch.object(ExecutiveSummaryAgent, "summarize", new=AsyncMock(return_value=exec_summary)),
    )


# ───── Class identity ───────────────────────────────────────────────────


def test_architecture_identity() -> None:
    """Required Architecture ABC ClassVars are declared."""
    assert SequentialArchitecture.name == "arch_01_sequential"
    assert SequentialArchitecture.version == "0.1.0"
    assert SequentialArchitecture.description
    assert "sequential" in SequentialArchitecture.description.lower()


def test_constructor_instantiates_all_five_agents() -> None:
    """The orchestrator constructs the five stage agents at __init__."""
    arch = SequentialArchitecture(_fake_client())
    assert isinstance(arch._search, SearchAgent)
    assert isinstance(arch._triage, TriageAgent)
    assert isinstance(arch._synthesis, SynthesisAgent)
    assert isinstance(arch._era_partition, EraPartitionAgent)
    assert isinstance(arch._executive_summary, ExecutiveSummaryAgent)


# ───── Happy path ───────────────────────────────────────────────────────


async def test_run_happy_path_produces_valid_benchmark_result() -> None:
    """Five stages succeed; BenchmarkResult validates and contains every paper."""
    papers = _papers(8)
    patches = _patch_all_agents(
        search=_search_result(papers),
        triage=_triage_result(papers),
        synth=_synthesis_result(papers),
        eras=_era_partition_result(papers),
        exec_summary=_executive_summary_result(),
    )

    arch = SequentialArchitecture(_fake_client())
    with patches[0], patches[1], patches[2], patches[3], patches[4]:
        result = await arch.run("the topic")

    assert isinstance(result, BenchmarkResult)
    assert result.deliverable.topic == "the topic"
    assert len(result.deliverable.papers) == 8
    assert {p.arxiv_id for p in result.deliverable.papers} == {p.arxiv_id for p in papers}
    assert len(result.deliverable.timeline) == 2  # foundations + extensions


# ───── Confidence propagation ───────────────────────────────────────────


async def test_paper_entry_confidence_propagates_from_synthesis() -> None:
    """PaperEntry.confidence comes directly from PaperSynthesis.confidence."""
    papers = _papers(8)
    custom_confidences = {p.arxiv_id: 0.5 + i * 0.05 for i, p in enumerate(papers)}
    patches = _patch_all_agents(
        search=_search_result(papers),
        triage=_triage_result(papers),
        synth=_synthesis_result(papers, confidences=custom_confidences),
        eras=_era_partition_result(papers),
        exec_summary=_executive_summary_result(),
    )

    arch = SequentialArchitecture(_fake_client())
    with patches[0], patches[1], patches[2], patches[3], patches[4]:
        result = await arch.run("t")

    by_id = {p.arxiv_id: p for p in result.deliverable.papers}
    for paper in papers:
        assert by_id[paper.arxiv_id].confidence == custom_confidences[paper.arxiv_id]


async def test_disclaimer_synthesis_yields_zero_confidence_paper_entry() -> None:
    """A synthesis with confidence=0.0 (disclaimer) carries through to PaperEntry."""
    papers = _papers(8)
    confidences = {p.arxiv_id: 0.85 for p in papers}
    confidences[papers[2].arxiv_id] = 0.0  # disclaimer case
    patches = _patch_all_agents(
        search=_search_result(papers),
        triage=_triage_result(papers),
        synth=_synthesis_result(papers, confidences=confidences),
        eras=_era_partition_result(papers),
        exec_summary=_executive_summary_result(),
    )

    arch = SequentialArchitecture(_fake_client())
    with patches[0], patches[1], patches[2], patches[3], patches[4]:
        result = await arch.run("t")

    by_id = {p.arxiv_id: p for p in result.deliverable.papers}
    assert by_id[papers[2].arxiv_id].confidence == 0.0


async def test_paper_entry_citation_fields_are_none() -> None:
    """ADR-0005: citation lookup deferred for arch_01; citation_* stay None."""
    papers = _papers(8)
    patches = _patch_all_agents(
        search=_search_result(papers),
        triage=_triage_result(papers),
        synth=_synthesis_result(papers),
        eras=_era_partition_result(papers),
        exec_summary=_executive_summary_result(),
    )

    arch = SequentialArchitecture(_fake_client())
    with patches[0], patches[1], patches[2], patches[3], patches[4]:
        result = await arch.run("t")

    for paper in result.deliverable.papers:
        assert paper.citation_count is None
        assert paper.citation_source is None


# ───── Era lifting (year-int → date) ────────────────────────────────────


async def test_era_year_ints_lift_to_jan_1_dec_31_dates() -> None:
    """date_range_start_year=2014 → date(2014, 1, 1); end=2018 → date(2018, 12, 31)."""
    papers = _papers(8)
    patches = _patch_all_agents(
        search=_search_result(papers),
        triage=_triage_result(papers),
        synth=_synthesis_result(papers),
        eras=_era_partition_result(papers),  # foundations 2014-2018, extensions 2017-2022
        exec_summary=_executive_summary_result(),
    )

    arch = SequentialArchitecture(_fake_client())
    with patches[0], patches[1], patches[2], patches[3], patches[4]:
        result = await arch.run("t")

    foundations = next(t for t in result.deliverable.timeline if t.era_id == "foundations")
    extensions = next(t for t in result.deliverable.timeline if t.era_id == "extensions")
    assert foundations.date_range_start == date(2014, 1, 1)
    assert foundations.date_range_end == date(2018, 12, 31)
    assert extensions.date_range_start == date(2017, 1, 1)
    assert extensions.date_range_end == date(2022, 12, 31)


async def test_open_ended_era_preserves_none_end_date() -> None:
    """date_range_end_year=None → TimelineEra.date_range_end=None (ongoing)."""
    papers = _papers(6)
    half = len(papers) // 2
    eras = EraPartitionResult(
        eras=(
            _era("past", [p.arxiv_id for p in papers[:half]], start=2014, end=2018),
            _era("frontier", [p.arxiv_id for p in papers[half:]], start=2019, end=None),
        ),
        papers_by_era={
            "past": tuple(p.arxiv_id for p in papers[:half]),
            "frontier": tuple(p.arxiv_id for p in papers[half:]),
        },
        usage=EraPartitionUsage(input_tokens=100, output_tokens=50, cost_usd=0.001),
    )
    patches = _patch_all_agents(
        search=_search_result(papers),
        triage=_triage_result(papers),
        synth=_synthesis_result(papers),
        eras=eras,
        exec_summary=_executive_summary_result(),
    )

    arch = SequentialArchitecture(_fake_client())
    with patches[0], patches[1], patches[2], patches[3], patches[4]:
        result = await arch.run("t")

    frontier = next(t for t in result.deliverable.timeline if t.era_id == "frontier")
    assert frontier.date_range_end is None
    assert frontier.date_range_start == date(2019, 1, 1)


# ───── Telemetry aggregation ────────────────────────────────────────────


async def test_telemetry_sums_tokens_and_cost_across_stages() -> None:
    """Total tokens/cost == sum of the five stages."""
    papers = _papers(8)
    patches = _patch_all_agents(
        search=_search_result(papers, input_tokens=1000, output_tokens=200, cost_usd=0.002),
        triage=_triage_result(papers, input_tokens=2000, output_tokens=500, cost_usd=0.0045),
        synth=_synthesis_result(papers, input_tokens=3000, output_tokens=1000, cost_usd=0.008),
        eras=_era_partition_result(papers, input_tokens=2500, output_tokens=400, cost_usd=0.0045),
        exec_summary=_executive_summary_result(
            input_tokens=4000, output_tokens=200, cost_usd=0.005
        ),
    )

    arch = SequentialArchitecture(_fake_client())
    with patches[0], patches[1], patches[2], patches[3], patches[4]:
        result = await arch.run("t")

    assert result.telemetry.total_input_tokens == 1000 + 2000 + 3000 + 2500 + 4000
    assert result.telemetry.total_output_tokens == 200 + 500 + 1000 + 400 + 200
    assert result.telemetry.total_cost_usd == pytest.approx(0.002 + 0.0045 + 0.008 + 0.0045 + 0.005)


async def test_telemetry_agent_and_tool_call_counts() -> None:
    """agent_call_count = search_iters + 1 + synth_batches + 1 + 1."""
    papers = _papers(8)
    patches = _patch_all_agents(
        search=_search_result(papers, iterations=4, queries=("q1", "q2", "q3", "q4")),
        triage=_triage_result(papers),
        synth=_synthesis_result(papers, batches_attempted=3),
        eras=_era_partition_result(papers),
        exec_summary=_executive_summary_result(),
    )

    arch = SequentialArchitecture(_fake_client())
    with patches[0], patches[1], patches[2], patches[3], patches[4]:
        result = await arch.run("t")

    # 4 (search) + 1 (triage) + 3 (synth) + 1 (era) + 1 (exec) = 10
    assert result.telemetry.agent_call_count == 10
    # tool_call_count = 4 queries + 1 + 3 + 1 + 1 = 10 (1:1 here)
    assert result.telemetry.tool_call_count == 10


async def test_telemetry_aggregates_error_records_from_search_and_synthesis() -> None:
    """error_records from search + synthesis flow into Telemetry.errors."""
    papers = _papers(8)
    search_error = ErrorRecord(
        step_index=1,
        category="api",
        message="search hit cap, recovered with 8 papers",
        recovered=True,
    )
    synth_error = ErrorRecord(
        step_index=3,
        category="api",
        message="synthesis batch 2 failed",
        recovered=True,
    )
    patches = _patch_all_agents(
        search=_search_result(papers, error_records=(search_error,), recovered=True),
        triage=_triage_result(papers),
        synth=_synthesis_result(papers, error_records=(synth_error,)),
        eras=_era_partition_result(papers),
        exec_summary=_executive_summary_result(),
    )

    arch = SequentialArchitecture(_fake_client())
    with patches[0], patches[1], patches[2], patches[3], patches[4]:
        result = await arch.run("t")

    assert len(result.telemetry.errors) == 2
    assert result.telemetry.errors[0].step_index == 1
    assert result.telemetry.errors[1].step_index == 3


async def test_telemetry_carries_synthesis_notes_and_candidate_records() -> None:
    """synth.notes → Telemetry.synthesis_notes; triage.candidate_records → Telemetry.candidates."""
    papers = _papers(8)
    note = SynthesisNote(arxiv_id=papers[0].arxiv_id, note="abstract has a typo")
    patches = _patch_all_agents(
        search=_search_result(papers),
        triage=_triage_result(papers),
        synth=_synthesis_result(papers, notes=(note,)),
        eras=_era_partition_result(papers),
        exec_summary=_executive_summary_result(),
    )

    arch = SequentialArchitecture(_fake_client())
    with patches[0], patches[1], patches[2], patches[3], patches[4]:
        result = await arch.run("t")

    assert len(result.telemetry.synthesis_notes) == 1
    assert result.telemetry.synthesis_notes[0].arxiv_id == papers[0].arxiv_id
    assert result.telemetry.synthesis_notes[0].note == "abstract has a typo"
    # Candidates: 8 papers all "included" verdicts.
    assert len(result.telemetry.candidates) == 8


# ───── Prompt versions / modes ──────────────────────────────────────────


async def test_default_prompt_versions_recorded() -> None:
    """When prompt_versions=None, Telemetry records the agents' own ClassVar versions."""
    papers = _papers(8)
    patches = _patch_all_agents(
        search=_search_result(papers),
        triage=_triage_result(papers),
        synth=_synthesis_result(papers),
        eras=_era_partition_result(papers),
        exec_summary=_executive_summary_result(),
    )

    arch = SequentialArchitecture(_fake_client())
    with patches[0], patches[1], patches[2], patches[3], patches[4]:
        result = await arch.run("t")

    expected = {
        "arch_01_search": "v1",
        "arch_01_triage": "v1",
        "arch_01_synthesis": "v1",
        "arch_01_era_partition": "v1",
        "arch_01_executive_summary": "v1",
    }
    assert result.telemetry.prompt_versions == expected


async def test_supplied_prompt_versions_recorded() -> None:
    """When prompt_versions is supplied, it's used in Telemetry verbatim."""
    papers = _papers(8)
    patches = _patch_all_agents(
        search=_search_result(papers),
        triage=_triage_result(papers),
        synth=_synthesis_result(papers),
        eras=_era_partition_result(papers),
        exec_summary=_executive_summary_result(),
    )
    supplied = {"arch_01_search": "v2", "arch_01_triage": "v1"}

    arch = SequentialArchitecture(_fake_client())
    with patches[0], patches[1], patches[2], patches[3], patches[4]:
        result = await arch.run("t", prompt_versions=supplied)

    assert result.telemetry.prompt_versions == supplied


# ───── Failure propagation ──────────────────────────────────────────────


async def test_run_propagates_search_insufficient_results_error() -> None:
    """SearchInsufficientResultsError from stage 1 bubbles up cleanly."""
    arch = SequentialArchitecture(_fake_client())
    err = SearchInsufficientResultsError(topic="niche-topic", found=2, stop_reason="end_turn")
    with (
        patch.object(SearchAgent, "search", new=AsyncMock(side_effect=err)),
        pytest.raises(SearchInsufficientResultsError) as exc_info,
    ):
        await arch.run("niche-topic")
    assert exc_info.value.found == 2


# ───── Provenance ───────────────────────────────────────────────────────


async def test_provenance_uses_finished_at_for_created_at() -> None:
    """build_provenance(created_at=finished_at) — the run timestamp lines up."""
    papers = _papers(8)
    patches = _patch_all_agents(
        search=_search_result(papers),
        triage=_triage_result(papers),
        synth=_synthesis_result(papers),
        eras=_era_partition_result(papers),
        exec_summary=_executive_summary_result(),
    )

    arch = SequentialArchitecture(_fake_client())
    with patches[0], patches[1], patches[2], patches[3], patches[4]:
        result = await arch.run("t")

    # Provenance.created_at and Telemetry.finished_at must match: the
    # orchestrator passes finished_at into build_provenance().
    assert result.provenance.created_at == result.telemetry.finished_at
    assert result.provenance.papertrail_version
    assert result.provenance.git_sha
