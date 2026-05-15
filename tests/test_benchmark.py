"""Tests for the BenchmarkResult schema.

Covers happy-path construction and each validator failure case. The fixtures
build minimal-but-valid instances; tests mutate one field at a time to assert
the corresponding validator catches the bad input.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Any

import pytest
from pydantic import ValidationError

from papertrail.benchmark import (
    SCHEMA_VERSION,
    BenchmarkResult,
    CandidateRecord,
    Deliverable,
    ErrorRecord,
    Modes,
    PaperEntry,
    Provenance,
    Telemetry,
    TimelineEra,
    ToolCallRecord,
    TraceStep,
    UnresolvedLookup,
)

# ───── Helpers ──────────────────────────────────────────────────────────


def _paper(arxiv_id: str, era_id: str, **overrides: Any) -> PaperEntry:
    """Build a valid PaperEntry; pass overrides to mutate specific fields."""
    defaults: dict[str, Any] = {
        "arxiv_id": arxiv_id,
        "title": f"Paper {arxiv_id}",
        "authors": ["Doe, J."],
        "published_date": date(2020, 1, 1),
        "url": f"https://arxiv.org/abs/{arxiv_id}",
        "era_id": era_id,
        "summary_about": "What the paper is about.",
        "summary_relation_to_topic": "How it relates to the topic.",
        "summary_problem": "Problem it tackles.",
        "summary_approach": "How it solves the problem.",
        "summary_impact": "Impact on the field.",
        "confidence": 0.8,
    }
    defaults.update(overrides)
    return PaperEntry(**defaults)


def _era(era_id: str, paper_ids: list[str], **overrides: Any) -> TimelineEra:
    """Build a valid TimelineEra."""
    defaults: dict[str, Any] = {
        "era_id": era_id,
        "name": era_id.title(),
        "date_range_start": date(2017, 1, 1),
        "date_range_end": date(2020, 12, 31),
        "narrative": "What happened in this era.",
        "paper_ids": paper_ids,
    }
    defaults.update(overrides)
    return TimelineEra(**defaults)


def _arxiv_ids(n: int) -> list[str]:
    """Generate ``n`` distinct, regex-valid arxiv IDs."""
    return [f"1706.{i:05d}" for i in range(n)]


@pytest.fixture
def valid_deliverable() -> Deliverable:
    """Minimal-but-valid Deliverable: 8 papers split across 2 eras."""
    ids = _arxiv_ids(8)
    papers = [_paper(ids[i], era_id="era1" if i < 4 else "era2") for i in range(8)]
    return Deliverable(
        topic="Positional embeddings for attention",
        overall_summary="Executive summary covering all eras.",
        papers=papers,
        timeline=[
            _era("era1", paper_ids=ids[:4]),
            _era("era2", paper_ids=ids[4:]),
        ],
    )


@pytest.fixture
def valid_telemetry() -> Telemetry:
    """Minimal-but-valid Telemetry."""
    return Telemetry(
        architecture_name="arch_01_sequential",
        architecture_version="0.1.0",
        prompt_versions={"researcher": "v1"},
        modes=Modes(),
        started_at=datetime(2026, 5, 13, 10, 0, tzinfo=UTC),
        finished_at=datetime(2026, 5, 13, 10, 5, tzinfo=UTC),
        duration_seconds=300.0,
        total_input_tokens=1000,
        total_output_tokens=500,
        total_cost_usd=0.01,
        agent_call_count=3,
        tool_call_count=2,
    )


@pytest.fixture
def valid_provenance() -> Provenance:
    """Minimal-but-valid Provenance."""
    return Provenance(
        schema_version=SCHEMA_VERSION,
        papertrail_version="0.1.0",
        git_sha="abc123",
        created_at=datetime(2026, 5, 13, 10, 5, tzinfo=UTC),
    )


# ───── Happy path ───────────────────────────────────────────────────────


def test_valid_benchmark_result_constructs(
    valid_deliverable: Deliverable,
    valid_telemetry: Telemetry,
    valid_provenance: Provenance,
) -> None:
    """A valid BenchmarkResult assembles from valid sub-models."""
    result = BenchmarkResult(
        deliverable=valid_deliverable,
        telemetry=valid_telemetry,
        provenance=valid_provenance,
    )
    assert result.deliverable.topic == "Positional embeddings for attention"
    assert len(result.deliverable.papers) == 8
    assert result.telemetry.architecture_name == "arch_01_sequential"
    assert result.provenance.schema_version == SCHEMA_VERSION


def test_papertrail_result_round_trips_through_json(
    valid_deliverable: Deliverable,
    valid_telemetry: Telemetry,
    valid_provenance: Provenance,
) -> None:
    """A round-trip via JSON preserves equality — the schema is JSON-safe."""
    original = BenchmarkResult(
        deliverable=valid_deliverable,
        telemetry=valid_telemetry,
        provenance=valid_provenance,
    )
    restored = BenchmarkResult.model_validate_json(original.model_dump_json())
    assert restored == original


# ───── Deliverable size floor (ADR-0006) ────────────────────────────────


def test_deliverable_rejects_fewer_than_four_papers() -> None:
    """ADR-0006 sets the lower bound at 4 papers — 3 must raise."""
    ids = _arxiv_ids(3)
    papers = [_paper(ids[i], era_id="era1") for i in range(3)]
    with pytest.raises(ValidationError, match="at least 4 items"):
        Deliverable(
            topic="t",
            overall_summary="s",
            papers=papers,
            timeline=[_era("era1", paper_ids=ids)],
        )


def test_deliverable_accepts_exactly_four_papers() -> None:
    """The 4-paper boundary is inclusive — exactly 4 must construct cleanly."""
    ids = _arxiv_ids(4)
    papers = [_paper(ids[i], era_id="era1") for i in range(4)]
    d = Deliverable(
        topic="thin-data topic",
        overall_summary="A topic with only four high-signal papers.",
        papers=papers,
        timeline=[_era("era1", paper_ids=ids)],
    )
    assert len(d.papers) == 4


# ───── Cross-list integrity on Deliverable ──────────────────────────────


def test_paper_with_unknown_era_id_rejected() -> None:
    """A paper.era_id that doesn't match any timeline entry is rejected."""
    ids = _arxiv_ids(8)
    papers = [_paper(ids[i], era_id="ghost") for i in range(8)]
    with pytest.raises(ValidationError, match="unknown era_id"):
        Deliverable(
            topic="t",
            overall_summary="s",
            papers=papers,
            timeline=[_era("era1", paper_ids=ids)],
        )


def test_era_referencing_unknown_paper_rejected() -> None:
    """An era.paper_ids entry that points at no real paper is rejected."""
    ids = _arxiv_ids(8)
    papers = [_paper(ids[i], era_id="era1") for i in range(8)]
    bogus_id = "9999.99999"
    with pytest.raises(ValidationError, match="unknown paper_id"):
        Deliverable(
            topic="t",
            overall_summary="s",
            papers=papers,
            timeline=[_era("era1", paper_ids=[*ids, bogus_id])],
        )


def test_paper_appearing_in_multiple_eras_rejected() -> None:
    """No paper may appear in more than one era's paper_ids."""
    ids = _arxiv_ids(8)
    papers = [_paper(ids[i], era_id="era1") for i in range(8)]
    with pytest.raises(ValidationError, match="multiple eras"):
        Deliverable(
            topic="t",
            overall_summary="s",
            papers=papers,
            timeline=[
                _era("era1", paper_ids=ids),
                _era("era2", paper_ids=[ids[0]]),  # duplicate of era1
            ],
        )


def test_orphan_paper_rejected() -> None:
    """A paper whose era exists but doesn't list it is caught as 'not assigned'."""
    ids = _arxiv_ids(8)
    papers = [_paper(ids[i], era_id="era1") for i in range(8)]
    with pytest.raises(ValidationError, match="not assigned"):
        Deliverable(
            topic="t",
            overall_summary="s",
            papers=papers,
            timeline=[_era("era1", paper_ids=ids[:7])],  # last paper orphaned
        )


# ───── Other validators ─────────────────────────────────────────────────


def test_era_end_before_start_rejected() -> None:
    """date_range_end must not predate date_range_start."""
    with pytest.raises(ValidationError, match="date_range_end"):
        TimelineEra(
            era_id="bad",
            name="Bad",
            date_range_start=date(2020, 1, 1),
            date_range_end=date(2019, 1, 1),
            narrative="x",
            paper_ids=["1706.03762"],
        )


def test_era_end_none_is_allowed() -> None:
    """An era with no end date is valid (ongoing/frontier work)."""
    era = TimelineEra(
        era_id="frontier",
        name="Frontier",
        date_range_start=date(2024, 1, 1),
        date_range_end=None,
        narrative="Ongoing work.",
        paper_ids=["1706.03762"],
    )
    assert era.date_range_end is None


def test_telemetry_finished_before_started_rejected() -> None:
    """finished_at must be on or after started_at."""
    with pytest.raises(ValidationError, match="finished_at"):
        Telemetry(
            architecture_name="x",
            architecture_version="0.1.0",
            prompt_versions={},
            modes=Modes(),
            started_at=datetime(2026, 5, 13, 10, 5, tzinfo=UTC),
            finished_at=datetime(2026, 5, 13, 10, 0, tzinfo=UTC),
            duration_seconds=0.0,
            total_input_tokens=0,
            total_output_tokens=0,
            total_cost_usd=0.0,
            agent_call_count=0,
            tool_call_count=0,
        )


def test_citation_count_without_source_rejected() -> None:
    """citation_count without a source is invalid."""
    with pytest.raises(ValidationError, match="both be set or both None"):
        _paper("1706.03762", "era1", citation_count=100, citation_source=None)


def test_citation_source_without_count_rejected() -> None:
    """citation_source without a count is invalid."""
    with pytest.raises(ValidationError, match="both be set or both None"):
        _paper("1706.03762", "era1", citation_count=None, citation_source="openalex")


def test_citation_both_set_accepted() -> None:
    """Both citation fields set together is the happy-path."""
    p = _paper("1706.03762", "era1", citation_count=100000, citation_source="openalex")
    assert p.citation_count == 100000
    assert p.citation_source == "openalex"


def test_paper_count_above_12_rejected() -> None:
    """Deliverable rejects more than 12 papers."""
    ids = _arxiv_ids(13)
    papers = [_paper(ids[i], era_id="era1") for i in range(13)]
    with pytest.raises(ValidationError):
        Deliverable(
            topic="t",
            overall_summary="s",
            papers=papers,
            timeline=[_era("era1", paper_ids=ids)],
        )


def test_modes_target_paper_count_band() -> None:
    """Modes.target_paper_count must fall within [8, 12]."""
    with pytest.raises(ValidationError):
        Modes(target_paper_count=7)
    with pytest.raises(ValidationError):
        Modes(target_paper_count=13)
    assert Modes(target_paper_count=8).target_paper_count == 8
    assert Modes(target_paper_count=12).target_paper_count == 12


def test_modes_defaults() -> None:
    """Modes defaults: tools on, target_paper_count=10."""
    m = Modes()
    assert m.tools is True
    assert m.target_paper_count == 10


def test_arxiv_id_pattern_enforced() -> None:
    """Malformed arxiv ids are rejected."""
    with pytest.raises(ValidationError):
        _paper("bad-id", "era1")
    with pytest.raises(ValidationError):
        _paper("hep-th/9901001", "era1")  # pre-2007 format intentionally rejected
    # Versioned ids are accepted.
    p = _paper("1706.03762v2", "era1")
    assert p.arxiv_id == "1706.03762v2"


def test_confidence_band_enforced() -> None:
    """Confidence must be in [0, 1]."""
    with pytest.raises(ValidationError):
        _paper("1706.03762", "era1", confidence=-0.1)
    with pytest.raises(ValidationError):
        _paper("1706.03762", "era1", confidence=1.1)


def test_extra_fields_forbidden() -> None:
    """A typo in a field name surfaces as a validation error, not silent drop."""
    with pytest.raises(ValidationError, match="Extra inputs"):
        _paper("1706.03762", "era1", confidance=0.5)  # typo: should be 'confidence'


# ───── Telemetry sub-models (smoke construction) ────────────────────────


def test_telemetry_sub_models_construct() -> None:
    """The telemetry sub-models all accept reasonable inputs."""
    tool_call = ToolCallRecord(
        tool_name="arxiv_search",
        tool_input={"query": "attention"},
        tool_output_summary="47 results",
        duration_ms=320,
    )
    step = TraceStep(
        step_index=0,
        agent_name="researcher",
        prompt_name="researcher",
        prompt_version="v1",
        model="claude-haiku-4-5",
        started_at=datetime(2026, 5, 13, 10, 0, tzinfo=UTC),
        duration_ms=1500,
        input_tokens=200,
        output_tokens=80,
        cost_usd=0.001,
        input_summary="topic=attention",
        output_summary="selected 47 candidates",
        tool_calls=[tool_call],
        decision="forward to synthesizer",
    )
    err = ErrorRecord(category="api", message="429 rate limited", recovered=True)
    cand = CandidateRecord(
        arxiv_id="1706.03762",
        title="Attention Is All You Need",
        verdict="included",
        reason="seminal text",
    )
    unres = UnresolvedLookup(
        kind="citation_lookup",
        query="2410.99999",
        reason="no record at openalex",
        attempted_at=datetime(2026, 5, 13, 10, 1, tzinfo=UTC),
    )

    tel = Telemetry(
        architecture_name="arch_01",
        architecture_version="0.1.0",
        prompt_versions={"researcher": "v1"},
        modes=Modes(),
        started_at=datetime(2026, 5, 13, 10, 0, tzinfo=UTC),
        finished_at=datetime(2026, 5, 13, 10, 5, tzinfo=UTC),
        duration_seconds=300.0,
        total_input_tokens=200,
        total_output_tokens=80,
        total_cost_usd=0.001,
        agent_call_count=1,
        tool_call_count=1,
        trace=[step],
        errors=[err],
        candidates=[cand],
        unresolved_lookups=[unres],
    )
    assert len(tel.trace) == 1
    assert tel.errors[0].recovered is True
    assert tel.candidates[0].verdict == "included"
    assert tel.unresolved_lookups[0].kind == "citation_lookup"
