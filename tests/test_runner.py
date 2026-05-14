"""Tests for the benchmark runner.

These tests use a tiny in-test ``FakeArchitecture`` that returns a hand-built
``BenchmarkResult`` so we don't hit arxiv. The DryRunEvaluator covers the
evaluator side without any LLM calls. Together that makes every test in
this file zero-cost and offline.
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path
from typing import ClassVar
from uuid import uuid4

import pytest

from papertrail.architecture import Architecture
from papertrail.benchmark import (
    SCHEMA_VERSION,
    BenchmarkResult,
    Deliverable,
    Modes,
    PaperEntry,
    Provenance,
    Telemetry,
    TimelineEra,
)
from papertrail.evaluator import DryRunEvaluator
from papertrail.runner import (
    ARCHITECTURES,
    _filename_for_run,
    _resolve_architecture,
    run_benchmark,
)

# ───── Fixtures ─────────────────────────────────────────────────────────


def _make_result(arch_name: str = "fake_arch") -> BenchmarkResult:
    """Build a minimal-but-valid BenchmarkResult by hand."""
    papers = [
        PaperEntry(
            arxiv_id=f"1706.0376{i}",
            title=f"Paper {i}",
            authors=["Doe, J."],
            published_date=date(2017, 6, (i % 28) + 1),
            url="https://arxiv.org/abs/1706.03760",
            citation_count=None,
            citation_source=None,
            era_id="era_1",
            summary_about="about",
            summary_relation_to_topic="rel",
            summary_problem="prob",
            summary_approach="approach",
            summary_impact="impact",
            confidence=0.5,
        )
        for i in range(8)
    ]
    timeline = [
        TimelineEra(
            era_id="era_1",
            name="2017",
            date_range_start=date(2017, 6, 1),
            date_range_end=date(2017, 6, 30),
            narrative="one era narrative",
            paper_ids=[p.arxiv_id for p in papers],
        )
    ]
    started = datetime(2026, 5, 14, 10, 0, 0, tzinfo=UTC)
    finished = datetime(2026, 5, 14, 10, 0, 1, tzinfo=UTC)
    return BenchmarkResult(
        deliverable=Deliverable(
            topic="self-attention",
            overall_summary="overall summary",
            papers=papers,
            timeline=timeline,
        ),
        telemetry=Telemetry(
            architecture_name=arch_name,
            architecture_version="0.1.0",
            prompt_versions={},
            modes=Modes(),
            started_at=started,
            finished_at=finished,
            duration_seconds=1.0,
            total_input_tokens=0,
            total_output_tokens=0,
            total_cost_usd=0.0,
            agent_call_count=0,
            tool_call_count=1,
        ),
        provenance=Provenance(
            run_id=uuid4(),
            schema_version=SCHEMA_VERSION,
            papertrail_version="0.1.0",
            git_sha="abc123def456",
            created_at=finished,
        ),
    )


class FakeArchitecture(Architecture):
    """Architecture that returns a canned BenchmarkResult, no network calls."""

    name: ClassVar[str] = "fake_arch"
    version: ClassVar[str] = "0.1.0"
    description: ClassVar[str] = "test-only architecture"

    def __init__(self, result: BenchmarkResult | None = None) -> None:
        # Allow injecting a specific result for tests that care about its
        # contents; otherwise build a default one on demand in ``run``.
        self._result = result

    async def run(
        self,
        topic: str,
        *,
        prompt_versions: dict[str, str] | None = None,
        modes: Modes | None = None,
    ) -> BenchmarkResult:
        # Silence unused-param warnings without changing the signature.
        _ = (topic, prompt_versions, modes)
        return self._result or _make_result(arch_name=self.name)


class ExplodingArchitecture(Architecture):
    """Architecture that always raises, for the propagation tests."""

    name: ClassVar[str] = "exploding_arch"
    version: ClassVar[str] = "0.1.0"
    description: ClassVar[str] = "raises on every run"

    async def run(
        self,
        topic: str,
        *,
        prompt_versions: dict[str, str] | None = None,
        modes: Modes | None = None,
    ) -> BenchmarkResult:
        _ = (topic, prompt_versions, modes)
        raise RuntimeError("boom")


# ───── Helpers ──────────────────────────────────────────────────────────


def test_filename_for_run_is_sortable_and_identifiable() -> None:
    # The format ``<ts>_<arch>_<short_id>.json`` must be lex-sortable in
    # the same order as the runs occurred — that's how ``ls`` becomes a
    # chronological run list with zero tooling.
    r = _make_result(arch_name="arch_00_baseline")
    name = _filename_for_run(r)
    assert name.endswith(".json")
    assert "arch_00_baseline" in name
    # UTC timestamps with no colons sort lexically the same as chronologically.
    assert name.startswith("20260514T100001Z")


def test_resolve_architecture_accepts_string_name() -> None:
    arch = _resolve_architecture("arch_00_baseline")
    assert arch.name == "arch_00_baseline"


def test_resolve_architecture_accepts_instance() -> None:
    instance = FakeArchitecture()
    assert _resolve_architecture(instance) is instance


def test_resolve_architecture_rejects_unknown_name() -> None:
    with pytest.raises(ValueError) as exc:
        _resolve_architecture("arch_99_chimera")
    assert "arch_99_chimera" in str(exc.value)
    assert "Available" in str(exc.value)


# ───── run_benchmark ────────────────────────────────────────────────────


async def test_run_benchmark_with_dry_run_evaluator(tmp_path: Path) -> None:
    """Full wire-up against a fake arch + DryRunEvaluator. Zero network calls."""
    arch = FakeArchitecture()
    evaluator = DryRunEvaluator()

    result, score, path = await run_benchmark(
        topic="self-attention",
        architecture=arch,
        evaluator=evaluator,
        output_dir=tmp_path,
    )

    # In-memory objects came back populated.
    assert result.deliverable.topic == "self-attention"
    assert score is not None
    assert score.verdict.overall == 0.123  # the dry-run sentinel
    assert score.evaluator_model == "dry-run-no-llm"

    # File was written, has the expected shape, and round-trips.
    assert path.exists()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert "result" in payload
    assert "score" in payload
    assert payload["result"]["deliverable"]["topic"] == "self-attention"
    assert payload["score"]["verdict"]["overall"] == 0.123


async def test_run_benchmark_with_no_evaluator_writes_null_score(
    tmp_path: Path,
) -> None:
    _result, score, path = await run_benchmark(
        topic="t",
        architecture=FakeArchitecture(),
        evaluator=None,
        output_dir=tmp_path,
    )

    assert score is None
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["score"] is None
    # The result side is still fully populated.
    assert payload["result"]["telemetry"]["architecture_name"] == "fake_arch"


async def test_run_benchmark_creates_output_dir_if_missing(tmp_path: Path) -> None:
    nested = tmp_path / "does" / "not" / "exist" / "yet"
    assert not nested.exists()

    _, _, path = await run_benchmark(
        topic="t",
        architecture=FakeArchitecture(),
        evaluator=None,
        output_dir=nested,
    )

    assert nested.exists()
    assert path.parent == nested


async def test_run_benchmark_propagates_architecture_failure(tmp_path: Path) -> None:
    # An architecture-side error must not be swallowed; persistence does
    # not happen and the exception surfaces unchanged so the caller can
    # debug the root cause.
    with pytest.raises(RuntimeError, match="boom"):
        await run_benchmark(
            topic="t",
            architecture=ExplodingArchitecture(),
            evaluator=None,
            output_dir=tmp_path,
        )
    # And no file was written.
    assert not list(tmp_path.iterdir())


async def test_run_benchmark_resolves_string_name(tmp_path: Path) -> None:
    # The string-name path goes through ARCHITECTURES; arch_00_baseline is
    # in there, but it would call arxiv. Patch the registry to point at our
    # fake so we cover the string path without touching the network.
    original = ARCHITECTURES.copy()
    try:
        ARCHITECTURES.clear()
        ARCHITECTURES["fake_arch"] = FakeArchitecture
        result, _, _ = await run_benchmark(
            topic="t",
            architecture="fake_arch",
            evaluator=None,
            output_dir=tmp_path,
        )
        assert result.telemetry.architecture_name == "fake_arch"
    finally:
        # Restore so other tests aren't poisoned.
        ARCHITECTURES.clear()
        ARCHITECTURES.update(original)


async def test_run_benchmark_returns_filename_using_run_metadata(
    tmp_path: Path,
) -> None:
    # The filename must come from the run's own provenance/telemetry so
    # historical files remain interpretable independent of when they were
    # listed or read back.
    expected_result = _make_result(arch_name="fake_arch")
    arch = FakeArchitecture(result=expected_result)

    _result, _score, path = await run_benchmark(
        topic="t",
        architecture=arch,
        evaluator=None,
        output_dir=tmp_path,
    )
    assert path.name == _filename_for_run(expected_result)
