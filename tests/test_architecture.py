"""Tests for the Architecture abstract base class.

The ABC enforces its contract on two axes:

- abstract method (``run``) must be implemented — checked at instantiation
- ClassVars (``name``/``version``/``description``) must be declared — checked
  at class-creation time via ``__init_subclass__``

Both axes are exercised below.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, date, datetime
from typing import ClassVar

import pytest
from pydantic import HttpUrl

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


def test_cannot_instantiate_abstract_architecture() -> None:
    """The bare ABC cannot be instantiated because ``run`` is abstract."""
    with pytest.raises(TypeError):
        Architecture()  # type: ignore[abstract]


def test_subclass_missing_classvars_rejected_at_class_creation() -> None:
    """A concrete subclass that forgets ClassVars fails at *class definition*.

    This is the value-add of ``__init_subclass__``: the error fires on
    import, not on first instantiation, so a misconfigured architecture
    never makes it into the registry.
    """
    with pytest.raises(TypeError, match="must declare ClassVar"):

        class Broken(Architecture):
            # No name/version/description declared.
            async def run(
                self,
                topic: str,
                *,
                prompt_versions: dict[str, str] | None = None,
                modes: Modes | None = None,
            ) -> BenchmarkResult:
                raise NotImplementedError


def test_subclass_with_empty_classvar_rejected() -> None:
    """A subclass with an empty-string ClassVar is rejected.

    Empty strings would silently break logging/registry lookups; the
    __init_subclass__ check treats them as a configuration error.
    """
    with pytest.raises(TypeError, match="non-empty"):

        class EmptyName(Architecture):
            name: ClassVar[str] = ""
            version: ClassVar[str] = "0.1.0"
            description: ClassVar[str] = "x"

            async def run(
                self,
                topic: str,
                *,
                prompt_versions: dict[str, str] | None = None,
                modes: Modes | None = None,
            ) -> BenchmarkResult:
                raise NotImplementedError


def test_subclass_missing_run_cannot_be_instantiated() -> None:
    """A subclass that declares ClassVars but skips ``run`` is still abstract.

    The __init_subclass__ check is skipped (the class still has abstract
    methods, so it's an 'intermediate' from the ABC's point of view), and
    instantiation fails with the usual ABC error.
    """

    class StillAbstract(Architecture):
        name: ClassVar[str] = "test"
        version: ClassVar[str] = "0.1.0"
        description: ClassVar[str] = "Test"
        # No run() implementation.

    with pytest.raises(TypeError):
        StillAbstract()  # type: ignore[abstract]


def _make_minimal_result(
    topic: str,
    architecture_name: str,
    architecture_version: str,
    modes: Modes,
) -> BenchmarkResult:
    """Helper: assemble a minimal valid BenchmarkResult for the fake arch."""
    ids = [f"1706.{i:05d}" for i in range(8)]
    papers = [
        PaperEntry(
            arxiv_id=ids[i],
            title=f"Paper {i}",
            authors=["Doe, J."],
            published_date=date(2020, 1, 1),
            url=HttpUrl(f"https://arxiv.org/abs/{ids[i]}"),
            era_id="era1",
            summary_about="x",
            summary_relation_to_topic="x",
            summary_problem="x",
            summary_approach="x",
            summary_impact="x",
            confidence=0.5,
        )
        for i in range(8)
    ]
    return BenchmarkResult(
        deliverable=Deliverable(
            topic=topic,
            overall_summary="summary",
            papers=papers,
            timeline=[
                TimelineEra(
                    era_id="era1",
                    name="Era 1",
                    date_range_start=date(2017, 1, 1),
                    date_range_end=None,
                    narrative="n",
                    paper_ids=ids,
                ),
            ],
        ),
        telemetry=Telemetry(
            architecture_name=architecture_name,
            architecture_version=architecture_version,
            prompt_versions={},
            modes=modes,
            started_at=datetime(2026, 5, 13, 10, 0, tzinfo=UTC),
            finished_at=datetime(2026, 5, 13, 10, 1, tzinfo=UTC),
            duration_seconds=60.0,
            total_input_tokens=0,
            total_output_tokens=0,
            total_cost_usd=0.0,
            agent_call_count=0,
            tool_call_count=0,
        ),
        provenance=Provenance(
            schema_version=SCHEMA_VERSION,
            papertrail_version="0.1.0",
            git_sha="test",
            created_at=datetime(2026, 5, 13, 10, 1, tzinfo=UTC),
        ),
    )


def test_valid_subclass_runs_and_returns_benchmark_result() -> None:
    """A fully-implemented subclass instantiates and ``run()`` returns a result."""

    class Fake(Architecture):
        name: ClassVar[str] = "arch_test_fake"
        version: ClassVar[str] = "0.1.0"
        description: ClassVar[str] = "Fake architecture for ABC tests."

        async def run(
            self,
            topic: str,
            *,
            prompt_versions: dict[str, str] | None = None,
            modes: Modes | None = None,
        ) -> BenchmarkResult:
            return _make_minimal_result(
                topic=topic,
                architecture_name=self.name,
                architecture_version=self.version,
                modes=modes or Modes(),
            )

    fake = Fake()
    result = asyncio.run(fake.run("test topic"))
    assert result.deliverable.topic == "test topic"
    assert result.telemetry.architecture_name == "arch_test_fake"
    assert result.telemetry.architecture_version == "0.1.0"
    # When no modes are passed, defaults flow through.
    assert result.telemetry.modes.tools is True
    assert result.telemetry.modes.target_paper_count == 10


def test_valid_subclass_respects_passed_modes() -> None:
    """Modes passed to ``run()`` reach the resulting Telemetry."""

    class Fake(Architecture):
        name: ClassVar[str] = "arch_test_fake_2"
        version: ClassVar[str] = "0.1.0"
        description: ClassVar[str] = "Fake architecture."

        async def run(
            self,
            topic: str,
            *,
            prompt_versions: dict[str, str] | None = None,
            modes: Modes | None = None,
        ) -> BenchmarkResult:
            return _make_minimal_result(
                topic=topic,
                architecture_name=self.name,
                architecture_version=self.version,
                modes=modes or Modes(),
            )

    fake = Fake()
    custom = Modes(tools=False, target_paper_count=12)
    result = asyncio.run(fake.run("t", modes=custom))
    assert result.telemetry.modes.tools is False
    assert result.telemetry.modes.target_paper_count == 12
