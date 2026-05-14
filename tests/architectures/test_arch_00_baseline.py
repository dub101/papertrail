"""Tests for ``BaselineArchitecture`` (arch_00).

The architecture is deterministic and Anthropic-free, so unit testing is
unusually simple: mock ``arxiv_search`` with a stable list of ``ArxivPaper``s
and assert the resulting ``BenchmarkResult`` has the expected shape.

Two correctness contracts are exercised in dedicated tests:

* **Zero Claude tokens** — checked at both the runtime level
  (telemetry totals == 0, agent_call_count == 0) and the static level
  (AST walk of ``baseline.py`` never imports anything starting with
  ``anthropic``). The static check is belt-and-braces against a future
  accidental ``from anthropic import ...`` that slips through review.

* **Strict era partition** — every paper appears in exactly one era; every
  era has ≥1 paper. The pydantic validator on ``Deliverable`` catches
  violations at construction, but we also walk the structure ourselves so
  the failure mode is "test name says what's wrong" rather than "pydantic
  ValidationError somewhere".
"""

from __future__ import annotations

import ast
from contextlib import AbstractContextManager
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import HttpUrl

from papertrail.architectures.arch_00_baseline import (
    BaselineArchitecture,
    BaselineTooFewResultsError,
)
from papertrail.architectures.arch_00_baseline import baseline as baseline_module
from papertrail.benchmark import Modes
from papertrail.tools.arxiv import ArxivPaper

# ───── Test helpers ─────────────────────────────────────────────────────


def _make_paper(
    idx: int,
    year: int = 2020,
    *,
    arxiv_id: str | None = None,
    authors: list[str] | None = None,
    categories: list[str] | None = None,
    abstract: str | None = None,
) -> ArxivPaper:
    """Build a valid ``ArxivPaper`` for tests; override fields as needed.

    Default arxiv_id format ``"1706.{idx:05d}"`` matches the post-2007 regex
    that PaperEntry enforces, so the default paper survives the baseline's
    filter step. Pass an override for negative tests.
    """
    return ArxivPaper(
        arxiv_id=arxiv_id if arxiv_id is not None else f"1706.{idx:05d}",
        title=f"Paper {idx} title",
        authors=authors if authors is not None else [f"Author{idx}, A."],
        categories=categories if categories is not None else ["cs.LG"],
        abstract=(
            abstract
            if abstract is not None
            else (
                f"First sentence about paper {idx}. "
                f"Second sentence with more detail. "
                f"Third sentence that should be trimmed."
            )
        ),
        entry_url=HttpUrl(f"https://arxiv.org/abs/1706.{idx:05d}"),
        pdf_url=HttpUrl(f"https://arxiv.org/pdf/1706.{idx:05d}"),
        published=datetime(year, 1, 1, tzinfo=UTC),
        updated=datetime(year, 1, 1, tzinfo=UTC),
    )


def _papers_spanning_years(count: int, start_year: int, end_year: int) -> list[ArxivPaper]:
    """Distribute ``count`` papers evenly across ``[start_year, end_year]``."""
    if count <= 0:
        return []
    span = max(end_year - start_year, 0)
    out: list[ArxivPaper] = []
    for i in range(count):
        # Linear interpolation; both endpoints inclusive when count >= 2.
        year = start_year + (span * i // max(count - 1, 1)) if count > 1 else start_year
        out.append(_make_paper(idx=i, year=year))
    return out


def _patch_arxiv_search(papers: list[ArxivPaper]) -> AbstractContextManager[AsyncMock]:
    """Patch ``arxiv_search`` *as bound in the baseline module*.

    Patching the symbol at its definition site
    (``papertrail.tools.arxiv.arxiv_search``) would NOT affect the already
    imported reference inside ``baseline.py``. We patch where it's looked
    up, not where it's defined. Typed as ``AbstractContextManager`` (the
    public protocol) so mypy can verify ``with _patch_arxiv_search(...):``
    works without us touching ``unittest.mock._patch`` directly.
    """
    return patch(
        "papertrail.architectures.arch_00_baseline.baseline.arxiv_search",
        new=AsyncMock(return_value=papers),
    )


# ───── ClassVar contract ────────────────────────────────────────────────


def test_classvars_are_declared() -> None:
    """``Architecture``'s three required ClassVars are non-empty strings."""
    assert BaselineArchitecture.name == "arch_00_baseline"
    assert isinstance(BaselineArchitecture.version, str) and BaselineArchitecture.version
    assert isinstance(BaselineArchitecture.description, str) and BaselineArchitecture.description


# ───── Happy path ───────────────────────────────────────────────────────


async def test_run_happy_path_produces_valid_result() -> None:
    """A 10-paper search yields a valid 10-paper BenchmarkResult with ≤3 eras."""
    papers = _papers_spanning_years(count=10, start_year=2018, end_year=2024)
    arch = BaselineArchitecture()

    with _patch_arxiv_search(papers):
        result = await arch.run("transformer attention")

    assert result.deliverable.topic == "transformer attention"
    assert len(result.deliverable.papers) == 10
    assert 1 <= len(result.deliverable.timeline) <= 3
    # Every paper has all five summary fields populated.
    for p in result.deliverable.papers:
        assert p.summary_about
        assert p.summary_relation_to_topic
        assert p.summary_problem
        assert p.summary_approach
        assert p.summary_impact


# ───── Zero-Claude-tokens contract ──────────────────────────────────────


async def test_zero_tokens_and_no_agent_calls() -> None:
    """The baseline burns no Claude tokens and records no agent invocations."""
    papers = _papers_spanning_years(count=10, start_year=2018, end_year=2024)
    arch = BaselineArchitecture()

    with _patch_arxiv_search(papers):
        result = await arch.run("topic")

    t = result.telemetry
    assert t.total_input_tokens == 0
    assert t.total_output_tokens == 0
    assert t.total_cost_usd == 0.0
    assert t.agent_call_count == 0
    assert t.tool_call_count == 1
    assert t.trace == []
    assert t.errors == []
    assert t.candidates == []
    assert t.unresolved_lookups == []
    assert t.prompt_versions == {}


def test_baseline_module_does_not_import_anthropic() -> None:
    """Static AST check: ``baseline.py`` never imports ``anthropic``.

    Belt-and-braces backup for the runtime token-count check. A future
    accidental ``import anthropic`` would still produce zero tokens if the
    import is unused, but it would betray intent — the baseline is meant
    to have *no* dependency on the SDK at all.
    """
    module_path = Path(baseline_module.__file__)
    tree = ast.parse(module_path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert not (node.module or "").startswith("anthropic"), (
                f"baseline.py imports from {node.module}"
            )
        elif isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith("anthropic"), f"baseline.py imports {alias.name}"


# ───── Filtering and trimming ───────────────────────────────────────────


async def test_too_few_papers_raises() -> None:
    """When arxiv returns < 8 usable papers, the architecture raises."""
    papers = _papers_spanning_years(count=5, start_year=2020, end_year=2022)
    arch = BaselineArchitecture()

    with _patch_arxiv_search(papers), pytest.raises(BaselineTooFewResultsError) as exc_info:
        await arch.run("narrow topic")

    assert exc_info.value.topic == "narrow topic"
    assert exc_info.value.found == 5


async def test_pre_2007_ids_filtered_out() -> None:
    """A pre-2007 arxiv id (``hep-th/...``) is dropped, not crashed on."""
    papers = _papers_spanning_years(count=10, start_year=2018, end_year=2024)
    # Replace one paper's id with the pre-2007 slash-separated form. The
    # ArxivPaper schema accepts it (faithful echo), but PaperEntry rejects
    # it — the baseline filter is what saves the run.
    papers[3] = _make_paper(idx=3, year=2005, arxiv_id="hep-th/0501001")
    arch = BaselineArchitecture()

    with _patch_arxiv_search(papers):
        result = await arch.run("topic")

    # 9 usable papers after filtering one out; with 9 ≥ 8 we still validate.
    arxiv_ids = {p.arxiv_id for p in result.deliverable.papers}
    assert "hep-th/0501001" not in arxiv_ids
    assert len(result.deliverable.papers) == 9


async def test_overfetch_then_trim_to_target() -> None:
    """When arxiv returns more than target_paper_count, we trim to target."""
    papers = _papers_spanning_years(count=20, start_year=2017, end_year=2024)
    arch = BaselineArchitecture()

    with _patch_arxiv_search(papers):
        result = await arch.run("topic")

    assert len(result.deliverable.papers) == 10  # Modes default


async def test_modes_target_paper_count_respected() -> None:
    """Custom ``Modes.target_paper_count`` controls the deliverable size."""
    papers = _papers_spanning_years(count=15, start_year=2017, end_year=2024)
    arch = BaselineArchitecture()

    with _patch_arxiv_search(papers):
        result = await arch.run("topic", modes=Modes(target_paper_count=12))

    assert len(result.deliverable.papers) == 12


# ───── Era partitioning ─────────────────────────────────────────────────


async def test_era_partition_is_strict() -> None:
    """Every paper belongs to exactly one era; no era is empty."""
    papers = _papers_spanning_years(count=10, start_year=2018, end_year=2024)
    arch = BaselineArchitecture()

    with _patch_arxiv_search(papers):
        result = await arch.run("topic")

    deliverable = result.deliverable
    seen_ids: set[str] = set()
    for era in deliverable.timeline:
        assert len(era.paper_ids) >= 1, f"era {era.era_id} is empty"
        for pid in era.paper_ids:
            assert pid not in seen_ids, f"{pid} appears in multiple eras"
            seen_ids.add(pid)
    assert seen_ids == {p.arxiv_id for p in deliverable.papers}


async def test_three_eras_when_papers_span_multiple_years() -> None:
    """10 papers across 2018-2024 → 3 eras (the cap)."""
    papers = _papers_spanning_years(count=10, start_year=2018, end_year=2024)
    arch = BaselineArchitecture()

    with _patch_arxiv_search(papers):
        result = await arch.run("topic")

    assert len(result.deliverable.timeline) == 3


async def test_one_era_per_paper_when_papers_share_year() -> None:
    """All papers in the same year still partition into ≤3 eras with ≥1 paper each."""
    # 10 papers all in 2020 — era boundaries by date will be degenerate
    # (same start/end year) but the partition still has to satisfy the
    # ≥1-paper-per-era invariant.
    papers = [_make_paper(idx=i, year=2020) for i in range(10)]
    arch = BaselineArchitecture()

    with _patch_arxiv_search(papers):
        result = await arch.run("topic")

    timeline = result.deliverable.timeline
    assert 1 <= len(timeline) <= 3
    for era in timeline:
        assert era.date_range_start.year == 2020
        assert era.date_range_end is not None and era.date_range_end.year == 2020
        # Single-year era name is "2020", not "2020-2020".
        assert era.name == "2020"


# ───── Per-field policy ─────────────────────────────────────────────────


async def test_disclaimer_fields_exact_match() -> None:
    """The three disclaimer summary fields match the documented constants."""
    papers = _papers_spanning_years(count=10, start_year=2018, end_year=2024)
    arch = BaselineArchitecture()

    with _patch_arxiv_search(papers):
        result = await arch.run("topic")

    for p in result.deliverable.papers:
        assert p.summary_problem == baseline_module._DISCLAIMER_PROBLEM
        assert p.summary_approach == baseline_module._DISCLAIMER_APPROACH
        assert p.summary_impact == baseline_module._DISCLAIMER_IMPACT
    for era in result.deliverable.timeline:
        assert era.narrative == baseline_module._DISCLAIMER_ERA_NARRATIVE


async def test_summary_about_contains_title_and_abstract_slice() -> None:
    """``summary_about`` includes the paper title and abstract content."""
    papers = _papers_spanning_years(count=10, start_year=2018, end_year=2024)
    arch = BaselineArchitecture()

    with _patch_arxiv_search(papers):
        result = await arch.run("topic")

    first = result.deliverable.papers[0]
    assert first.title in first.summary_about
    # Default fixture's abstract starts with "First sentence about ..."
    assert "First sentence" in first.summary_about


async def test_summary_relation_uses_topic_and_rank() -> None:
    """``summary_relation_to_topic`` mentions the topic and a 1-indexed rank."""
    papers = _papers_spanning_years(count=10, start_year=2018, end_year=2024)
    arch = BaselineArchitecture()

    with _patch_arxiv_search(papers):
        result = await arch.run("retrieval augmented generation")

    first = result.deliverable.papers[0]
    assert "retrieval augmented generation" in first.summary_relation_to_topic
    assert "rank 1 of 10" in first.summary_relation_to_topic


async def test_overall_summary_self_discloses_as_baseline() -> None:
    """``overall_summary`` mentions the topic and identifies as baseline output."""
    papers = _papers_spanning_years(count=10, start_year=2018, end_year=2024)
    arch = BaselineArchitecture()

    with _patch_arxiv_search(papers):
        result = await arch.run("transformer attention")

    s = result.deliverable.overall_summary
    assert "transformer attention" in s
    assert "baseline" in s.lower()


async def test_confidence_uniform_05() -> None:
    """Every paper has confidence == 0.5 (no rank-decay leakage)."""
    papers = _papers_spanning_years(count=10, start_year=2018, end_year=2024)
    arch = BaselineArchitecture()

    with _patch_arxiv_search(papers):
        result = await arch.run("topic")

    for p in result.deliverable.papers:
        assert p.confidence == 0.5


async def test_citation_fields_are_none() -> None:
    """``citation_count`` and ``citation_source`` are both ``None``."""
    papers = _papers_spanning_years(count=10, start_year=2018, end_year=2024)
    arch = BaselineArchitecture()

    with _patch_arxiv_search(papers):
        result = await arch.run("topic")

    for p in result.deliverable.papers:
        assert p.citation_count is None
        assert p.citation_source is None


# ───── Provenance helpers ───────────────────────────────────────────────


def test_papertrail_version_is_a_string() -> None:
    """``_papertrail_version`` returns a non-empty version string."""
    v = baseline_module._papertrail_version()
    assert isinstance(v, str)
    assert v  # Either the metadata version or the in-module fallback.


def test_git_sha_returns_a_string() -> None:
    """``_git_sha`` returns a non-empty string (a sha or ``"unknown"``)."""
    sha = baseline_module._git_sha()
    assert isinstance(sha, str)
    assert sha  # Either a hex sha or "unknown"; never empty.
