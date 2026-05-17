"""BaselineArchitecture — the deterministic, no-LLM floor.

What this module does (in one paragraph):
    Call ``arxiv_search`` once for the topic, filter to papers whose ids fit
    the post-2007 numeric scheme, trim to ``Modes.target_paper_count``, sort
    by publication date and partition into at most three contiguous eras,
    and assemble a fully-valid ``BenchmarkResult``. Five summary fields per
    paper are populated according to a per-field heuristic-or-disclaimer
    policy: ``summary_about`` and ``summary_relation_to_topic`` carry honest
    derivable content (title + abstract slice / arxiv rank within the
    relevance search); ``summary_problem``, ``summary_approach`` and
    ``summary_impact`` carry distinct disclaimer constants. Same shape for
    eras: ``era.narrative`` is a disclaimer, dates/ids are derived.

Why three disclaimer constants and not one:
    The Evaluator scores each summary dimension independently. Distinct
    strings let it attribute the floor score to the specific dimension
    rather than a single "this architecture didn't try" signal.

What this module is *not*:
    - Not an LLM caller. Zero Claude tokens. The unit-test suite verifies
      this both at the API level (telemetry totals == 0) and at the static
      level (no ``anthropic`` import anywhere in this file's source).
    - Not a citation lookup. ``citation_count`` / ``citation_source`` are
      both ``None`` — arch_00 has no honest way to count.
    - Not a retrying client. arxiv's 429-only retry happens inside
      ``arxiv_search``; any architecture-level resilience is intentionally
      absent here because there is no agent loop to recover into.

Cert mapping: none. arch_00 is the floor by design. The downstream
architectures (arch_01 onward) are the cert-relevant ones; arch_00 exists
so we can quantify what those architectures buy us.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import ClassVar

from papertrail.architecture import Architecture
from papertrail.benchmark import (
    BenchmarkResult,
    Deliverable,
    Modes,
    PaperEntry,
    Telemetry,
    TimelineEra,
)
from papertrail.provenance import build_provenance
from papertrail.tools.arxiv import ArxivPaper, arxiv_search

# ───── Constants ────────────────────────────────────────────────────────

# Per the Phase-3 decision: distinct disclaimer strings per dimension so the
# Evaluator can attribute its low scores to specific fields. Do NOT collapse
# these into one shared constant — the asymmetry is the point.
_DISCLAIMER_PROBLEM = "Baseline architecture — no problem analysis."
_DISCLAIMER_APPROACH = "Baseline architecture — no approach analysis."
_DISCLAIMER_IMPACT = "Baseline architecture — no impact analysis."
_DISCLAIMER_ERA_NARRATIVE = "Baseline architecture — no era narrative."

# Honest "we have no opinion" confidence. Rank-decay would leak arxiv's
# relevance signal into our confidence field, which would be a small lie.
_BASELINE_CONFIDENCE = 0.5

# Over-fetch from arxiv so the filter step (post-2007 ids only, non-empty
# abstracts) doesn't drop us below the 8-paper schema floor for typical
# topics. The number is generous; arxiv_search's MAX_RESULTS_CEILING is 100.
_OVERFETCH_SIZE = 20

# Maximum number of eras the baseline ever produces. Three buckets is the
# smallest count that gives the deliverable temporal structure without
# inflating the partitioning logic. With <3 papers we degrade to fewer
# buckets automatically.
_MAX_ERAS = 3

# Post-2007 numeric arxiv id scheme. The benchmark schema enforces this at
# PaperEntry construction; we filter here so that one stray pre-2007 result
# from arxiv doesn't crash the whole run.
_ARXIV_ID_RE = re.compile(r"^\d{4}\.\d{4,5}(v\d+)?$")

# Provenance helpers moved to ``papertrail.provenance`` once arch_01
# appeared as a second consumer. ``build_provenance()`` is imported
# above and used directly in ``run()``.


# ───── Exceptions ───────────────────────────────────────────────────────


class BaselineTooFewResultsError(RuntimeError):
    """Raised when arxiv returns fewer than 8 usable papers for the topic.

    The benchmark schema enforces a deliverable size of 8-12 papers as a hard
    floor. If arxiv simply doesn't have that many results for the topic (or
    too many are pre-2007 and get filtered), the baseline can't honestly
    produce a valid result. We raise rather than pad — padding would mean
    fabricating papers, which violates the no-LLM contract twice over.
    """

    def __init__(self, topic: str, found: int) -> None:
        self.topic = topic
        self.found = found
        super().__init__(
            f"arxiv returned {found} usable papers for topic '{topic}'; "
            f"baseline requires at least 8"
        )


# ───── Internal dataclasses ─────────────────────────────────────────────


@dataclass(frozen=True)
class _EraSlice:
    """One era's worth of papers, computed by the date-partition step.

    Holds the original ``ArxivPaper`` objects (not yet mapped to
    ``PaperEntry``) so the mapper can still see the abstract text and other
    fields. Dataclass because this is internal plumbing — no validation,
    no JSON, just a typed tuple-with-names.
    """

    era_id: str
    era_name: str
    date_range_start: date
    date_range_end: date
    papers: tuple[ArxivPaper, ...]


# ───── Internal helpers ─────────────────────────────────────────────────


def _summary_about(paper: ArxivPaper) -> str:
    """Title plus the first two sentences of the abstract. Pure slicing.

    The sentence splitter is intentionally naive — splitting on
    ``[.!?]\\s+`` will misfire on "e.g." and "Dr." but that wonkiness is a
    feature, not a bug: the baseline is supposed to look like exactly what
    it is, a sliced abstract. The Evaluator can grade the wonkiness.
    """
    sentences = re.split(r"(?<=[.!?])\s+", paper.abstract.strip(), maxsplit=2)
    excerpt = " ".join(sentences[:2]).strip()
    return f"{paper.title}. {excerpt}".rstrip()


def _summary_relation(paper: ArxivPaper, rank: int, total: int, topic: str) -> str:
    """The one honest "why is this paper here" we can offer without an LLM.

    Tautological by design — the only relation a non-LLM baseline can claim
    is "arxiv's relevance scorer ranked it #N for this query". Naming that
    explicitly is more useful than fabricating semantic prose.
    """
    return f"Returned by arXiv relevance search for query '{topic}' (rank {rank} of {total})."


def _build_overall_summary(
    papers: list[ArxivPaper],
    topic: str,
) -> str:
    """Factual aggregate over the paper list. No synthesis.

    Mentions: paper count, year range, top categories (by frequency), top
    authors (by appearance count), and self-discloses as baseline output so
    the Evaluator isn't misled into grading it as a "real" summary.
    """
    years = sorted({p.published.year for p in papers})
    year_range = f"between {years[0]} and {years[-1]}" if len(years) > 1 else f"in {years[0]}"

    # Top categories by frequency. Each paper contributes each of its
    # categories once; ties broken by first-seen order (stable).
    cat_counts: dict[str, int] = {}
    for p in papers:
        for c in p.categories:
            cat_counts[c] = cat_counts.get(c, 0) + 1
    top_cats = sorted(cat_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:3]
    cats_str = ", ".join(c for c, _ in top_cats) if top_cats else "(none reported)"

    # Top authors by appearance. Same tie-break policy.
    author_counts: dict[str, int] = {}
    for p in papers:
        for a in p.authors:
            author_counts[a] = author_counts.get(a, 0) + 1
    top_authors = sorted(author_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:3]
    authors_str = ", ".join(f"{a} ({n})" for a, n in top_authors) if top_authors else "(none)"

    return (
        f"Returned {len(papers)} papers on '{topic}' via arXiv relevance search, "
        f"published {year_range} across categories {cats_str}. "
        f"Top author appearances: {authors_str}. "
        f"This deliverable is produced by the deterministic baseline (arch_00) "
        f"and contains no LLM-generated analysis; per-paper disclaimers mark "
        f"fields that require synthesis."
    )


def _partition_into_eras(papers: list[ArxivPaper]) -> list[_EraSlice]:
    """Sort papers by publication date and split into up to ``_MAX_ERAS`` buckets.

    Each bucket holds at least one paper; we never produce an empty era.
    Extra papers (when ``n`` doesn't divide evenly) land in the earlier
    buckets — arbitrary but deterministic. The schema validator on
    Deliverable verifies the partition is strict; if this function has a
    bug, ``BenchmarkResult(...)`` construction blows up loudly.
    """
    if not papers:
        raise ValueError("Cannot partition zero papers into eras")

    n = len(papers)
    k = min(_MAX_ERAS, n)

    # ``published`` is a timezone-aware datetime; sorting by it is stable
    # across runs because ties are resolved by arxiv_id order (insertion
    # order in the search result).
    by_date = sorted(papers, key=lambda p: p.published)

    # Equal-as-possible bucket sizes: the first ``n % k`` buckets get one
    # extra paper. For n=10, k=3 -> [4, 3, 3]; for n=8, k=3 -> [3, 3, 2].
    sizes = [n // k + (1 if i < n % k else 0) for i in range(k)]

    slices: list[_EraSlice] = []
    offset = 0
    for bucket_idx, size in enumerate(sizes):
        bucket = by_date[offset : offset + size]
        offset += size
        start_date = bucket[0].published.date()
        end_date = bucket[-1].published.date()
        era_name = (
            f"{start_date.year}-{end_date.year}"
            if start_date.year != end_date.year
            else str(start_date.year)
        )
        slices.append(
            _EraSlice(
                era_id=f"era_{bucket_idx + 1}",
                era_name=era_name,
                date_range_start=start_date,
                date_range_end=end_date,
                papers=tuple(bucket),
            )
        )
    return slices


def _is_usable(paper: ArxivPaper) -> bool:
    """Cheap filter: must satisfy PaperEntry's stricter arxiv_id regex.

    ``ArxivPaper.arxiv_id`` accepts anything non-empty (pre-2007 ids like
    ``hep-th/9901001`` pass), but ``PaperEntry.arxiv_id`` requires the
    post-2007 numeric scheme. We drop pre-2007 results here rather than
    letting PaperEntry construction raise — one stray old paper shouldn't
    take down the whole run.
    """
    return bool(_ARXIV_ID_RE.match(paper.arxiv_id))


def _arxiv_to_paper_entry(
    paper: ArxivPaper,
    rank: int,
    total: int,
    topic: str,
    era_id: str,
) -> PaperEntry:
    """Build one PaperEntry, mixing heuristic and disclaimer fields per policy."""
    return PaperEntry(
        arxiv_id=paper.arxiv_id,
        title=paper.title,
        authors=paper.authors,
        published_date=paper.published.date(),
        url=paper.entry_url,
        citation_count=None,
        citation_source=None,
        era_id=era_id,
        summary_about=_summary_about(paper),
        summary_relation_to_topic=_summary_relation(paper, rank, total, topic),
        summary_problem=_DISCLAIMER_PROBLEM,
        summary_approach=_DISCLAIMER_APPROACH,
        summary_impact=_DISCLAIMER_IMPACT,
        confidence=_BASELINE_CONFIDENCE,
    )


# ───── The architecture ─────────────────────────────────────────────────


class BaselineArchitecture(Architecture):
    """Deterministic, no-LLM baseline.

    Calls ``arxiv_search`` once, filters and trims, partitions into eras by
    publication date, fills the per-paper summary fields with the Phase-3
    heuristic-or-disclaimer policy, and returns a valid ``BenchmarkResult``
    with zeroed token counters.

    Class-level identity declared via the ``Architecture`` ABC's three
    required ClassVars. ``version`` bumps when the wiring changes (e.g. if
    we swap the partitioning strategy or change a disclaimer string).
    """

    name: ClassVar[str] = "arch_00_baseline"
    version: ClassVar[str] = "0.1.0"
    description: ClassVar[str] = (
        "Deterministic no-LLM floor: arxiv relevance search + schema dressing. "
        "Heuristic for derivable fields, distinct disclaimers for synthesis fields."
    )

    async def run(
        self,
        topic: str,
        *,
        prompt_versions: dict[str, str] | None = None,
        modes: Modes | None = None,
    ) -> BenchmarkResult:
        """See ``Architecture.run``. No-LLM execution path."""
        # Resolve modes early so target_paper_count is available to the
        # filter/trim step. ``Modes()`` provides the project-default 10.
        modes = modes or Modes()

        # Time bookkeeping: ``perf_counter`` for duration math (monotonic,
        # not affected by clock adjustments); ``datetime.now(UTC)`` for the
        # wall-clock timestamps that go into telemetry/provenance.
        started_at = datetime.now(UTC)
        t0 = time.perf_counter()

        # One arxiv call. tool_call_count below is 1; agent_call_count is 0.
        raw = await arxiv_search(topic, max_results=_OVERFETCH_SIZE)

        # Filter to PaperEntry-compatible ids; trim to the deliverable target.
        usable = [p for p in raw if _is_usable(p)]
        trimmed = usable[: modes.target_paper_count]

        # Honest guard rail: if we have <8, the schema can't validate.
        # Raise rather than fabricate.
        if len(trimmed) < 8:
            raise BaselineTooFewResultsError(topic=topic, found=len(trimmed))

        # Partition by publication date into ≤3 eras. The partition operates
        # on a sorted copy; the original ``trimmed`` list (in arxiv relevance
        # order) drives PaperEntry construction so the deliverable retains
        # the user-meaningful "most relevant first" ordering.
        slices = _partition_into_eras(trimmed)

        # Build the arxiv_id -> era_id map from the partition. The mapper
        # consults this map; PaperEntries come out in relevance order.
        paper_to_era: dict[str, str] = {}
        for s in slices:
            for p in s.papers:
                paper_to_era[p.arxiv_id] = s.era_id

        # Map each paper in original (relevance) order. Rank is 1-indexed
        # because it's user-facing prose ("rank 1 of 10" reads better than
        # "rank 0 of 10").
        total = len(trimmed)
        papers: list[PaperEntry] = [
            _arxiv_to_paper_entry(
                paper=p,
                rank=idx + 1,
                total=total,
                topic=topic,
                era_id=paper_to_era[p.arxiv_id],
            )
            for idx, p in enumerate(trimmed)
        ]

        # Build TimelineEra objects. paper_ids within each era are listed in
        # date order (the partition order), which is a sensible user-facing
        # default since the era *is* a temporal grouping.
        timeline: list[TimelineEra] = [
            TimelineEra(
                era_id=s.era_id,
                name=s.era_name,
                date_range_start=s.date_range_start,
                date_range_end=s.date_range_end,
                narrative=_DISCLAIMER_ERA_NARRATIVE,
                paper_ids=[p.arxiv_id for p in s.papers],
            )
            for s in slices
        ]

        # Pydantic's ``_era_paper_consistency`` validator runs here. If the
        # partitioner ever drifts (orphan paper, duplicate id, dangling era
        # reference), this raises a ValidationError — the bug is caught at
        # the boundary, not by silently producing wrong output.
        deliverable = Deliverable(
            topic=topic,
            overall_summary=_build_overall_summary(trimmed, topic),
            papers=papers,
            timeline=timeline,
        )

        finished_at = datetime.now(UTC)
        duration_seconds = time.perf_counter() - t0

        telemetry = Telemetry(
            architecture_name=self.name,
            architecture_version=self.version,
            prompt_versions=prompt_versions or {},
            modes=modes,
            started_at=started_at,
            finished_at=finished_at,
            duration_seconds=duration_seconds,
            total_input_tokens=0,
            total_output_tokens=0,
            total_cost_usd=0.0,
            agent_call_count=0,
            tool_call_count=1,
            trace=[],
            errors=[],
            candidates=[],
            unresolved_lookups=[],
        )

        provenance = build_provenance(created_at=finished_at)

        return BenchmarkResult(
            deliverable=deliverable,
            telemetry=telemetry,
            provenance=provenance,
        )


__all__ = ["BaselineArchitecture", "BaselineTooFewResultsError"]
