"""Benchmark runner: glues an architecture, an evaluator, and persistence.

What this module does (in one paragraph):
    ``run_benchmark`` takes a topic, an architecture (by name or instance),
    and an optional evaluator, calls ``architecture.run(topic)`` to produce
    a ``BenchmarkResult``, optionally grades it with the evaluator, and
    writes both as a single JSON document to ``benchmark_runs/<file>.json``.
    The function returns the in-memory objects too, so callers (CLI, tests,
    future Jupyter notebooks) can inspect them without re-reading the file.

Why one file per run with both pieces inside:
    Results and scores are always read together — you never want the verdict
    without the artifact it judged. Co-locating them prevents the failure
    mode where score files outlive the result they reference.

Why the architecture registry lives here:
    There are 7 architectures total across this project; an explicit dict
    is more honest about that bounded set than plugin discovery, and a typo
    in ``--architecture`` produces a clear "unknown architecture" error
    instead of silently picking the wrong one.

Cert mapping: none directly — this is harness infrastructure. The evaluator
it calls is the D4 TS 4.6 / TS 4.3 piece; the architecture is the D1 piece.
The runner just wires them.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Final

from papertrail.architecture import Architecture
from papertrail.architectures.arch_00_baseline import BaselineArchitecture
from papertrail.benchmark import BenchmarkResult
from papertrail.evaluator import DryRunEvaluator, Evaluator, EvaluatorScore

# Architecture registry. As new architectures land (arch_01 ... arch_06)
# they get one line added here. Adding via plugin discovery would be more
# flexible but less debuggable — a misspelled name should error out, not
# silently miss.
ARCHITECTURES: Final[dict[str, type[Architecture]]] = {
    BaselineArchitecture.name: BaselineArchitecture,
}

# Default location for run artifacts. Gitignored — these are produced
# outputs, not source.
DEFAULT_OUTPUT_DIR: Final[Path] = Path("benchmark_runs")

# Type alias for "anything that fulfils the evaluator protocol". Both the
# real and the dry-run class match it structurally; we use a union for
# explicit, mypy-friendly typing rather than introducing a Protocol for
# two implementations.
EvaluatorLike = Evaluator | DryRunEvaluator


# ───── Internal helpers ─────────────────────────────────────────────────


def _filename_for_run(result: BenchmarkResult) -> str:
    """Build a stable, sortable filename for one run's JSON artifact.

    Shape: ``<UTC-iso-compact>_<arch-name>_<short-run-id>.json``

    UTC compact format (``20260514T091500Z``) means ``ls`` shows runs in
    chronological order naturally. The short id makes it tractable to
    eyeball a directory listing; the full ``run_id`` lives inside the
    file for unambiguous reference.
    """
    ts = result.provenance.created_at.strftime("%Y%m%dT%H%M%SZ")
    short_id = str(result.provenance.run_id)[:8]
    return f"{ts}_{result.telemetry.architecture_name}_{short_id}.json"


def _resolve_architecture(architecture: Architecture | str) -> Architecture:
    """Accept either a name or a pre-constructed instance.

    Names go through the registry; instances are returned as-is. This dual
    shape lets the CLI keep a small surface (string from argparse) while
    tests stay clean (inject a fake architecture instance directly without
    touching the registry).
    """
    if isinstance(architecture, Architecture):
        return architecture
    arch_cls = ARCHITECTURES.get(architecture)
    if arch_cls is None:
        raise ValueError(
            f"Unknown architecture {architecture!r}. "
            f"Available: {sorted(ARCHITECTURES)}"
        )
    return arch_cls()


# ───── Public API ───────────────────────────────────────────────────────


async def run_benchmark(
    *,
    topic: str,
    architecture: Architecture | str,
    evaluator: EvaluatorLike | None,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> tuple[BenchmarkResult, EvaluatorScore | None, Path]:
    """Execute one architecture against ``topic`` and persist the run.

    Args:
        topic: The research topic to investigate.
        architecture: Either an architecture name (e.g. ``"arch_00_baseline"``)
            or a pre-constructed ``Architecture`` instance. Tests pass
            instances; the CLI passes names.
        evaluator: ``Evaluator``, ``DryRunEvaluator``, or ``None`` to skip
            evaluation entirely. ``None`` writes the result with
            ``"score": null`` in the JSON.
        output_dir: Where to write the artifact. Created if missing.

    Returns:
        ``(result, score, path)`` — ``score`` is ``None`` iff ``evaluator``
        was ``None``. The file at ``path`` contains both as JSON.

    Raises:
        ValueError: If ``architecture`` is a string with no registry entry.
        Anything ``architecture.run()`` raises (e.g. ``BaselineTooFewResultsError``)
            propagates — there is no swallow-and-continue path.
        Anything ``evaluator.evaluate()`` raises (``EvaluatorRefusedError``,
            ``EvaluatorInvalidOutputError``) propagates for the same reason.
    """
    arch = _resolve_architecture(architecture)

    # The architecture call is the heavy step for everything from arch_01
    # onward. For arch_00 it's one arxiv search.
    result = await arch.run(topic)

    score: EvaluatorScore | None = None
    if evaluator is not None:
        # Pass only the deliverable, not the full BenchmarkResult — the
        # evaluator must grade on output merits, not telemetry context.
        score = await evaluator.evaluate(result.deliverable, topic=topic)

    # Persistence happens last so a grading failure doesn't leave a
    # half-written file. ``mkdir(parents=True, exist_ok=True)`` is safe
    # to call every run.
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / _filename_for_run(result)

    payload = {
        "result": result.model_dump(mode="json"),
        "score": score.model_dump(mode="json") if score is not None else None,
    }
    # ``default=str`` is a belt-and-braces fallback — pydantic's
    # ``mode="json"`` already serializes datetimes, but if a downstream
    # dump ever produces something stdlib JSON can't handle, we get a
    # readable string instead of a crash.
    path.write_text(
        json.dumps(payload, indent=2, default=str),
        encoding="utf-8",
    )

    return result, score, path


__all__ = [
    "ARCHITECTURES",
    "DEFAULT_OUTPUT_DIR",
    "EvaluatorLike",
    "run_benchmark",
]
