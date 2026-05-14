#!/usr/bin/env python3
"""CLI launcher for ``papertrail.runner.run_benchmark``.

This script is intentionally thin — argparse + env loading + console output.
Anything that needs to be tested lives inside ``papertrail.runner``.

Usage examples:

    # Dry run — exercise the wire-up at $0 cost, sentinel scores in JSON.
    python scripts/run_benchmark.py --topic "self-attention" --dry-run

    # Architecture-only — run arch_00 against arxiv, skip evaluation.
    python scripts/run_benchmark.py --topic "self-attention" --no-evaluator

    # Real run — arch_00 + Sonnet evaluator. Requires ANTHROPIC_API_KEY.
    python scripts/run_benchmark.py --topic "self-attention" --verbose

Exit codes:
    0  success
    2  configuration error (missing API key when real evaluator was asked for)
    Other non-zero  any exception from the runner is allowed to surface as a
                    traceback — operator-facing tool, fail-loud is the right
                    default.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from papertrail.benchmark import BenchmarkResult
from papertrail.evaluator import DryRunEvaluator, Evaluator, EvaluatorScore
from papertrail.runner import (
    ARCHITECTURES,
    DEFAULT_OUTPUT_DIR,
    EvaluatorLike,
    run_benchmark,
)


def _build_parser() -> argparse.ArgumentParser:
    """Construct the argparse parser. Factored out so tests can inspect it."""
    parser = argparse.ArgumentParser(
        prog="run_benchmark",
        description=(
            "Run a PaperTrail architecture against a topic, optionally grade "
            "the deliverable with the Evaluator, and persist the result to "
            "benchmark_runs/."
        ),
    )
    parser.add_argument(
        "--topic",
        required=True,
        help="The research topic to investigate (e.g. 'self-attention in LLMs').",
    )
    parser.add_argument(
        "--architecture",
        default="arch_00_baseline",
        choices=sorted(ARCHITECTURES),
        help="Which architecture to run. Default: arch_00_baseline.",
    )
    # --dry-run and --no-evaluator are mutually exclusive: one swaps the
    # evaluator for sentinels, the other skips it entirely. Both can't apply.
    evaluator_group = parser.add_mutually_exclusive_group()
    evaluator_group.add_argument(
        "--dry-run",
        action="store_true",
        help="Use DryRunEvaluator — sentinel scores, zero LLM calls.",
    )
    evaluator_group.add_argument(
        "--no-evaluator",
        action="store_true",
        help="Skip evaluation entirely; persisted JSON has score=null.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print the full per-dimension verdict table after the run.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Where to write the run artifact. Default: {DEFAULT_OUTPUT_DIR}.",
    )
    return parser


def _pick_evaluator(args: argparse.Namespace) -> EvaluatorLike | None:
    """Construct the evaluator implied by the CLI flags.

    Returns ``None`` when ``--no-evaluator`` was passed. Returns a
    ``DryRunEvaluator`` for ``--dry-run``. Otherwise constructs a real
    ``Evaluator`` and exits with code 2 if ``ANTHROPIC_API_KEY`` is unset
    (no point reaching ``messages.create`` only to fail there).
    """
    if args.no_evaluator:
        return None
    if args.dry_run:
        return DryRunEvaluator()
    # Real evaluator path — needs an API key.
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print(
            "error: ANTHROPIC_API_KEY is not set. "
            "Use --dry-run for zero-cost testing, or add the key to your .env.",
            file=sys.stderr,
        )
        raise SystemExit(2)
    # Lazy import: ``anthropic`` only needed when a real evaluator is built.
    from anthropic import AsyncAnthropic

    return Evaluator(AsyncAnthropic(api_key=api_key))


# ───── Console output ───────────────────────────────────────────────────


def _print_minimal(result: BenchmarkResult, score: EvaluatorScore | None) -> None:
    """One-line-per-fact summary. The default, low-noise output."""
    tel = result.telemetry
    print(
        f"[{tel.architecture_name} v{tel.architecture_version}] "
        f"{len(result.deliverable.papers)} papers · "
        f"duration {tel.duration_seconds:.2f}s · "
        f"arch cost ${tel.total_cost_usd:.4f}"
    )
    if score is not None:
        v = score.verdict
        u = score.usage
        # Tokens are exact (from API), cost is local-estimate. Flagged ~ to
        # make the approximate-ness visible at a glance in the terminal.
        print(
            f"  evaluator: {score.evaluator_model} "
            f"prompt={score.evaluator_prompt_version} · "
            f"overall={v.overall:.2f} confidence={v.confidence:.2f} · "
            f"tokens in/out={u.input_tokens}/{u.output_tokens} · "
            f"~${u.cost_usd_estimated:.4f} (est.)"
        )


def _print_verbose_table(result: BenchmarkResult, score: EvaluatorScore | None) -> None:
    """Per-dimension table using rich. Triggered by ``--verbose``.

    ``rich`` is already a project dependency; the import is local so the
    minimal-output path doesn't pay the import cost.
    """
    from rich.console import Console
    from rich.table import Table

    console = Console()
    _print_minimal(result, score)

    if score is None:
        console.print("[dim](no evaluator score — nothing to tabulate)[/dim]")
        return

    v = score.verdict

    # Single flat table: dimension name, score, rationale (rich wraps long
    # rationales naturally so we don't have to truncate by hand).
    table = Table(title=f"Evaluator verdict for '{result.deliverable.topic}'")
    table.add_column("Dimension", style="bold")
    table.add_column("Score", justify="right")
    table.add_column("Rationale", overflow="fold")

    # Top-line first so the overall score reads at a glance.
    table.add_row("overall", f"{v.overall:.2f}", "(model-judged top-line)")
    table.add_row("confidence", f"{v.confidence:.2f}", "(evaluator self-report)")
    table.add_section()

    table.add_row(
        "selection_relevance",
        f"{v.selection_relevance.score:.2f}",
        v.selection_relevance.rationale,
    )
    table.add_row(
        "timeline_quality",
        f"{v.timeline_quality.score:.2f}",
        v.timeline_quality.rationale,
    )
    table.add_row(
        "timeline_veracity",
        f"{v.timeline_veracity.score:.2f}",
        v.timeline_veracity.rationale,
    )
    table.add_section()
    # synthesis.* rows mirror the per-paper summary field names so the user
    # can connect a low score to which summary field arch_00 disclaimed.
    table.add_row(
        "synthesis.about",
        f"{v.synthesis.about.score:.2f}",
        v.synthesis.about.rationale,
    )
    table.add_row(
        "synthesis.relation_to_topic",
        f"{v.synthesis.relation_to_topic.score:.2f}",
        v.synthesis.relation_to_topic.rationale,
    )
    table.add_row(
        "synthesis.problem",
        f"{v.synthesis.problem.score:.2f}",
        v.synthesis.problem.rationale,
    )
    table.add_row(
        "synthesis.approach",
        f"{v.synthesis.approach.score:.2f}",
        v.synthesis.approach.rationale,
    )
    table.add_row(
        "synthesis.impact",
        f"{v.synthesis.impact.score:.2f}",
        v.synthesis.impact.rationale,
    )
    table.add_section()
    table.add_row(
        "executive_summary",
        f"{v.executive_summary.score:.2f}",
        v.executive_summary.rationale,
    )

    console.print(table)
    console.print(f"\n[bold]Critique:[/bold] {v.critique}\n")


# ───── Entry point ──────────────────────────────────────────────────────


async def main(argv: list[str] | None = None) -> int:
    """Async entry point. Returns the process exit code."""
    # ``load_dotenv`` is no-op if .env is missing — safe to call unconditionally.
    # We do it before constructing the evaluator so ANTHROPIC_API_KEY is in env.
    load_dotenv()

    args = _build_parser().parse_args(argv)
    evaluator = _pick_evaluator(args)

    result, score, path = await run_benchmark(
        topic=args.topic,
        architecture=args.architecture,
        evaluator=evaluator,
        output_dir=args.output_dir,
    )

    print(f"Run saved to: {path}")
    if args.verbose:
        _print_verbose_table(result, score)
    else:
        _print_minimal(result, score)
    return 0


if __name__ == "__main__":  # pragma: no cover
    # Bottom-of-script idiom. asyncio.run drives the async main() to
    # completion and forwards its exit code to the OS.
    sys.exit(asyncio.run(main()))
