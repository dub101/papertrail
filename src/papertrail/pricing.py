"""Per-model Anthropic API pricing — single source of truth.

What this module does (in one paragraph):
    Carries the per-million-token rates for every Claude model PaperTrail
    invokes, plus the helper that turns (model, input_tokens, output_tokens)
    into an estimated USD cost. Previously duplicated across ``evaluator.py``
    and ``architectures/arch_01_sequential/search.py``; extracted here once
    the triage stage made it the third caller, per the codebase rule (see
    ``evaluator.py``'s ``_StrictModel`` comment: "duplicate twice, extract
    on the third caller").

Why module-level not config:
    The rates change rarely (Anthropic pricing announcements) and the
    update is always a code change with a visible diff in the
    ``MODEL_PRICING_PER_MTOK`` literal. A config file would hide that
    behind a deployment artifact and lose the audit trail.

Cert mapping:
    None — this is plumbing. The cost numbers flow into
    ``BenchmarkResult.telemetry.total_cost_usd`` and into the Evaluator's
    ``EvaluatorScore.usage.cost_usd_estimated``, both labelled as
    *estimated* downstream because Anthropic's billing is the ground truth.

Libraries: stdlib only.
"""

from __future__ import annotations

from typing import Final

# Anthropic's messages API returns exact token counts
# (``Usage.input_tokens`` / ``Usage.output_tokens``) but does NOT return a
# per-call dollar amount. Cost is computed client-side from these tokens
# multiplied by the per-million-token rates below. Keep this table in sync
# with anthropic.com/pricing.
#
# Tuple shape: ``(input_per_mtok_usd, output_per_mtok_usd)``.
# Verified 2026-05-15. Bump the date in this comment when updating rates.
#
# An unknown model falls back to ``0.0`` cost in ``compute_cost_usd`` -- a
# visibly-wrong zero against a real model is a louder "the table is stale,
# fix it" signal than a silent miss or a crash.
MODEL_PRICING_PER_MTOK: Final[dict[str, tuple[float, float]]] = {
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
    "claude-haiku-4-5-20251001": (1.0, 5.0),
    # Sentinel -- DryRunEvaluator stamps this model id, explicit zero makes
    # the no-cost behavior a documented property rather than a fallback.
    "dry-run-no-llm": (0.0, 0.0),
}


def compute_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate USD cost from token counts using ``MODEL_PRICING_PER_MTOK``.

    The result is an *estimate*: ground truth lives on Anthropic's billing
    side; this function does the per-million-token arithmetic with the
    rates we have locally. Use ``input_tokens`` / ``output_tokens`` (exact,
    from the API) when you need authoritative numbers.

    Unknown models return ``0.0``. The downstream code should label these
    values as estimated (see ``EvaluatorUsage.cost_usd_estimated``) so a
    surprise zero is interpreted correctly.
    """
    pricing = MODEL_PRICING_PER_MTOK.get(model)
    if pricing is None:
        return 0.0
    input_rate, output_rate = pricing
    return (
        (input_tokens / 1_000_000) * input_rate
        + (output_tokens / 1_000_000) * output_rate
    )


__all__ = ["MODEL_PRICING_PER_MTOK", "compute_cost_usd"]
