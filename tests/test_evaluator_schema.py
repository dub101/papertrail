"""Schema-level tests for the Evaluator types.

Covers each validator and bound on ``DimensionScore``, ``SynthesisScores``,
``EvaluatorVerdict``, and ``EvaluatorScore``. The LLM-caller side of the
Evaluator lives in 5b and gets its own test file with a mocked AsyncAnthropic
client — this file only exercises the pydantic contract.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest
from pydantic import ValidationError

from papertrail.evaluator import (
    SUMMARY_FIELDS,
    DimensionScore,
    EvaluatorScore,
    EvaluatorUsage,
    EvaluatorVerdict,
    SynthesisScores,
    _compute_cost_usd,
)

# ───── Helpers ──────────────────────────────────────────────────────────


def _dim(score: float = 0.5, rationale: str = "ok") -> DimensionScore:
    """Build a valid DimensionScore for use inside larger fixtures."""
    return DimensionScore(score=score, rationale=rationale)


def _synthesis(**overrides: DimensionScore) -> SynthesisScores:
    """Build a valid SynthesisScores; overrides replace specific subfields."""
    defaults: dict[str, DimensionScore] = {
        "about": _dim(),
        "relation_to_topic": _dim(),
        "problem": _dim(),
        "approach": _dim(),
        "impact": _dim(),
    }
    defaults.update(overrides)
    return SynthesisScores(**defaults)


def _verdict(**overrides: Any) -> EvaluatorVerdict:
    """Build a valid EvaluatorVerdict; overrides mutate specific fields."""
    defaults: dict[str, Any] = {
        "overall": 0.5,
        "confidence": 0.5,
        "selection_relevance": _dim(),
        "timeline_quality": _dim(),
        "timeline_veracity": _dim(),
        "synthesis": _synthesis(),
        "executive_summary": _dim(),
        "critique": "Baseline grading, no issues flagged.",
    }
    defaults.update(overrides)
    return EvaluatorVerdict(**defaults)


# ───── DimensionScore ───────────────────────────────────────────────────


def test_dimension_score_happy_path() -> None:
    d = DimensionScore(score=0.75, rationale="solid coverage of the topic")
    assert d.score == 0.75
    assert d.rationale.startswith("solid")


@pytest.mark.parametrize("bad_score", [-0.01, 1.01, 2.0, -1.0])
def test_dimension_score_rejects_out_of_range(bad_score: float) -> None:
    with pytest.raises(ValidationError):
        DimensionScore(score=bad_score, rationale="x")


def test_dimension_score_rejects_empty_rationale() -> None:
    # A bare number with no justification is the failure mode this schema
    # is designed to prevent — verify the validator catches it.
    with pytest.raises(ValidationError):
        DimensionScore(score=0.5, rationale="")


def test_dimension_score_forbids_extra_fields() -> None:
    # Typos like ``scor=`` should fail loudly, not be silently dropped.
    with pytest.raises(ValidationError):
        DimensionScore(score=0.5, rationale="ok", notes="should-not-exist")


# ───── SynthesisScores ──────────────────────────────────────────────────


def test_synthesis_scores_requires_all_five_fields() -> None:
    # Drop one field; construction must fail.
    with pytest.raises(ValidationError):
        SynthesisScores(
            about=_dim(),
            relation_to_topic=_dim(),
            problem=_dim(),
            approach=_dim(),
            # impact missing
        )


def test_synthesis_scores_field_names_match_summary_fields_constant() -> None:
    # ``SUMMARY_FIELDS`` is the canonical PaperEntry list; SynthesisScores
    # must mirror it (minus the ``summary_`` prefix) or arch_00's per-field
    # disclaimer scoring will drift.
    expected = {f.removeprefix("summary_") for f in SUMMARY_FIELDS}
    actual = set(SynthesisScores.model_fields.keys())
    assert actual == expected


# ───── EvaluatorVerdict ─────────────────────────────────────────────────


def test_verdict_happy_path() -> None:
    v = _verdict()
    assert 0.0 <= v.overall <= 1.0
    assert 0.0 <= v.confidence <= 1.0
    assert v.critique


@pytest.mark.parametrize("bad", [-0.1, 1.1])
def test_verdict_overall_bounded(bad: float) -> None:
    with pytest.raises(ValidationError):
        _verdict(overall=bad)


@pytest.mark.parametrize("bad", [-0.1, 1.1])
def test_verdict_confidence_bounded(bad: float) -> None:
    with pytest.raises(ValidationError):
        _verdict(confidence=bad)


def test_verdict_rejects_empty_critique() -> None:
    with pytest.raises(ValidationError):
        _verdict(critique="")


def test_verdict_forbids_extra_fields() -> None:
    with pytest.raises(ValidationError):
        _verdict(extra_dimension=_dim())


def test_verdict_round_trips_through_json() -> None:
    # The Evaluator class (5b) will reconstruct verdicts from the model's
    # tool_use input; round-trip ensures the JSON shape matches the pydantic
    # shape exactly.
    v1 = _verdict(overall=0.42, confidence=0.9)
    v2 = EvaluatorVerdict.model_validate_json(v1.model_dump_json())
    assert v2 == v1


def test_verdict_json_schema_has_required_top_level_fields() -> None:
    # ``input_schema`` for the tool_use call is derived from this; if any of
    # these fields stop appearing, the model can't be guided to produce them.
    schema = EvaluatorVerdict.model_json_schema()
    required = set(schema["required"])
    expected = {
        "overall",
        "confidence",
        "selection_relevance",
        "timeline_quality",
        "timeline_veracity",
        "synthesis",
        "executive_summary",
        "critique",
    }
    assert required == expected


# ───── EvaluatorUsage ───────────────────────────────────────────────────


def _usage(
    input_tokens: int = 100,
    output_tokens: int = 50,
    cost_usd_estimated: float = 0.0,
) -> EvaluatorUsage:
    return EvaluatorUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd_estimated=cost_usd_estimated,
    )


def test_usage_happy_path() -> None:
    u = _usage(input_tokens=4500, output_tokens=1100, cost_usd_estimated=0.0300)
    assert u.input_tokens == 4500
    assert u.output_tokens == 1100
    assert u.cost_usd_estimated == 0.0300


@pytest.mark.parametrize("bad", [-1, -100])
def test_usage_rejects_negative_token_counts(bad: int) -> None:
    with pytest.raises(ValidationError):
        EvaluatorUsage(input_tokens=bad, output_tokens=0, cost_usd_estimated=0.0)
    with pytest.raises(ValidationError):
        EvaluatorUsage(input_tokens=0, output_tokens=bad, cost_usd_estimated=0.0)


def test_usage_rejects_negative_cost() -> None:
    with pytest.raises(ValidationError):
        EvaluatorUsage(input_tokens=0, output_tokens=0, cost_usd_estimated=-0.01)


# ───── Cost helper ──────────────────────────────────────────────────────


def test_compute_cost_usd_for_known_sonnet_model() -> None:
    # 1M input tokens x $3 + 1M output tokens x $15 = $18.
    cost = _compute_cost_usd("claude-sonnet-4-6", 1_000_000, 1_000_000)
    assert cost == pytest.approx(18.0)


def test_compute_cost_usd_for_typical_call() -> None:
    # 5000 input + 1200 output for sonnet ≈ $0.033.
    cost = _compute_cost_usd("claude-sonnet-4-6", 5000, 1200)
    assert cost == pytest.approx(5000 / 1e6 * 3.0 + 1200 / 1e6 * 15.0)


def test_compute_cost_usd_for_unknown_model_returns_zero() -> None:
    # The fallback behavior is "visibly wrong zero, not a crash" so the
    # operator notices the pricing table is stale.
    assert _compute_cost_usd("model-not-in-table", 100_000, 100_000) == 0.0


def test_compute_cost_usd_for_dry_run_sentinel_returns_zero() -> None:
    # DryRunEvaluator stamps "dry-run-no-llm" and we want explicit $0,
    # not an "unknown model" $0. Both end up zero but for different reasons.
    assert _compute_cost_usd("dry-run-no-llm", 5000, 1000) == 0.0


# ───── EvaluatorScore ───────────────────────────────────────────────────


def test_score_happy_path() -> None:
    s = EvaluatorScore(
        verdict=_verdict(),
        evaluator_model="claude-sonnet-4-6",
        evaluator_prompt_version="v1",
        evaluated_at=datetime.now(UTC),
        usage=_usage(),
    )
    assert s.evaluator_model == "claude-sonnet-4-6"
    assert s.verdict.overall == 0.5
    assert s.usage.input_tokens == 100


def test_score_requires_non_empty_model_and_prompt_version() -> None:
    base: dict[str, Any] = {
        "verdict": _verdict(),
        "evaluated_at": datetime.now(UTC),
        "usage": _usage(),
    }
    with pytest.raises(ValidationError):
        EvaluatorScore(**base, evaluator_model="", evaluator_prompt_version="v1")
    with pytest.raises(ValidationError):
        EvaluatorScore(**base, evaluator_model="claude-sonnet-4-6", evaluator_prompt_version="")


def test_score_requires_usage() -> None:
    # Usage is non-optional — every persisted score must carry token counts.
    with pytest.raises(ValidationError):
        EvaluatorScore(
            verdict=_verdict(),
            evaluator_model="claude-sonnet-4-6",
            evaluator_prompt_version="v1",
            evaluated_at=datetime.now(UTC),
            # usage missing
        )


def test_score_forbids_extra_fields() -> None:
    with pytest.raises(ValidationError):
        EvaluatorScore(
            verdict=_verdict(),
            evaluator_model="claude-sonnet-4-6",
            evaluator_prompt_version="v1",
            evaluated_at=datetime.now(UTC),
            usage=_usage(),
            note="should-not-exist",
        )
