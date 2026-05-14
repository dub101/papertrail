"""Behavioural tests for the Evaluator class.

These tests never make a real Anthropic API call. The ``AsyncAnthropic``
client is replaced with a duck-typed fake that returns canned ``Message``
objects, so we can drive the Evaluator through every branch of its
response-parsing logic at zero cost.

What's covered:
    - Happy path: tool_use block parses cleanly into an EvaluatorVerdict
    - Wiring: model, system prompt, tool_choice, tool input_schema all
      reach client.messages.create unchanged
    - Refusal: response with no tool_use block raises EvaluatorRefusedError
    - Wrong-tool-name: tool_use block with a different name raises Refused
    - Validation failure: out-of-range score raises EvaluatorInvalidOutputError
    - Leading text block is skipped, not treated as failure
    - Provenance: model + prompt_version + recent timestamp on the returned score

DryRunEvaluator gets its own dedicated tests below.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from papertrail.benchmark import Deliverable, PaperEntry, TimelineEra
from papertrail.evaluator import (
    DryRunEvaluator,
    Evaluator,
    EvaluatorInvalidOutputError,
    EvaluatorRefusedError,
    EvaluatorVerdict,
)

# ───── Fixtures ─────────────────────────────────────────────────────────


def _make_deliverable() -> Deliverable:
    """Build a minimal-but-valid 8-paper deliverable for evaluator tests."""
    papers = [
        PaperEntry(
            # Use the post-2007 numeric arxiv id scheme that PaperEntry requires.
            arxiv_id=f"1706.0376{i}",
            title=f"Paper {i}",
            authors=["Doe, J."],
            published_date=date(2017, 6, (i % 28) + 1),
            url="https://arxiv.org/abs/1706.03760",
            citation_count=None,
            citation_source=None,
            era_id="era_1",
            summary_about=f"about-{i}",
            summary_relation_to_topic=f"relation-{i}",
            summary_problem=f"problem-{i}",
            summary_approach=f"approach-{i}",
            summary_impact=f"impact-{i}",
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
            narrative="single bucket narrative",
            paper_ids=[p.arxiv_id for p in papers],
        )
    ]
    return Deliverable(
        topic="self-attention",
        overall_summary="overall summary for self-attention",
        papers=papers,
        timeline=timeline,
    )


def _valid_verdict_input() -> dict[str, Any]:
    """A dict that round-trips through ``EvaluatorVerdict.model_validate``."""
    dim = {"score": 0.5, "rationale": "ok"}
    return {
        "overall": 0.5,
        "confidence": 0.5,
        "selection_relevance": dim,
        "timeline_quality": dim,
        "timeline_veracity": dim,
        "synthesis": {
            "about": dim,
            "relation_to_topic": dim,
            "problem": dim,
            "approach": dim,
            "impact": dim,
        },
        "executive_summary": dim,
        "critique": "looks fine",
    }


def _tool_use_block(name: str, tool_input: dict[str, Any]) -> MagicMock:
    """Build a fake content block that looks like an SDK ToolUseBlock."""
    block = MagicMock()
    block.type = "tool_use"
    block.name = name
    block.input = tool_input
    return block


def _text_block(text: str) -> MagicMock:
    """Build a fake content block that looks like an SDK TextBlock."""
    block = MagicMock()
    block.type = "text"
    block.text = text
    return block


def _fake_response(
    content: list[MagicMock],
    stop_reason: str = "tool_use",
    input_tokens: int = 5000,
    output_tokens: int = 1200,
) -> MagicMock:
    """Build a fake Message-shaped response object.

    The Anthropic SDK's ``Message`` has a ``.usage`` attribute carrying
    token counts; the Evaluator reads from it, so the fake must provide
    one or the new usage-tracking code path crashes on the attribute miss.
    """
    msg = MagicMock()
    msg.content = content
    msg.stop_reason = stop_reason
    usage = MagicMock()
    usage.input_tokens = input_tokens
    usage.output_tokens = output_tokens
    msg.usage = usage
    return msg


def _fake_client(response: MagicMock) -> MagicMock:
    """Build a fake AsyncAnthropic-shaped client that returns ``response``."""
    client = MagicMock()
    client.messages = MagicMock()
    client.messages.create = AsyncMock(return_value=response)
    return client


# ───── Happy path ───────────────────────────────────────────────────────


async def test_evaluate_happy_path_returns_valid_score() -> None:
    block = _tool_use_block("submit_evaluation", _valid_verdict_input())
    client = _fake_client(_fake_response([block]))
    evaluator = Evaluator(client)

    score = await evaluator.evaluate(_make_deliverable(), topic="self-attention")

    # Verdict came out validated and matches what the model "produced".
    assert score.verdict.overall == 0.5
    assert score.verdict.critique == "looks fine"
    assert score.verdict.synthesis.about.score == 0.5


async def test_evaluate_stamps_provenance_correctly() -> None:
    client = _fake_client(
        _fake_response([_tool_use_block("submit_evaluation", _valid_verdict_input())])
    )
    evaluator = Evaluator(client)

    score = await evaluator.evaluate(_make_deliverable(), topic="self-attention")

    assert score.evaluator_model == Evaluator.DEFAULT_MODEL
    assert score.evaluator_prompt_version == Evaluator.PROMPT_VERSION
    # Within 5 seconds of now, in UTC.
    assert (datetime.now(UTC) - score.evaluated_at).total_seconds() < 5


async def test_evaluate_records_usage_from_response() -> None:
    # The Evaluator must read token counts from response.usage and compute
    # the estimated cost from the local pricing table.
    response = _fake_response(
        [_tool_use_block("submit_evaluation", _valid_verdict_input())],
        input_tokens=4500,
        output_tokens=1100,
    )
    client = _fake_client(response)
    evaluator = Evaluator(client)

    score = await evaluator.evaluate(_make_deliverable(), topic="t")

    assert score.usage.input_tokens == 4500
    assert score.usage.output_tokens == 1100
    # Sonnet 4.6 default: 4500 x $3/M + 1100 x $15/M ≈ $0.0300.
    expected = 4500 / 1e6 * 3.0 + 1100 / 1e6 * 15.0
    assert score.usage.cost_usd_estimated == pytest.approx(expected)


async def test_evaluate_uses_model_specific_pricing_for_override() -> None:
    # When the constructor takes a model override (e.g. Haiku for iteration),
    # the cost estimate must use that model's rates, not the default's.
    response = _fake_response(
        [_tool_use_block("submit_evaluation", _valid_verdict_input())],
        input_tokens=1_000_000,
        output_tokens=1_000_000,
    )
    client = _fake_client(response)
    evaluator = Evaluator(client, model="claude-haiku-4-5")

    score = await evaluator.evaluate(_make_deliverable(), topic="t")

    # Haiku 4.5: 1M x $1 + 1M x $5 = $6.00. (Sonnet would have been $18.00.)
    assert score.usage.cost_usd_estimated == pytest.approx(6.0)


async def test_evaluate_skips_leading_text_block() -> None:
    # Model emits a brief preface text block, then the tool call. The
    # evaluator must ignore the preface and parse the tool_use block.
    content = [
        _text_block("Sure, here's my evaluation:"),
        _tool_use_block("submit_evaluation", _valid_verdict_input()),
    ]
    evaluator = Evaluator(_fake_client(_fake_response(content)))
    score = await evaluator.evaluate(_make_deliverable(), topic="t")
    assert score.verdict.overall == 0.5


# ───── Wiring: assert we passed the right args to the SDK ───────────────


async def test_evaluate_passes_correct_model_and_tool_choice() -> None:
    client = _fake_client(
        _fake_response([_tool_use_block("submit_evaluation", _valid_verdict_input())])
    )
    evaluator = Evaluator(client)

    await evaluator.evaluate(_make_deliverable(), topic="self-attention")

    # AsyncMock records the kwargs the client was called with.
    kwargs = client.messages.create.await_args.kwargs
    assert kwargs["model"] == "claude-sonnet-4-6"
    assert kwargs["tool_choice"] == {"type": "tool", "name": "submit_evaluation"}
    # The tool must be the only one defined, named correctly, with the
    # pydantic-generated input_schema.
    assert len(kwargs["tools"]) == 1
    tool = kwargs["tools"][0]
    assert tool["name"] == "submit_evaluation"
    assert tool["input_schema"] == EvaluatorVerdict.model_json_schema()


async def test_evaluate_passes_loaded_system_prompt() -> None:
    client = _fake_client(
        _fake_response([_tool_use_block("submit_evaluation", _valid_verdict_input())])
    )
    evaluator = Evaluator(client)

    await evaluator.evaluate(_make_deliverable(), topic="self-attention")

    kwargs = client.messages.create.await_args.kwargs
    system = kwargs["system"]
    # The v1 prompt header is stable across edits — looking for a substring
    # rather than the whole prompt to avoid a brittle string compare.
    assert "You are the Evaluator agent" in system
    assert "submit_evaluation" in system


async def test_evaluate_serializes_deliverable_as_markdown() -> None:
    client = _fake_client(
        _fake_response([_tool_use_block("submit_evaluation", _valid_verdict_input())])
    )
    evaluator = Evaluator(client)

    await evaluator.evaluate(_make_deliverable(), topic="self-attention")

    user_content = client.messages.create.await_args.kwargs["messages"][0]["content"]
    # Smoke checks on the markdown serialization: topic header, papers
    # section count, presence of one of the per-paper fields.
    assert "# Topic" in user_content
    assert "self-attention" in user_content
    assert "# Papers (8)" in user_content
    assert "summary_about:" in user_content


async def test_evaluate_overrides_model_when_constructor_takes_one() -> None:
    # ADR-0003 keeps Sonnet as the default, but the constructor accepts an
    # override for ad-hoc prompt iteration. Confirm the override propagates.
    client = _fake_client(
        _fake_response([_tool_use_block("submit_evaluation", _valid_verdict_input())])
    )
    evaluator = Evaluator(client, model="claude-haiku-4-5")
    await evaluator.evaluate(_make_deliverable(), topic="self-attention")
    assert client.messages.create.await_args.kwargs["model"] == "claude-haiku-4-5"


# ───── Refusal cases ────────────────────────────────────────────────────


async def test_evaluate_raises_refused_when_no_tool_use_block() -> None:
    # Model returned only a text block (refusal, or model decided to chat).
    client = _fake_client(
        _fake_response([_text_block("I can't evaluate this.")], stop_reason="end_turn")
    )
    evaluator = Evaluator(client)

    with pytest.raises(EvaluatorRefusedError) as exc_info:
        await evaluator.evaluate(_make_deliverable(), topic="t")

    err = exc_info.value
    assert err.stop_reason == "end_turn"
    # The model's prose is surfaced so the user can read it.
    assert "I can't evaluate this." in err.content_summary


async def test_evaluate_raises_refused_when_tool_name_mismatches() -> None:
    # Tool_use block exists but for a different tool — should still refuse
    # rather than try to interpret it.
    block = _tool_use_block("some_other_tool", {"foo": "bar"})
    client = _fake_client(_fake_response([block]))
    evaluator = Evaluator(client)

    with pytest.raises(EvaluatorRefusedError):
        await evaluator.evaluate(_make_deliverable(), topic="t")


async def test_evaluate_raises_refused_on_empty_content() -> None:
    client = _fake_client(_fake_response([], stop_reason="end_turn"))
    evaluator = Evaluator(client)

    with pytest.raises(EvaluatorRefusedError):
        await evaluator.evaluate(_make_deliverable(), topic="t")


# ───── Schema-violation cases ───────────────────────────────────────────


async def test_evaluate_raises_invalid_output_on_out_of_range_score() -> None:
    bad = _valid_verdict_input()
    bad["overall"] = 1.5  # Outside [0, 1].
    client = _fake_client(_fake_response([_tool_use_block("submit_evaluation", bad)]))
    evaluator = Evaluator(client)

    with pytest.raises(EvaluatorInvalidOutputError) as exc_info:
        await evaluator.evaluate(_make_deliverable(), topic="t")

    # The validation message names the failing field so the user knows what
    # to change. This is the actionable artifact of the no-retry policy.
    assert "overall" in exc_info.value.validation_message


async def test_evaluate_raises_invalid_output_on_missing_required_field() -> None:
    bad = _valid_verdict_input()
    del bad["synthesis"]
    client = _fake_client(_fake_response([_tool_use_block("submit_evaluation", bad)]))
    evaluator = Evaluator(client)

    with pytest.raises(EvaluatorInvalidOutputError):
        await evaluator.evaluate(_make_deliverable(), topic="t")


# ───── DryRunEvaluator ──────────────────────────────────────────────────


async def test_dry_run_evaluator_returns_sentinel_score() -> None:
    evaluator = DryRunEvaluator()
    score = await evaluator.evaluate(_make_deliverable(), topic="anything")

    # Every numeric field should be the sentinel 0.123.
    assert score.verdict.overall == 0.123
    assert score.verdict.confidence == 0.123
    assert score.verdict.selection_relevance.score == 0.123
    assert score.verdict.synthesis.impact.score == 0.123
    # Provenance is self-labeling.
    assert score.evaluator_model == "dry-run-no-llm"
    assert score.evaluator_prompt_version == "dry-run"
    # Usage is honestly zero — no real call happened.
    assert score.usage.input_tokens == 0
    assert score.usage.output_tokens == 0
    assert score.usage.cost_usd_estimated == 0.0


async def test_dry_run_evaluator_rationale_mentions_dry_run() -> None:
    # The string "DRY RUN" must appear in rationales so anyone scanning a
    # persisted JSON can spot the source instantly.
    evaluator = DryRunEvaluator()
    score = await evaluator.evaluate(_make_deliverable(), topic="t")
    assert "DRY RUN" in score.verdict.critique
    assert "DRY RUN" in score.verdict.selection_relevance.rationale
