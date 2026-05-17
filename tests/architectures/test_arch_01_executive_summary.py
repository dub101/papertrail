"""Tests for ``arch_01`` stage 5: ``ExecutiveSummaryAgent``.

Single-forced-tool-use machinery same as triage / era partition. The
extra surface area here is (a) the eras-first / syntheses-grouped-by-era
user message ordering for D5 TS 5.1 mitigation, and (b) the
defensive alignment check that catches orphan syntheses before any
API call is made.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from papertrail.architectures.arch_01_sequential.era_partition import EraEntry
from papertrail.architectures.arch_01_sequential.executive_summary import (
    ExecutiveSummary,
    ExecutiveSummaryAgent,
    ExecutiveSummaryError,
    ExecutiveSummaryInvalidOutputError,
    ExecutiveSummaryRefusedError,
)
from papertrail.architectures.arch_01_sequential.synthesis import PaperSynthesis

# ───── Builders ─────────────────────────────────────────────────────────


def _paper_ids(n: int) -> list[str]:
    return [f"1706.{i:05d}" for i in range(n)]


def _era_narrative() -> str:
    """A narrative long enough to pass EraEntry's 200-char minimum."""
    return (
        "This era of the field marks a methodological shift. Researchers "
        "moved from one set of assumptions to a substantively different "
        "framing, producing a body of work that downstream papers treat "
        "as the canonical starting point. The shift reshaped the questions "
        "practitioners considered worth asking."
    )


def _era(era_id: str, paper_ids: list[str], *, start: int = 2017, end: int | None = 2020) -> EraEntry:
    return EraEntry(
        era_id=era_id,
        name=era_id.replace("-", " ").title(),
        date_range_start_year=start,
        date_range_end_year=end,
        narrative=_era_narrative(),
        paper_ids=paper_ids,
    )


def _synth(arxiv_id: str, *, confidence: float = 0.85) -> PaperSynthesis:
    return PaperSynthesis(
        arxiv_id=arxiv_id,
        summary_about=f"What {arxiv_id} is about.",
        summary_relation_to_topic=f"How {arxiv_id} relates to the topic.",
        summary_problem=f"The problem {arxiv_id} addresses.",
        summary_approach=f"The approach {arxiv_id} takes.",
        summary_impact=f"The impact of {arxiv_id}.",
        confidence=confidence,
    )


def _valid_summary_text(seed: str = "topic") -> str:
    """An executive-summary string that passes the 600-char min_length floor."""
    return (
        f"The {seed} field has evolved across several distinct eras. Early "
        "work established the foundational framing and the canonical problem "
        "definition; subsequent extensions reframed the methodological centre "
        "and introduced techniques that addressed the limits of the original "
        "approach. The most recent generation of work synthesises both "
        "directions while addressing constraints the earlier methods could "
        "not handle. Throughout this evolution, the central tension has been "
        "between expressiveness and computational tractability, with each era "
        "negotiating that tension along a different axis. The current "
        "frontier focuses on bridging the remaining gaps."
    )


def _summary_dict(text: str | None = None, *, confidence: float = 0.82) -> dict[str, Any]:
    return {
        "summary": text or _valid_summary_text(),
        "confidence": confidence,
    }


def _tool_use_block(
    summary_input: dict[str, Any], *, name: str = "submit_executive_summary"
) -> MagicMock:
    block = MagicMock()
    block.type = "tool_use"
    block.name = name
    block.id = "tu_1"
    block.input = summary_input
    return block


def _text_block(text: str) -> MagicMock:
    block = MagicMock()
    block.type = "text"
    block.text = text
    return block


def _fake_message(
    content: list[MagicMock],
    *,
    stop_reason: str = "tool_use",
    input_tokens: int = 4000,
    output_tokens: int = 500,
) -> MagicMock:
    msg = MagicMock()
    msg.content = content
    msg.stop_reason = stop_reason
    usage = MagicMock()
    usage.input_tokens = input_tokens
    usage.output_tokens = output_tokens
    msg.usage = usage
    return msg


def _fake_client(response: MagicMock) -> MagicMock:
    client = MagicMock()
    client.messages = MagicMock()
    client.messages.create = AsyncMock(return_value=response)
    return client


def _two_eras_eight_papers() -> tuple[list[EraEntry], list[PaperSynthesis]]:
    """A typical input shape: 2 eras of 4 papers each, 8 syntheses aligned."""
    ids = _paper_ids(8)
    eras = [
        _era("foundations", ids[:4], start=2014, end=2018),
        _era("extensions", ids[4:], start=2017, end=2022),
    ]
    syntheses = [_synth(i) for i in ids]
    return eras, syntheses


# ───── Class-level wiring ───────────────────────────────────────────────


def test_classvars_default_to_haiku_and_v1_prompt() -> None:
    assert ExecutiveSummaryAgent.DEFAULT_MODEL == "claude-haiku-4-5"
    assert ExecutiveSummaryAgent.PROMPT_NAME == "arch_01_executive_summary"
    assert ExecutiveSummaryAgent.PROMPT_VERSION == "v1"
    assert ExecutiveSummaryAgent.TOOL_NAME == "submit_executive_summary"


def test_constructor_does_not_load_prompt() -> None:
    agent = ExecutiveSummaryAgent(_fake_client(_fake_message([])))
    assert agent._system_prompt is None


# ───── Happy path ───────────────────────────────────────────────────────


async def test_summarize_happy_path_returns_summary_and_confidence() -> None:
    eras, syntheses = _two_eras_eight_papers()
    response = _fake_message([_tool_use_block(_summary_dict(confidence=0.85))])
    agent = ExecutiveSummaryAgent(_fake_client(response))

    result = await agent.summarize(
        topic="positional encodings", eras=eras, syntheses=syntheses
    )

    assert _valid_summary_text() == result.summary
    assert result.confidence == 0.85


async def test_summarize_records_token_usage_and_cost() -> None:
    eras, syntheses = _two_eras_eight_papers()
    response = _fake_message(
        [_tool_use_block(_summary_dict())], input_tokens=5000, output_tokens=400
    )
    agent = ExecutiveSummaryAgent(_fake_client(response))

    result = await agent.summarize(topic="t", eras=eras, syntheses=syntheses)

    assert result.usage.input_tokens == 5000
    assert result.usage.output_tokens == 400
    expected = 5000 / 1e6 * 1.0 + 400 / 1e6 * 5.0  # Haiku
    assert result.usage.cost_usd == pytest.approx(expected)


# ───── SDK wiring ───────────────────────────────────────────────────────


async def test_summarize_forces_tool_use_and_passes_haiku() -> None:
    eras, syntheses = _two_eras_eight_papers()
    client = _fake_client(_fake_message([_tool_use_block(_summary_dict())]))
    agent = ExecutiveSummaryAgent(client)

    await agent.summarize(topic="t", eras=eras, syntheses=syntheses)

    kwargs = client.messages.create.await_args.kwargs
    assert kwargs["model"] == "claude-haiku-4-5"
    assert kwargs["tool_choice"] == {"type": "tool", "name": "submit_executive_summary"}
    assert len(kwargs["tools"]) == 1
    assert kwargs["tools"][0]["input_schema"] == ExecutiveSummary.model_json_schema()


# ───── User message ordering (D5 TS 5.1 mitigation) ─────────────────────


async def test_summarize_user_message_places_eras_before_per_paper_blocks() -> None:
    """Key findings (eras) at the top, supporting detail (syntheses) after."""
    eras, syntheses = _two_eras_eight_papers()
    client = _fake_client(_fake_message([_tool_use_block(_summary_dict())]))
    agent = ExecutiveSummaryAgent(client)

    await agent.summarize(topic="positional encodings", eras=eras, syntheses=syntheses)

    body: str = client.messages.create.await_args.kwargs["messages"][0]["content"]
    # Topic header first.
    topic_pos = body.index("# Topic")
    eras_pos = body.index("# Era partition")
    syntheses_pos = body.index("# Per-paper syntheses")
    assert topic_pos < eras_pos < syntheses_pos


async def test_summarize_user_message_groups_per_paper_blocks_by_era() -> None:
    """Per-paper blocks are tagged with era and appear in era order."""
    eras, syntheses = _two_eras_eight_papers()
    client = _fake_client(_fake_message([_tool_use_block(_summary_dict())]))
    agent = ExecutiveSummaryAgent(client)

    await agent.summarize(topic="t", eras=eras, syntheses=syntheses)

    body: str = client.messages.create.await_args.kwargs["messages"][0]["content"]
    # First per-paper block should be from the foundations era; the
    # last should be from the extensions era.
    first_pos = body.index("## arxiv_id=1706.00000")
    last_pos = body.index("## arxiv_id=1706.00007")
    assert first_pos < last_pos
    # Era tagging present in the per-paper block.
    assert "era: foundations" in body
    assert "era: extensions" in body


async def test_summarize_user_message_handles_ongoing_era_open_end() -> None:
    """Era with date_range_end_year=None renders as 'ongoing'."""
    ids = _paper_ids(6)
    eras = [
        _era("past", ids[:3], start=2014, end=2020),
        _era("frontier", ids[3:], start=2021, end=None),
    ]
    syntheses = [_synth(i) for i in ids]
    client = _fake_client(_fake_message([_tool_use_block(_summary_dict())]))
    agent = ExecutiveSummaryAgent(client)

    await agent.summarize(topic="t", eras=eras, syntheses=syntheses)

    body: str = client.messages.create.await_args.kwargs["messages"][0]["content"]
    assert "year range: 2021-ongoing" in body


# ───── Refusal ──────────────────────────────────────────────────────────


async def test_summarize_raises_refusal_on_no_tool_use_block() -> None:
    eras, syntheses = _two_eras_eight_papers()
    response = _fake_message(
        [_text_block("I cannot complete this task.")], stop_reason="refusal"
    )
    agent = ExecutiveSummaryAgent(_fake_client(response))

    with pytest.raises(ExecutiveSummaryRefusedError) as exc_info:
        await agent.summarize(topic="t", eras=eras, syntheses=syntheses)
    assert exc_info.value.stop_reason == "refusal"
    assert "I cannot complete" in exc_info.value.content_summary


async def test_summarize_raises_refusal_on_wrong_tool_name() -> None:
    eras, syntheses = _two_eras_eight_papers()
    response = _fake_message([_tool_use_block(_summary_dict(), name="other_tool")])
    agent = ExecutiveSummaryAgent(_fake_client(response))

    with pytest.raises(ExecutiveSummaryRefusedError):
        await agent.summarize(topic="t", eras=eras, syntheses=syntheses)


# ───── Schema validation ───────────────────────────────────────────────


async def test_summarize_raises_on_summary_too_short() -> None:
    eras, syntheses = _two_eras_eight_papers()
    response = _fake_message([_tool_use_block(_summary_dict("Too short."))])
    agent = ExecutiveSummaryAgent(_fake_client(response))

    with pytest.raises(ExecutiveSummaryInvalidOutputError, match="schema validation"):
        await agent.summarize(topic="t", eras=eras, syntheses=syntheses)


async def test_summarize_raises_on_summary_too_long() -> None:
    eras, syntheses = _two_eras_eight_papers()
    too_long = "word " * 600  # 3000 chars, exceeds 2500 cap
    response = _fake_message([_tool_use_block(_summary_dict(too_long))])
    agent = ExecutiveSummaryAgent(_fake_client(response))

    with pytest.raises(ExecutiveSummaryInvalidOutputError, match="schema validation"):
        await agent.summarize(topic="t", eras=eras, syntheses=syntheses)


async def test_summarize_raises_on_confidence_out_of_range() -> None:
    eras, syntheses = _two_eras_eight_papers()
    response = _fake_message([_tool_use_block(_summary_dict(confidence=1.5))])
    agent = ExecutiveSummaryAgent(_fake_client(response))

    with pytest.raises(ExecutiveSummaryInvalidOutputError, match="schema validation"):
        await agent.summarize(topic="t", eras=eras, syntheses=syntheses)


async def test_summarize_raises_on_negative_confidence() -> None:
    eras, syntheses = _two_eras_eight_papers()
    response = _fake_message([_tool_use_block(_summary_dict(confidence=-0.1))])
    agent = ExecutiveSummaryAgent(_fake_client(response))

    with pytest.raises(ExecutiveSummaryInvalidOutputError, match="schema validation"):
        await agent.summarize(topic="t", eras=eras, syntheses=syntheses)


async def test_summarize_accepts_confidence_boundary_values() -> None:
    """0.0 and 1.0 are inclusive bounds and should pass."""
    eras, syntheses = _two_eras_eight_papers()
    for value in (0.0, 1.0):
        response = _fake_message(
            [_tool_use_block(_summary_dict(confidence=value))]
        )
        agent = ExecutiveSummaryAgent(_fake_client(response))
        result = await agent.summarize(topic="t", eras=eras, syntheses=syntheses)
        assert result.confidence == value


# ───── Defensive: empty / orphan inputs ─────────────────────────────────


async def test_summarize_raises_on_empty_eras() -> None:
    syntheses = [_synth(i) for i in _paper_ids(4)]
    client = _fake_client(_fake_message([]))
    agent = ExecutiveSummaryAgent(client)

    with pytest.raises(ExecutiveSummaryError, match="no eras"):
        await agent.summarize(topic="t", eras=[], syntheses=syntheses)
    client.messages.create.assert_not_called()


async def test_summarize_raises_on_empty_syntheses() -> None:
    eras = [_era("a", _paper_ids(2)), _era("b", [*_paper_ids(2)[:1], "1706.00003"])]
    client = _fake_client(_fake_message([]))
    agent = ExecutiveSummaryAgent(client)

    with pytest.raises(ExecutiveSummaryError, match="no syntheses"):
        await agent.summarize(topic="t", eras=eras, syntheses=[])
    client.messages.create.assert_not_called()


async def test_summarize_raises_on_synthesis_orphaned_from_eras() -> None:
    """A synthesis whose arxiv_id appears in no era is a stage-4 bug."""
    eras_ids = _paper_ids(4)
    eras = [_era("a", eras_ids)]
    # 5th synthesis has no era assignment.
    extra_id = "1706.99999"
    syntheses = [_synth(i) for i in [*eras_ids, extra_id]]
    client = _fake_client(_fake_message([]))
    agent = ExecutiveSummaryAgent(client)

    with pytest.raises(ExecutiveSummaryError, match="without an era"):
        await agent.summarize(topic="t", eras=eras, syntheses=syntheses)
    client.messages.create.assert_not_called()
