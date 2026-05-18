"""Tests for ``arch_01`` stage 2: ``TriageAgent``.

The triage stage is a single forced ``tool_use`` call, so the mocking is
simpler than the SearchAgent's loop — one ``AsyncMock`` for the client
returning one ``Message``-shaped response. Tests cover the four failure
modes (refusal, schema invalid, cross-field invalid, quality gate) plus
the happy path and audit-trail integrity.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import HttpUrl

from papertrail.architectures.arch_01_sequential.triage import (
    INCLUDED_MAX,
    INCLUDED_MIN,
    TriageAgent,
    TriageInsufficientQualityError,
    TriageInvalidOutputError,
    TriageRefusedError,
    TriageSelection,
)
from papertrail.tools.arxiv import ArxivPaper

# ───── Builders ─────────────────────────────────────────────────────────


def _arxiv_paper(idx: int, *, year: int = 2020) -> ArxivPaper:
    """Build a valid ``ArxivPaper`` with a distinct ``arxiv_id`` per ``idx``."""
    return ArxivPaper(
        arxiv_id=f"1706.{idx:05d}",
        title=f"Paper {idx}",
        authors=["Doe, J."],
        categories=["cs.LG"],
        abstract=(
            f"This is the abstract for paper {idx}. It contains multiple "
            "sentences so the triage stage has something to read."
        ),
        entry_url=HttpUrl(f"https://arxiv.org/abs/1706.{idx:05d}"),
        pdf_url=HttpUrl(f"https://arxiv.org/pdf/1706.{idx:05d}"),
        published=datetime(year, 6, 1, tzinfo=UTC),
        updated=datetime(year, 6, 1, tzinfo=UTC),
    )


def _candidates(n: int) -> list[ArxivPaper]:
    """Build ``n`` candidate papers."""
    return [_arxiv_paper(i) for i in range(n)]


def _decision(arxiv_id: str, verdict: str, reason: str = "test reason") -> dict[str, str]:
    """Build a raw decision dict (the model-emitted shape)."""
    return {"arxiv_id": arxiv_id, "verdict": verdict, "reason": reason}


def _tool_use_block(tool_input: dict[str, Any], *, name: str = "submit_triage") -> MagicMock:
    """Build a fake content block that looks like an SDK ``ToolUseBlock``."""
    block = MagicMock()
    block.type = "tool_use"
    block.name = name
    block.id = "tu_1"
    block.input = tool_input
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
    input_tokens: int = 5000,
    output_tokens: int = 1000,
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


def _make_selection_input(candidates: list[ArxivPaper], *, include_first_n: int) -> dict[str, Any]:
    """Build a valid ``TriageSelection`` input dict: include first N, reject rest."""
    decisions = []
    for idx, p in enumerate(candidates):
        verdict = "included" if idx < include_first_n else "rejected"
        decisions.append(_decision(p.arxiv_id, verdict, f"reason for {p.arxiv_id}"))
    return {"decisions": decisions}


# ───── Class-level wiring ───────────────────────────────────────────────


def test_classvars_default_to_haiku_and_v1_prompt() -> None:
    """Defaults match ADR-0003 (Haiku) and the v1 prompt slug."""
    assert TriageAgent.DEFAULT_MODEL == "claude-haiku-4-5"
    assert TriageAgent.PROMPT_NAME == "arch_01_triage"
    assert TriageAgent.PROMPT_VERSION == "v1"
    assert TriageAgent.TOOL_NAME == "submit_triage"


def test_constructor_does_not_load_prompt() -> None:
    """Prompt is lazy-loaded so constructing an agent never touches disk."""
    agent = TriageAgent(_fake_client(_fake_message([])))
    assert agent._system_prompt is None


def test_quality_constants_match_schema_bounds() -> None:
    """``INCLUDED_MIN/MAX`` reflect the ``Deliverable.papers`` schema bounds."""
    # ADR-0006 sets papers floor=4, ceiling=12 — triage must use the same.
    assert INCLUDED_MIN == 4
    assert INCLUDED_MAX == 12


# ───── Happy path ───────────────────────────────────────────────────────


async def test_triage_happy_path_returns_selected_and_records() -> None:
    """8-of-15 inclusion → 8 selected papers + 15 CandidateRecord entries."""
    candidates = _candidates(15)
    selection_input = _make_selection_input(candidates, include_first_n=8)
    response = _fake_message([_tool_use_block(selection_input)])
    agent = TriageAgent(_fake_client(response))

    result = await agent.triage(topic="self-attention", candidates=candidates)

    assert len(result.selected) == 8
    # Model-emitted order is preserved on selected.
    assert [p.arxiv_id for p in result.selected] == [candidates[i].arxiv_id for i in range(8)]
    # Every candidate gets a CandidateRecord with title carried forward.
    assert len(result.candidate_records) == 15
    rec_by_id = {r.arxiv_id: r for r in result.candidate_records}
    for c in candidates:
        assert rec_by_id[c.arxiv_id].title == c.title
    # Included verdict count matches selection.
    included = [r for r in result.candidate_records if r.verdict == "included"]
    assert len(included) == 8


async def test_triage_records_token_usage_and_cost() -> None:
    """Usage tokens come from response.usage; cost uses Haiku rates."""
    candidates = _candidates(10)
    selection_input = _make_selection_input(candidates, include_first_n=5)
    response = _fake_message(
        [_tool_use_block(selection_input)],
        input_tokens=8000,
        output_tokens=1500,
    )
    agent = TriageAgent(_fake_client(response))

    result = await agent.triage(topic="t", candidates=candidates)

    assert result.usage.input_tokens == 8000
    assert result.usage.output_tokens == 1500
    # Haiku 4.5: 8000 / 1e6 * 1 + 1500 / 1e6 * 5 = 0.008 + 0.0075 = 0.0155
    expected = 8000 / 1e6 * 1.0 + 1500 / 1e6 * 5.0
    assert result.usage.cost_usd == pytest.approx(expected)


async def test_triage_at_exact_floor_succeeds() -> None:
    """Exactly INCLUDED_MIN=4 papers selected → passes the quality gate."""
    candidates = _candidates(10)
    selection_input = _make_selection_input(candidates, include_first_n=INCLUDED_MIN)
    response = _fake_message([_tool_use_block(selection_input)])
    agent = TriageAgent(_fake_client(response))

    result = await agent.triage(topic="t", candidates=candidates)
    assert len(result.selected) == INCLUDED_MIN


# ───── SDK wiring ───────────────────────────────────────────────────────


async def test_triage_forces_tool_use_and_passes_haiku_model() -> None:
    """``messages.create`` receives Haiku + the submit_triage tool + tool_choice forced."""
    candidates = _candidates(10)
    selection_input = _make_selection_input(candidates, include_first_n=5)
    client = _fake_client(_fake_message([_tool_use_block(selection_input)]))
    agent = TriageAgent(client)

    await agent.triage(topic="t", candidates=candidates)

    kwargs = client.messages.create.await_args.kwargs
    assert kwargs["model"] == "claude-haiku-4-5"
    assert kwargs["tool_choice"] == {"type": "tool", "name": "submit_triage"}
    assert len(kwargs["tools"]) == 1
    tool = kwargs["tools"][0]
    assert tool["name"] == "submit_triage"
    # input_schema is derived from TriageSelection — type=object with
    # a `decisions` property of type=array.
    assert tool["input_schema"] == TriageSelection.model_json_schema()


async def test_triage_user_message_renders_candidates_with_abstracts() -> None:
    """The user message carries every candidate's full abstract."""
    candidates = _candidates(3)
    selection_input = _make_selection_input(candidates, include_first_n=INCLUDED_MIN)
    # Bump to 4 included to clear the floor.
    candidates = _candidates(5)
    selection_input = _make_selection_input(candidates, include_first_n=INCLUDED_MIN)
    client = _fake_client(_fake_message([_tool_use_block(selection_input)]))
    agent = TriageAgent(client)

    await agent.triage(topic="the topic paragraph", candidates=candidates)

    kwargs = client.messages.create.await_args.kwargs
    user_message = kwargs["messages"][0]
    assert user_message["role"] == "user"
    body = user_message["content"]
    assert "the topic paragraph" in body
    for c in candidates:
        assert f"arxiv_id={c.arxiv_id}" in body
        assert c.title in body
        # Full abstract present (the first-sentence-only stage-1 compaction
        # does NOT apply here — triage needs full text per the prompt).
        assert c.abstract in body


# ───── Failure: refusal ─────────────────────────────────────────────────


async def test_triage_raises_when_model_returns_no_tool_use() -> None:
    """Response with only a text block → TriageRefusedError carrying that text."""
    candidates = _candidates(10)
    response = _fake_message(
        [_text_block("I cannot complete this task.")],
        stop_reason="refusal",
    )
    agent = TriageAgent(_fake_client(response))

    with pytest.raises(TriageRefusedError) as exc_info:
        await agent.triage(topic="t", candidates=candidates)

    assert exc_info.value.stop_reason == "refusal"
    assert "I cannot complete" in exc_info.value.content_summary


async def test_triage_raises_when_tool_name_is_different() -> None:
    """tool_use block with a wrong name is treated as refusal, not silent."""
    candidates = _candidates(10)
    selection_input = _make_selection_input(candidates, include_first_n=5)
    # Same input, but called with the wrong tool name.
    response = _fake_message([_tool_use_block(selection_input, name="some_other_tool")])
    agent = TriageAgent(_fake_client(response))

    with pytest.raises(TriageRefusedError):
        await agent.triage(topic="t", candidates=candidates)


# ───── Failure: schema invalid (pydantic) ───────────────────────────────


async def test_triage_raises_on_invalid_verdict_value() -> None:
    """A verdict outside the Literal set fails pydantic → TriageInvalidOutputError."""
    candidates = _candidates(10)
    bogus_input = {
        "decisions": [
            {"arxiv_id": candidates[0].arxiv_id, "verdict": "maybe", "reason": "huh"},
            *[_decision(c.arxiv_id, "rejected") for c in candidates[1:]],
        ]
    }
    response = _fake_message([_tool_use_block(bogus_input)])
    agent = TriageAgent(_fake_client(response))

    with pytest.raises(TriageInvalidOutputError, match="schema validation"):
        await agent.triage(topic="t", candidates=candidates)


async def test_triage_raises_on_malformed_arxiv_id() -> None:
    """A regex-invalid arxiv_id in decisions fails pydantic validation."""
    candidates = _candidates(10)
    bogus_input = {
        "decisions": [
            {"arxiv_id": "hep-th/9901001", "verdict": "included", "reason": "old id"},
            *[_decision(c.arxiv_id, "rejected") for c in candidates],
        ]
    }
    response = _fake_message([_tool_use_block(bogus_input)])
    agent = TriageAgent(_fake_client(response))

    with pytest.raises(TriageInvalidOutputError, match="schema validation"):
        await agent.triage(topic="t", candidates=candidates)


# ───── Failure: cross-field validators ──────────────────────────────────


async def test_triage_raises_on_fabricated_arxiv_id() -> None:
    """A decision for an id not in the input set is rejected."""
    candidates = _candidates(10)
    # Replace candidates[0]'s decision with a fabricated id.
    decisions = [_decision(c.arxiv_id, "rejected") for c in candidates]
    decisions.append(_decision("9999.99999", "included", "made-up paper"))
    response = _fake_message([_tool_use_block({"decisions": decisions})])
    agent = TriageAgent(_fake_client(response))

    with pytest.raises(TriageInvalidOutputError, match="not in the input"):
        await agent.triage(topic="t", candidates=candidates)


async def test_triage_raises_on_duplicate_decision() -> None:
    """Two decisions for the same arxiv_id is invalid."""
    candidates = _candidates(10)
    decisions = [_decision(c.arxiv_id, "rejected") for c in candidates]
    # Add a duplicate of the first.
    decisions.append(_decision(candidates[0].arxiv_id, "included", "dup"))
    response = _fake_message([_tool_use_block({"decisions": decisions})])
    agent = TriageAgent(_fake_client(response))

    with pytest.raises(TriageInvalidOutputError, match="duplicate decisions"):
        await agent.triage(topic="t", candidates=candidates)


async def test_triage_raises_on_missing_decision() -> None:
    """Skipping a candidate from the decisions list is invalid."""
    candidates = _candidates(10)
    # Decisions for only the first 9 candidates.
    decisions = [_decision(c.arxiv_id, "rejected") for c in candidates[:9]]
    decisions[0]["verdict"] = "included"
    response = _fake_message([_tool_use_block({"decisions": decisions})])
    agent = TriageAgent(_fake_client(response))

    with pytest.raises(TriageInvalidOutputError, match="missed 1 input"):
        await agent.triage(topic="t", candidates=candidates)


async def test_triage_raises_when_included_exceeds_ceiling() -> None:
    """`included` count > INCLUDED_MAX (12) is rejected."""
    candidates = _candidates(15)
    selection_input = _make_selection_input(candidates, include_first_n=13)
    response = _fake_message([_tool_use_block(selection_input)])
    agent = TriageAgent(_fake_client(response))

    with pytest.raises(TriageInvalidOutputError, match="ceiling is 12"):
        await agent.triage(topic="t", candidates=candidates)


# ───── Failure: quality gate ────────────────────────────────────────────


async def test_triage_raises_when_included_below_quality_floor() -> None:
    """`included` count below INCLUDED_MIN (4) → TriageInsufficientQualityError."""
    candidates = _candidates(15)
    selection_input = _make_selection_input(candidates, include_first_n=3)
    response = _fake_message([_tool_use_block(selection_input)])
    agent = TriageAgent(_fake_client(response))

    with pytest.raises(TriageInsufficientQualityError) as exc_info:
        await agent.triage(topic="niche topic", candidates=candidates)

    assert exc_info.value.included == 3
    assert exc_info.value.total == 15
    assert exc_info.value.topic == "niche topic"


async def test_triage_quality_gate_distinct_from_invalid_output() -> None:
    """A valid schema with 0 included still raises the *quality* error, not invalid."""
    candidates = _candidates(15)
    # All rejected — valid schema, but no inclusions.
    selection_input = _make_selection_input(candidates, include_first_n=0)
    response = _fake_message([_tool_use_block(selection_input)])
    agent = TriageAgent(_fake_client(response))

    # MUST be TriageInsufficientQualityError, not TriageInvalidOutputError.
    with pytest.raises(TriageInsufficientQualityError):
        await agent.triage(topic="t", candidates=candidates)


# ───── Defensive: empty input ───────────────────────────────────────────


async def test_triage_raises_immediately_on_empty_candidates() -> None:
    """Calling triage with no candidates is a stage-1 bug; surface without API call."""
    client = _fake_client(_fake_message([]))
    agent = TriageAgent(client)

    with pytest.raises(TriageInvalidOutputError, match="no candidates"):
        await agent.triage(topic="t", candidates=[])

    # Most important: we did NOT call the API for nothing.
    client.messages.create.assert_not_called()
