"""Tests for ``arch_01`` stage 1: the agentic ``SearchAgent`` loop.

Two layers of mocking:
    - ``arxiv_search`` is patched at the search-module's import site so
      the loop drives over predetermined per-call results.
    - ``AsyncAnthropic.messages.create`` is replaced with an ``AsyncMock``
      whose ``side_effect`` is a pre-staged list of ``Message``-shaped
      responses, one per loop iteration.

The agentic loop is exercised end-to-end without any network I/O.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import HttpUrl

from papertrail.architectures.arch_01_sequential.search import (
    MAX_SEARCH_ITERATIONS,
    MIN_PAPERS_TO_PROCEED,
    SearchAgent,
    SearchInsufficientResultsError,
)
from papertrail.tools.arxiv import ArxivPaper

# ``asyncio_mode = "auto"`` in pyproject.toml auto-detects async tests;
# no explicit ``pytestmark`` needed (and applying it to the sync tests
# below would trigger pytest warnings).


# ───── Builders ─────────────────────────────────────────────────────────


def _arxiv_paper(idx: int, *, year: int = 2020) -> ArxivPaper:
    """Build a valid ``ArxivPaper`` with a distinct ``arxiv_id`` per ``idx``."""
    return ArxivPaper(
        arxiv_id=f"1706.{idx:05d}",
        title=f"Paper {idx}",
        authors=["Doe, J."],
        categories=["cs.LG"],
        abstract=f"Abstract sentence one for paper {idx}. Abstract sentence two.",
        entry_url=HttpUrl(f"https://arxiv.org/abs/1706.{idx:05d}"),
        pdf_url=HttpUrl(f"https://arxiv.org/pdf/1706.{idx:05d}"),
        published=datetime(year, 6, 1, tzinfo=UTC),
        updated=datetime(year, 6, 1, tzinfo=UTC),
    )


def _tool_use_block(query: str, *, max_results: int = 10, tool_use_id: str = "tu_1") -> MagicMock:
    """Build a fake content block that looks like an SDK ``ToolUseBlock``."""
    block = MagicMock()
    block.type = "tool_use"
    block.name = "arxiv_search"
    block.id = tool_use_id
    block.input = {"query": query, "max_results": max_results}
    return block


def _text_block(text: str = "Coverage adequate; ending.") -> MagicMock:
    block = MagicMock()
    block.type = "text"
    block.text = text
    return block


def _fake_message(
    content: list[MagicMock],
    *,
    stop_reason: str,
    input_tokens: int = 100,
    output_tokens: int = 30,
) -> MagicMock:
    """Build a fake Anthropic ``Message`` response.

    ``stop_reason`` and ``usage`` are the two fields the agent actually
    reads; the rest is dummied. ``content`` is a list of fake blocks
    (``tool_use`` and/or ``text``).
    """
    msg = MagicMock()
    msg.content = content
    msg.stop_reason = stop_reason
    usage = MagicMock()
    usage.input_tokens = input_tokens
    usage.output_tokens = output_tokens
    msg.usage = usage
    return msg


def _fake_client(responses: list[MagicMock]) -> MagicMock:
    """Fake ``AsyncAnthropic`` whose ``messages.create`` returns ``responses`` in order."""
    client = MagicMock()
    client.messages = MagicMock()
    client.messages.create = AsyncMock(side_effect=responses)
    return client


def _patch_arxiv_search(
    side_effect: list[list[ArxivPaper]],
) -> AbstractContextManager[AsyncMock]:
    """Patch ``arxiv_search`` at the search-module's import site.

    Each call to ``arxiv_search`` returns the next list in ``side_effect``;
    callers should pre-stage one list per ``tool_use`` block they expect
    to be processed.
    """
    return patch(
        "papertrail.architectures.arch_01_sequential.search.arxiv_search",
        new=AsyncMock(side_effect=side_effect),
    )


# ───── Class-level wiring ───────────────────────────────────────────────


def test_classvars_default_to_haiku_and_v1_prompt() -> None:
    """Defaults match ADR-0003 (Haiku) and the v1 prompt slug."""
    assert SearchAgent.DEFAULT_MODEL == "claude-haiku-4-5"
    assert SearchAgent.PROMPT_NAME == "arch_01_search"
    assert SearchAgent.PROMPT_VERSION == "v1"


def test_constructor_does_not_load_prompt() -> None:
    """Prompt is lazy-loaded so constructing an agent never touches disk."""
    agent = SearchAgent(_fake_client([]))
    # Access the private attribute directly to assert the cache is empty.
    assert agent._system_prompt is None


def test_model_override_is_respected() -> None:
    """A constructor ``model`` override is reflected in ``.model``."""
    agent = SearchAgent(_fake_client([]), model="claude-haiku-4-5-20251001")
    assert agent.model == "claude-haiku-4-5-20251001"


# ───── Happy path: clean ``end_turn`` exit ──────────────────────────────


async def test_search_happy_path_returns_deduped_papers() -> None:
    """Two tool_use calls, then end_turn; papers across both calls flow out."""
    call_one = [_arxiv_paper(i) for i in range(5)]
    call_two = [_arxiv_paper(i) for i in range(5, 9)]
    responses = [
        _fake_message([_tool_use_block("q1")], stop_reason="tool_use"),
        _fake_message([_tool_use_block("q2", tool_use_id="tu_2")], stop_reason="tool_use"),
        _fake_message([_text_block()], stop_reason="end_turn"),
    ]
    agent = SearchAgent(_fake_client(responses))

    with _patch_arxiv_search([call_one, call_two]):
        result = await agent.search("transformer attention")

    # 9 distinct papers across the two calls.
    assert len(result.papers) == 9
    assert [p.arxiv_id for p in result.papers] == [f"1706.{i:05d}" for i in range(9)]
    assert result.telemetry.iterations_used == 3
    assert result.telemetry.queries_issued == ("q1", "q2")
    assert result.telemetry.final_stop_reason == "end_turn"
    assert result.telemetry.recovered is False
    assert result.error_records == ()


async def test_search_deduplicates_across_calls() -> None:
    """A paper returned by two queries appears once; queries list still has both."""
    overlap_paper = _arxiv_paper(0)
    call_one = [overlap_paper, _arxiv_paper(1), _arxiv_paper(2), _arxiv_paper(3)]
    call_two = [overlap_paper, _arxiv_paper(4)]  # 0 is a dup; 4 is new
    responses = [
        _fake_message([_tool_use_block("q1")], stop_reason="tool_use"),
        _fake_message([_tool_use_block("q2", tool_use_id="tu_2")], stop_reason="tool_use"),
        _fake_message([_text_block()], stop_reason="end_turn"),
    ]
    agent = SearchAgent(_fake_client(responses))

    with _patch_arxiv_search([call_one, call_two]):
        result = await agent.search("attention")

    # 4 unique papers (0,1,2,3,4 minus the dup of 0).
    assert {p.arxiv_id for p in result.papers} == {f"1706.{i:05d}" for i in range(5)}
    assert len(result.papers) == 5
    assert len(result.telemetry.queries_issued) == 2  # both queries recorded


# ───── Cost and token accounting ────────────────────────────────────────


async def test_search_sums_tokens_across_iterations() -> None:
    """``usage.input_tokens`` / ``output_tokens`` accumulate across the whole loop."""
    responses = [
        _fake_message(
            [_tool_use_block("q1")],
            stop_reason="tool_use",
            input_tokens=200,
            output_tokens=60,
        ),
        _fake_message(
            [_text_block()],
            stop_reason="end_turn",
            input_tokens=350,  # the tool_result gets billed too
            output_tokens=15,
        ),
    ]
    agent = SearchAgent(_fake_client(responses))

    with _patch_arxiv_search([[_arxiv_paper(i) for i in range(4)]]):
        result = await agent.search("t")

    assert result.telemetry.input_tokens == 550
    assert result.telemetry.output_tokens == 75
    # Haiku 4.5: 550 x $1/M + 75 x $5/M = $0.00055 + $0.000375 = $0.000925.
    expected = 550 / 1e6 * 1.0 + 75 / 1e6 * 5.0
    assert result.telemetry.cost_usd == pytest.approx(expected)


async def test_search_unknown_model_falls_back_to_zero_cost() -> None:
    """Pricing miss returns 0.0 so a stale table is visible, not crashing."""
    responses = [
        _fake_message([_tool_use_block("q1")], stop_reason="tool_use"),
        _fake_message([_text_block()], stop_reason="end_turn"),
    ]
    agent = SearchAgent(_fake_client(responses), model="claude-unknown-1-0")

    with _patch_arxiv_search([[_arxiv_paper(i) for i in range(4)]]):
        result = await agent.search("t")

    assert result.telemetry.cost_usd == 0.0


# ───── Partial-results recovery (D5 TS 5.3) ─────────────────────────────


async def test_search_recovers_when_cap_hits_with_enough_papers() -> None:
    """Hit ``max_iterations`` with ≥4 papers → recovered=True + ErrorRecord."""
    # max_iterations=1: after the one tool_use turn, we never get to ask the
    # model "are you done?". So we exit on cap, but we have 5 unique papers,
    # which is above the floor.
    responses = [_fake_message([_tool_use_block("q1")], stop_reason="tool_use")]
    agent = SearchAgent(_fake_client(responses), max_iterations=1)

    with _patch_arxiv_search([[_arxiv_paper(i) for i in range(5)]]):
        result = await agent.search("t")

    assert len(result.papers) == 5
    assert result.telemetry.recovered is True
    # final_stop_reason carries forward whatever the *last* response had,
    # which is tool_use (we never got a synthesis turn).
    assert result.telemetry.final_stop_reason == "tool_use"
    assert len(result.error_records) == 1
    rec = result.error_records[0]
    assert rec.step_index == 1
    assert rec.category == "api"
    assert rec.recovered is True
    assert "tool_use" in rec.message  # the stop reason is named in the message


async def test_search_recovers_on_refusal_with_enough_papers() -> None:
    """Refusal after enough papers → recovered=True (matches the cap case)."""
    responses = [
        _fake_message([_tool_use_block("q1")], stop_reason="tool_use"),
        _fake_message([_text_block("I can't continue.")], stop_reason="refusal"),
    ]
    agent = SearchAgent(_fake_client(responses))

    with _patch_arxiv_search([[_arxiv_paper(i) for i in range(6)]]):
        result = await agent.search("t")

    assert result.telemetry.final_stop_reason == "refusal"
    assert result.telemetry.recovered is True
    assert len(result.error_records) == 1
    assert "refusal" in result.error_records[0].message


# ───── Insufficient-results failures ────────────────────────────────────


async def test_search_raises_when_cap_hits_with_too_few_papers() -> None:
    """Hit cap with <``MIN_PAPERS_TO_PROCEED`` papers → raise, no recovery."""
    responses = [_fake_message([_tool_use_block("q1")], stop_reason="tool_use")]
    agent = SearchAgent(_fake_client(responses), max_iterations=1)

    with (
        _patch_arxiv_search([[_arxiv_paper(0), _arxiv_paper(1)]]),  # only 2 papers
        pytest.raises(SearchInsufficientResultsError) as exc_info,
    ):
        await agent.search("niche-topic")

    err = exc_info.value
    assert err.topic == "niche-topic"
    assert err.found == 2
    assert err.stop_reason == "tool_use"


async def test_search_raises_when_end_turn_fires_with_too_few_papers() -> None:
    """``end_turn`` with <4 papers still fails — schema floor will reject it anyway."""
    # Model decides it's done after one call returning only 3 papers.
    responses = [
        _fake_message([_tool_use_block("q1")], stop_reason="tool_use"),
        _fake_message([_text_block("Done; topic is very niche.")], stop_reason="end_turn"),
    ]
    agent = SearchAgent(_fake_client(responses))

    with (
        _patch_arxiv_search([[_arxiv_paper(i) for i in range(3)]]),
        pytest.raises(SearchInsufficientResultsError) as exc_info,
    ):
        await agent.search("ultra-niche topic")

    assert exc_info.value.found == 3
    assert exc_info.value.stop_reason == "end_turn"


async def test_search_end_turn_at_exact_floor_succeeds() -> None:
    """Exactly 4 unique papers + ``end_turn`` → clean SearchResult, no recovery."""
    responses = [
        _fake_message([_tool_use_block("q1")], stop_reason="tool_use"),
        _fake_message([_text_block()], stop_reason="end_turn"),
    ]
    agent = SearchAgent(_fake_client(responses))

    with _patch_arxiv_search([[_arxiv_paper(i) for i in range(MIN_PAPERS_TO_PROCEED)]]):
        result = await agent.search("thin-data topic")

    assert len(result.papers) == MIN_PAPERS_TO_PROCEED
    assert result.telemetry.recovered is False
    assert result.error_records == ()


# ───── Compact tool_result format ───────────────────────────────────────


def _find_tool_result_texts(recorded_messages: list[dict[str, Any]]) -> list[str]:
    """Scan a recorded ``messages`` list for tool_result block bodies.

    The agent passes the same ``messages`` list-object to every
    ``client.messages.create`` call and continues mutating it afterwards.
    ``AsyncMock.await_args_list`` records the *reference*, not a snapshot,
    so positional indexing into ``messages[-1]`` post-run sees the final
    state of the loop rather than the state at that call. Scanning for
    blocks by ``type == "tool_result"`` avoids that pitfall.
    """
    out: list[str] = []
    for msg in recorded_messages:
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                out.append(str(block["content"]))
    return out


async def test_search_hands_back_compact_tool_result_with_signal_counts() -> None:
    """The tool_result body carries 'N new / M already seen' + per-paper lines."""
    responses = [
        _fake_message([_tool_use_block("q1")], stop_reason="tool_use"),
        _fake_message([_text_block()], stop_reason="end_turn"),
    ]
    client = _fake_client(responses)
    agent = SearchAgent(client)

    with _patch_arxiv_search([[_arxiv_paper(i) for i in range(5)]]):
        await agent.search("t")

    # Scan the loop's final messages for the tool_result block we appended.
    recorded = client.messages.create.await_args_list[-1].kwargs["messages"]
    tool_result_texts = _find_tool_result_texts(recorded)
    assert len(tool_result_texts) == 1
    text = tool_result_texts[0]
    assert 'Query: "q1"' in text
    assert "5 new" in text
    assert "0 already seen" in text
    assert "Total unique papers so far: 5" in text
    # Per-paper line carries arxiv_id, year, primary_category.
    assert "1706.00000" in text
    assert "2020, cs.LG" in text


async def test_search_compact_format_notes_zero_new_papers() -> None:
    """A call returning only duplicates produces an explicit '(No new papers)' note."""
    overlap = [_arxiv_paper(i) for i in range(4)]
    responses = [
        _fake_message([_tool_use_block("q1")], stop_reason="tool_use"),
        _fake_message([_tool_use_block("q2", tool_use_id="tu_2")], stop_reason="tool_use"),
        _fake_message([_text_block()], stop_reason="end_turn"),
    ]
    client = _fake_client(responses)
    agent = SearchAgent(client)

    with _patch_arxiv_search([overlap, overlap]):  # second call: all dups
        await agent.search("t")

    recorded = client.messages.create.await_args_list[-1].kwargs["messages"]
    tool_result_texts = _find_tool_result_texts(recorded)
    # Two tool_result blocks total (one per tool_use call).
    assert len(tool_result_texts) == 2
    dup_text = tool_result_texts[1]
    assert "0 new" in dup_text
    assert "4 already seen" in dup_text
    assert "(No new papers in this call.)" in dup_text


# ───── SDK wiring ───────────────────────────────────────────────────────


async def test_search_passes_haiku_and_one_tool_to_sdk() -> None:
    """The first call goes to Haiku with exactly one tool: ``arxiv_search``."""
    responses = [
        _fake_message([_tool_use_block("q1")], stop_reason="tool_use"),
        _fake_message([_text_block()], stop_reason="end_turn"),
    ]
    client = _fake_client(responses)
    agent = SearchAgent(client)

    with _patch_arxiv_search([[_arxiv_paper(i) for i in range(4)]]):
        await agent.search("t")

    first_kwargs = client.messages.create.await_args_list[0].kwargs
    assert first_kwargs["model"] == "claude-haiku-4-5"
    assert len(first_kwargs["tools"]) == 1
    tool = first_kwargs["tools"][0]
    assert tool["name"] == "arxiv_search"
    # max_results is capped at 20 — not the full arxiv ceiling.
    assert tool["input_schema"]["properties"]["max_results"]["maximum"] == 20
    # No tool_choice forcing — the model is allowed to emit a final text turn.
    assert "tool_choice" not in first_kwargs


async def test_search_includes_topic_as_initial_user_message() -> None:
    """The topic string is the first user message handed to the model."""
    responses = [
        _fake_message([_tool_use_block("q1")], stop_reason="tool_use"),
        _fake_message([_text_block()], stop_reason="end_turn"),
    ]
    client = _fake_client(responses)
    agent = SearchAgent(client)

    with _patch_arxiv_search([[_arxiv_paper(i) for i in range(4)]]):
        await agent.search("the paragraph topic")

    first_kwargs = client.messages.create.await_args_list[0].kwargs
    messages: list[dict[str, Any]] = first_kwargs["messages"]
    assert messages[0] == {"role": "user", "content": "the paragraph topic"}


async def test_search_unknown_tool_call_raises() -> None:
    """A ``tool_use`` block naming an unknown tool is a programming error."""
    rogue_block = MagicMock()
    rogue_block.type = "tool_use"
    rogue_block.name = "not_arxiv_search"
    rogue_block.id = "tu_x"
    rogue_block.input = {}
    responses = [_fake_message([rogue_block], stop_reason="tool_use")]
    agent = SearchAgent(_fake_client(responses))

    with (
        _patch_arxiv_search([]),  # arxiv_search never called
        pytest.raises(RuntimeError, match="unexpected tool call"),
    ):
        await agent.search("t")


# ───── Module-level invariants ──────────────────────────────────────────


def test_constants_match_adr_0005_and_0006() -> None:
    """``MIN_PAPERS_TO_PROCEED`` reflects ADR-0006 (4), cap is the ADR-0005 (8)."""
    assert MIN_PAPERS_TO_PROCEED == 4
    assert MAX_SEARCH_ITERATIONS == 8
