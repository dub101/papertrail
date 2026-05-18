"""Tests for ``arch_01`` stage 4: ``EraPartitionAgent``.

Same single-forced-tool-use mocking pattern as the triage tests. The
extra surface area here is the four cross-field invariants (fabricated
ids, papers in multiple eras, duplicate era_ids, missing papers) and
the year-range / narrative-length schema constraints.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import HttpUrl, ValidationError

from papertrail.architectures.arch_01_sequential.era_partition import (
    ERAS_MAX,
    ERAS_MIN,
    EraEntry,
    EraPartition,
    EraPartitionAgent,
    EraPartitionInvalidOutputError,
    EraPartitionRefusedError,
)
from papertrail.architectures.arch_01_sequential.synthesis import PaperSynthesis
from papertrail.tools.arxiv import ArxivPaper

# ───── Builders ─────────────────────────────────────────────────────────


def _arxiv_paper(idx: int, *, year: int = 2020) -> ArxivPaper:
    return ArxivPaper(
        arxiv_id=f"1706.{idx:05d}",
        title=f"Paper {idx}",
        authors=["Doe, J."],
        categories=["cs.LG"],
        abstract=f"Abstract for paper {idx}. Second sentence.",
        entry_url=HttpUrl(f"https://arxiv.org/abs/1706.{idx:05d}"),
        pdf_url=HttpUrl(f"https://arxiv.org/pdf/1706.{idx:05d}"),
        published=datetime(year, 6, 1, tzinfo=UTC),
        updated=datetime(year, 6, 1, tzinfo=UTC),
    )


def _papers(n: int) -> list[ArxivPaper]:
    return [_arxiv_paper(i) for i in range(n)]


def _synthesis(arxiv_id: str, *, confidence: float = 0.85) -> PaperSynthesis:
    """Build a minimal PaperSynthesis for the given arxiv_id."""
    return PaperSynthesis(
        arxiv_id=arxiv_id,
        summary_about=f"What {arxiv_id} is about.",
        summary_relation_to_topic=f"How {arxiv_id} relates to the topic.",
        summary_problem=f"The problem {arxiv_id} addresses.",
        summary_approach=f"The approach {arxiv_id} takes.",
        summary_impact=f"The impact of {arxiv_id}.",
        confidence=confidence,
    )


def _syntheses(papers: list[ArxivPaper]) -> list[PaperSynthesis]:
    return [_synthesis(p.arxiv_id) for p in papers]


def _narrative(seed: str = "era") -> str:
    """A valid narrative (>200 chars) that the schema accepts."""
    return (
        f"This era of the {seed} field marks a methodological shift. "
        "Researchers moved from one set of assumptions to a substantively "
        "different framing, producing a body of work that downstream "
        "papers treat as the canonical starting point. The shift is "
        "visible across multiple sub-areas of the topic and reshaped the "
        "questions practitioners considered worth asking."
    )


def _era_dict(
    *,
    era_id: str,
    paper_ids: list[str],
    start: int = 2017,
    end: int | None = 2020,
    name: str | None = None,
    narrative: str | None = None,
) -> dict[str, Any]:
    """Build a raw era dict in the model-emitted shape."""
    out: dict[str, Any] = {
        "era_id": era_id,
        "name": name or era_id.replace("-", " ").title(),
        "date_range_start_year": start,
        "narrative": narrative or _narrative(era_id),
        "paper_ids": paper_ids,
    }
    if end is not None:
        out["date_range_end_year"] = end
    return out


def _tool_use_block(eras: list[dict[str, Any]], *, name: str = "submit_era_partition") -> MagicMock:
    block = MagicMock()
    block.type = "tool_use"
    block.name = name
    block.id = "tu_1"
    block.input = {"eras": eras}
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
    input_tokens: int = 3000,
    output_tokens: int = 800,
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


# ───── Class-level wiring ───────────────────────────────────────────────


def test_classvars_default_to_haiku_and_v1_prompt() -> None:
    assert EraPartitionAgent.DEFAULT_MODEL == "claude-haiku-4-5"
    assert EraPartitionAgent.PROMPT_NAME == "arch_01_era_partition"
    assert EraPartitionAgent.PROMPT_VERSION == "v1"
    assert EraPartitionAgent.TOOL_NAME == "submit_era_partition"


def test_constructor_does_not_load_prompt() -> None:
    agent = EraPartitionAgent(_fake_client(_fake_message([])))
    assert agent._system_prompt is None


def test_bounds_constants() -> None:
    """ERAS_MIN=2 and ERAS_MAX=4 per user direction."""
    assert ERAS_MIN == 2
    assert ERAS_MAX == 4


# ───── Happy path ───────────────────────────────────────────────────────


async def test_partition_happy_path_two_eras_over_six_papers() -> None:
    """6 papers split into 2 balanced eras; all paper_ids covered."""
    papers = _papers(6)
    eras = [
        _era_dict(
            era_id="foundations", paper_ids=[p.arxiv_id for p in papers[:3]], start=2014, end=2018
        ),
        _era_dict(
            era_id="extensions", paper_ids=[p.arxiv_id for p in papers[3:]], start=2017, end=2022
        ),
    ]
    response = _fake_message([_tool_use_block(eras)])
    agent = EraPartitionAgent(_fake_client(response))

    result = await agent.partition(
        topic="positional encodings", papers=papers, syntheses=_syntheses(papers)
    )

    assert len(result.eras) == 2
    # papers_by_era convenience map populated.
    assert set(result.papers_by_era.keys()) == {"foundations", "extensions"}
    assert sum(len(v) for v in result.papers_by_era.values()) == 6


async def test_partition_allows_overlapping_year_ranges() -> None:
    """Eras with overlapping year ranges are accepted (content-driven)."""
    papers = _papers(4)
    eras = [
        _era_dict(
            era_id="a", paper_ids=[papers[0].arxiv_id, papers[1].arxiv_id], start=2014, end=2020
        ),
        _era_dict(
            era_id="b", paper_ids=[papers[2].arxiv_id, papers[3].arxiv_id], start=2018, end=2022
        ),
    ]
    response = _fake_message([_tool_use_block(eras)])
    agent = EraPartitionAgent(_fake_client(response))

    result = await agent.partition(topic="t", papers=papers, syntheses=_syntheses(papers))
    assert len(result.eras) == 2


async def test_partition_allows_single_paper_eras() -> None:
    """One-paper eras are valid (Field(paper_ids, min_length=1))."""
    papers = _papers(4)
    eras = [
        _era_dict(era_id="solo", paper_ids=[papers[0].arxiv_id]),
        _era_dict(era_id="trio", paper_ids=[p.arxiv_id for p in papers[1:]]),
    ]
    response = _fake_message([_tool_use_block(eras)])
    agent = EraPartitionAgent(_fake_client(response))

    result = await agent.partition(topic="t", papers=papers, syntheses=_syntheses(papers))
    assert result.papers_by_era["solo"] == (papers[0].arxiv_id,)


async def test_partition_allows_open_ended_era() -> None:
    """date_range_end_year=None ('ongoing') is accepted."""
    papers = _papers(4)
    eras = [
        _era_dict(era_id="past", paper_ids=[p.arxiv_id for p in papers[:2]], start=2014, end=2020),
        _era_dict(
            era_id="frontier", paper_ids=[p.arxiv_id for p in papers[2:]], start=2021, end=None
        ),
    ]
    response = _fake_message([_tool_use_block(eras)])
    agent = EraPartitionAgent(_fake_client(response))

    result = await agent.partition(topic="t", papers=papers, syntheses=_syntheses(papers))
    frontier = next(e for e in result.eras if e.era_id == "frontier")
    assert frontier.date_range_end_year is None


# ───── Token + cost ─────────────────────────────────────────────────────


async def test_partition_records_token_usage_and_cost() -> None:
    """Tokens come from response.usage; cost uses Haiku rates."""
    papers = _papers(4)
    eras = [
        _era_dict(era_id="a", paper_ids=[p.arxiv_id for p in papers[:2]]),
        _era_dict(era_id="b", paper_ids=[p.arxiv_id for p in papers[2:]]),
    ]
    response = _fake_message([_tool_use_block(eras)], input_tokens=4000, output_tokens=900)
    agent = EraPartitionAgent(_fake_client(response))

    result = await agent.partition(topic="t", papers=papers, syntheses=_syntheses(papers))

    assert result.usage.input_tokens == 4000
    assert result.usage.output_tokens == 900
    expected = 4000 / 1e6 * 1.0 + 900 / 1e6 * 5.0  # Haiku
    assert result.usage.cost_usd == pytest.approx(expected)


# ───── SDK wiring ───────────────────────────────────────────────────────


async def test_partition_forces_tool_use_and_passes_haiku() -> None:
    papers = _papers(4)
    eras = [
        _era_dict(era_id="a", paper_ids=[p.arxiv_id for p in papers[:2]]),
        _era_dict(era_id="b", paper_ids=[p.arxiv_id for p in papers[2:]]),
    ]
    client = _fake_client(_fake_message([_tool_use_block(eras)]))
    agent = EraPartitionAgent(client)

    await agent.partition(topic="t", papers=papers, syntheses=_syntheses(papers))

    kwargs = client.messages.create.await_args.kwargs
    assert kwargs["model"] == "claude-haiku-4-5"
    assert kwargs["tool_choice"] == {"type": "tool", "name": "submit_era_partition"}
    assert len(kwargs["tools"]) == 1
    assert kwargs["tools"][0]["input_schema"] == EraPartition.model_json_schema()


async def test_partition_user_message_carries_topic_and_full_synthesis() -> None:
    """User message has topic + each paper's 5 synthesis fields (not abstract)."""
    papers = _papers(4)
    eras = [
        _era_dict(era_id="a", paper_ids=[p.arxiv_id for p in papers[:2]]),
        _era_dict(era_id="b", paper_ids=[p.arxiv_id for p in papers[2:]]),
    ]
    client = _fake_client(_fake_message([_tool_use_block(eras)]))
    agent = EraPartitionAgent(client)

    await agent.partition(topic="positional encodings", papers=papers, syntheses=_syntheses(papers))

    body = client.messages.create.await_args.kwargs["messages"][0]["content"]
    assert "positional encodings" in body
    for p in papers:
        assert f"arxiv_id={p.arxiv_id}" in body
        # Synthesis fields present (not the abstract).
        assert f"about: What {p.arxiv_id} is about." in body
        assert f"impact: The impact of {p.arxiv_id}." in body
    # The abstract text from stage 1 should NOT appear here.
    for p in papers:
        assert p.abstract not in body


# ───── Refusal ──────────────────────────────────────────────────────────


async def test_partition_raises_refusal_on_no_tool_use_block() -> None:
    papers = _papers(4)
    response = _fake_message([_text_block("I cannot complete this task.")], stop_reason="refusal")
    agent = EraPartitionAgent(_fake_client(response))

    with pytest.raises(EraPartitionRefusedError) as exc_info:
        await agent.partition(topic="t", papers=papers, syntheses=_syntheses(papers))
    assert exc_info.value.stop_reason == "refusal"
    assert "I cannot complete" in exc_info.value.content_summary


async def test_partition_raises_refusal_on_wrong_tool_name() -> None:
    papers = _papers(4)
    eras = [
        _era_dict(era_id="a", paper_ids=[p.arxiv_id for p in papers[:2]]),
        _era_dict(era_id="b", paper_ids=[p.arxiv_id for p in papers[2:]]),
    ]
    response = _fake_message([_tool_use_block(eras, name="some_other_tool")])
    agent = EraPartitionAgent(_fake_client(response))

    with pytest.raises(EraPartitionRefusedError):
        await agent.partition(topic="t", papers=papers, syntheses=_syntheses(papers))


# ───── Schema-level (pydantic) invalid output ───────────────────────────


async def test_partition_raises_when_below_two_eras() -> None:
    """Schema floor of 2 eras."""
    papers = _papers(4)
    one_era = [_era_dict(era_id="only", paper_ids=[p.arxiv_id for p in papers])]
    response = _fake_message([_tool_use_block(one_era)])
    agent = EraPartitionAgent(_fake_client(response))

    with pytest.raises(EraPartitionInvalidOutputError, match="schema validation"):
        await agent.partition(topic="t", papers=papers, syntheses=_syntheses(papers))


async def test_partition_raises_when_above_four_eras() -> None:
    """Schema ceiling of 4 eras."""
    papers = _papers(5)
    five_eras = [_era_dict(era_id=f"era{i}", paper_ids=[papers[i].arxiv_id]) for i in range(5)]
    response = _fake_message([_tool_use_block(five_eras)])
    agent = EraPartitionAgent(_fake_client(response))

    with pytest.raises(EraPartitionInvalidOutputError, match="schema validation"):
        await agent.partition(topic="t", papers=papers, syntheses=_syntheses(papers))


async def test_partition_raises_on_invalid_era_id_slug() -> None:
    """era_id violates the lowercase slug regex (uppercase rejected)."""
    papers = _papers(4)
    bad_eras = [
        _era_dict(era_id="Foundations", paper_ids=[p.arxiv_id for p in papers[:2]]),
        _era_dict(era_id="b", paper_ids=[p.arxiv_id for p in papers[2:]]),
    ]
    response = _fake_message([_tool_use_block(bad_eras)])
    agent = EraPartitionAgent(_fake_client(response))

    with pytest.raises(EraPartitionInvalidOutputError, match="schema validation"):
        await agent.partition(topic="t", papers=papers, syntheses=_syntheses(papers))


async def test_partition_raises_on_too_short_narrative() -> None:
    """Narratives under the 200-char floor fail the pydantic check."""
    papers = _papers(4)
    bad_eras = [
        _era_dict(
            era_id="a",
            paper_ids=[p.arxiv_id for p in papers[:2]],
            narrative="Too short.",
        ),
        _era_dict(era_id="b", paper_ids=[p.arxiv_id for p in papers[2:]]),
    ]
    response = _fake_message([_tool_use_block(bad_eras)])
    agent = EraPartitionAgent(_fake_client(response))

    with pytest.raises(EraPartitionInvalidOutputError, match="schema validation"):
        await agent.partition(topic="t", papers=papers, syntheses=_syntheses(papers))


# ───── Cross-field invariants ──────────────────────────────────────────


async def test_partition_raises_on_fabricated_paper_id() -> None:
    papers = _papers(4)
    bad_eras = [
        _era_dict(era_id="a", paper_ids=[papers[0].arxiv_id, papers[1].arxiv_id]),
        _era_dict(era_id="b", paper_ids=[papers[2].arxiv_id, papers[3].arxiv_id, "9999.99999"]),
    ]
    response = _fake_message([_tool_use_block(bad_eras)])
    agent = EraPartitionAgent(_fake_client(response))

    with pytest.raises(EraPartitionInvalidOutputError, match="not in the input"):
        await agent.partition(topic="t", papers=papers, syntheses=_syntheses(papers))


async def test_partition_raises_when_paper_assigned_to_multiple_eras() -> None:
    """Option A: paper in exactly one era. Duplicates across eras rejected."""
    papers = _papers(4)
    bad_eras = [
        _era_dict(era_id="a", paper_ids=[papers[0].arxiv_id, papers[1].arxiv_id]),
        _era_dict(
            era_id="b", paper_ids=[papers[1].arxiv_id, papers[2].arxiv_id, papers[3].arxiv_id]
        ),
    ]
    response = _fake_message([_tool_use_block(bad_eras)])
    agent = EraPartitionAgent(_fake_client(response))

    with pytest.raises(EraPartitionInvalidOutputError, match="multiple eras"):
        await agent.partition(topic="t", papers=papers, syntheses=_syntheses(papers))


async def test_partition_raises_on_duplicate_era_id() -> None:
    papers = _papers(4)
    bad_eras = [
        _era_dict(era_id="same", paper_ids=[papers[0].arxiv_id, papers[1].arxiv_id]),
        _era_dict(era_id="same", paper_ids=[papers[2].arxiv_id, papers[3].arxiv_id]),
    ]
    response = _fake_message([_tool_use_block(bad_eras)])
    agent = EraPartitionAgent(_fake_client(response))

    with pytest.raises(EraPartitionInvalidOutputError, match="Duplicate era_id"):
        await agent.partition(topic="t", papers=papers, syntheses=_syntheses(papers))


async def test_partition_raises_when_paper_missing_from_all_eras() -> None:
    """Every input paper must be assigned somewhere."""
    papers = _papers(4)
    # Omit papers[3]
    bad_eras = [
        _era_dict(era_id="a", paper_ids=[papers[0].arxiv_id, papers[1].arxiv_id]),
        _era_dict(era_id="b", paper_ids=[papers[2].arxiv_id]),
    ]
    response = _fake_message([_tool_use_block(bad_eras)])
    agent = EraPartitionAgent(_fake_client(response))

    with pytest.raises(EraPartitionInvalidOutputError, match="not assigned to any era"):
        await agent.partition(topic="t", papers=papers, syntheses=_syntheses(papers))


async def test_partition_raises_when_end_year_before_start_year() -> None:
    """date_range_end_year < date_range_start_year is invalid."""
    papers = _papers(4)
    bad_eras = [
        _era_dict(
            era_id="a",
            paper_ids=[p.arxiv_id for p in papers[:2]],
            start=2020,
            end=2018,  # end before start
        ),
        _era_dict(era_id="b", paper_ids=[p.arxiv_id for p in papers[2:]]),
    ]
    response = _fake_message([_tool_use_block(bad_eras)])
    agent = EraPartitionAgent(_fake_client(response))

    with pytest.raises(EraPartitionInvalidOutputError, match="before"):
        await agent.partition(topic="t", papers=papers, syntheses=_syntheses(papers))


# ───── Defensive: input mismatch ────────────────────────────────────────


async def test_partition_raises_immediately_on_empty_papers() -> None:
    """Empty papers list is a stage-3 bug; surface without an API call."""
    client = _fake_client(_fake_message([]))
    agent = EraPartitionAgent(client)

    with pytest.raises(EraPartitionInvalidOutputError, match="no papers"):
        await agent.partition(topic="t", papers=[], syntheses=[])

    client.messages.create.assert_not_called()


async def test_partition_raises_on_synthesis_mismatch() -> None:
    """A paper without a matching synthesis triggers an immediate error."""
    papers = _papers(4)
    # Synthesis for only 3 of the 4 papers.
    bad_syntheses = _syntheses(papers[:3])
    client = _fake_client(_fake_message([]))
    agent = EraPartitionAgent(client)

    with pytest.raises(EraPartitionInvalidOutputError, match="without synthesis"):
        await agent.partition(topic="t", papers=papers, syntheses=bad_syntheses)
    client.messages.create.assert_not_called()


# ───── EraEntry schema standalone ───────────────────────────────────────


def test_era_entry_rejects_uppercase_era_id() -> None:
    """era_id must match the lowercase slug regex."""
    with pytest.raises(ValidationError):
        EraEntry(
            era_id="Foundations",
            name="Foundations",
            date_range_start_year=2017,
            date_range_end_year=2020,
            narrative=_narrative(),
            paper_ids=["1706.00001"],
        )


def test_era_entry_accepts_none_end_year() -> None:
    """date_range_end_year is Optional/nullable."""
    e = EraEntry(
        era_id="frontier",
        name="Frontier",
        date_range_start_year=2022,
        date_range_end_year=None,
        narrative=_narrative(),
        paper_ids=["1706.00001"],
    )
    assert e.date_range_end_year is None
