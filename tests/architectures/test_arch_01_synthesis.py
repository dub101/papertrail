"""Tests for ``arch_01`` stage 3: ``SynthesisAgent``.

Three mocking concerns make this suite slightly more involved than triage:
    - The agent makes one ``client.messages.create`` call per batch and
      runs them concurrently via ``asyncio.gather``. ``AsyncMock`` with
      ``side_effect=[r1, r2, ...]`` consumes responses in order.
    - The retry round (round 2) only fires when round 1 left papers
      missing. We construct scenarios that force or suppress retry by
      shaping round-1 responses.
    - The cross-field invariants (fabricated id, duplicate id) propagate
      via ``_safe_batch`` into ``_BatchOutcome.error`` — the test asserts
      on the resulting ``ErrorRecord`` plus the retry behaviour.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import HttpUrl, ValidationError

from papertrail.architectures.arch_01_sequential.synthesis import (
    TARGET_BATCH_SIZE,
    PaperSynthesis,
    SynthesisAgent,
    SynthesisError,
    _partition,
    _split_balanced,
)
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


def _synthesis_dict(
    arxiv_id: str, *, notes: str | None = None, confidence: float = 0.85
) -> dict[str, Any]:
    """Build a valid PaperSynthesis dict (the model-emitted shape)."""
    out: dict[str, Any] = {
        "arxiv_id": arxiv_id,
        "summary_about": f"What {arxiv_id} is about.",
        "summary_relation_to_topic": f"How {arxiv_id} relates to the topic.",
        "summary_problem": f"The problem {arxiv_id} addresses.",
        "summary_approach": f"The approach {arxiv_id} takes.",
        "summary_impact": f"The impact of {arxiv_id}.",
        "confidence": confidence,
    }
    if notes is not None:
        out["notes"] = notes
    return out


def _tool_use_block(
    syntheses: list[dict[str, Any]],
    *,
    name: str = "submit_synthesis",
    block_id: str = "tu_1",
) -> MagicMock:
    block = MagicMock()
    block.type = "tool_use"
    block.name = name
    block.id = block_id
    block.input = {"syntheses": syntheses}
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
    input_tokens: int = 1000,
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


def _fake_client_seq(responses: list[MagicMock | Exception]) -> MagicMock:
    """Client whose ``messages.create`` returns or raises in order."""
    client = MagicMock()
    client.messages = MagicMock()
    client.messages.create = AsyncMock(side_effect=responses)
    return client


# ───── Unit: batch sizing ───────────────────────────────────────────────


def test_split_balanced_handles_zero_and_small() -> None:
    """0 -> empty, 1..5 -> single-batch."""
    assert _split_balanced(0) == []
    assert _split_balanced(1) == [1]
    assert _split_balanced(4) == [4]
    assert _split_balanced(5) == [5]


def test_split_balanced_balances_at_target() -> None:
    """6 -> (3,3), not (5,1); 7 -> (4,3); 10 -> (5,5)."""
    assert _split_balanced(6) == [3, 3]
    assert _split_balanced(7) == [4, 3]
    assert _split_balanced(8) == [4, 4]
    assert _split_balanced(9) == [5, 4]
    assert _split_balanced(10) == [5, 5]


def test_split_balanced_uses_three_batches_when_over_ten() -> None:
    """11 and 12 produce 3 balanced batches each (v2 algorithm)."""
    assert _split_balanced(11) == [4, 4, 3]
    assert _split_balanced(12) == [4, 4, 4]


def test_partition_slices_papers_in_order() -> None:
    """_partition preserves input order across batch boundaries."""
    papers = _papers(7)  # batches (4, 3)
    batches = _partition(papers)
    assert [len(b) for b in batches] == [4, 3]
    assert [p.arxiv_id for p in batches[0]] == [p.arxiv_id for p in papers[:4]]
    assert [p.arxiv_id for p in batches[1]] == [p.arxiv_id for p in papers[4:]]


# ───── Class-level wiring ───────────────────────────────────────────────


def test_classvars_default_to_haiku_and_v1_prompt() -> None:
    assert SynthesisAgent.DEFAULT_MODEL == "claude-haiku-4-5"
    assert SynthesisAgent.PROMPT_NAME == "arch_01_synthesis"
    assert SynthesisAgent.PROMPT_VERSION == "v1"
    assert SynthesisAgent.TOOL_NAME == "submit_synthesis"
    assert TARGET_BATCH_SIZE == 5


def test_constructor_does_not_load_prompt() -> None:
    """Prompt lazy-loaded so construction touches no disk."""
    agent = SynthesisAgent(_fake_client_seq([_fake_message([])]))
    assert agent._system_prompt is None


# ───── Happy path: round 1 covers everything ────────────────────────────


async def test_synthesize_happy_path_8_papers_in_two_batches() -> None:
    """8 papers -> 2 batches of (4, 4); both succeed; no retry."""
    papers = _papers(8)
    batch1_ids = [p.arxiv_id for p in papers[:4]]
    batch2_ids = [p.arxiv_id for p in papers[4:]]
    responses = [
        _fake_message(
            [_tool_use_block([_synthesis_dict(aid) for aid in batch1_ids])]
        ),
        _fake_message(
            [_tool_use_block([_synthesis_dict(aid) for aid in batch2_ids])]
        ),
    ]
    client = _fake_client_seq(responses)
    agent = SynthesisAgent(client)

    result = await agent.synthesize(topic="t", papers=papers)

    assert len(result.syntheses) == 8
    assert [s.arxiv_id for s in result.syntheses] == [p.arxiv_id for p in papers]
    assert result.error_records == ()
    assert result.notes == ()
    # Only 2 API calls — no retry round.
    assert client.messages.create.await_count == 2


async def test_synthesize_preserves_input_order_when_batches_return_unordered() -> None:
    """Within a batch, the model may emit decisions in any order; we sort by input."""
    papers = _papers(4)  # one batch of 4
    # Model returns them in reverse:
    reversed_ids = [papers[3].arxiv_id, papers[2].arxiv_id, papers[1].arxiv_id, papers[0].arxiv_id]
    response = _fake_message(
        [_tool_use_block([_synthesis_dict(aid) for aid in reversed_ids])]
    )
    agent = SynthesisAgent(_fake_client_seq([response]))

    result = await agent.synthesize(topic="t", papers=papers)

    # Output order matches papers input order, not model-emit order.
    assert [s.arxiv_id for s in result.syntheses] == [p.arxiv_id for p in papers]


# ───── Notes routing ────────────────────────────────────────────────────


async def test_synthesize_collects_non_null_notes_into_telemetry_channel() -> None:
    """Only papers with non-null notes appear in result.notes."""
    papers = _papers(4)
    syntheses = [
        _synthesis_dict(papers[0].arxiv_id, notes="something odd about this paper"),
        _synthesis_dict(papers[1].arxiv_id),  # no notes
        _synthesis_dict(papers[2].arxiv_id, notes="another flag"),
        _synthesis_dict(papers[3].arxiv_id),  # no notes
    ]
    response = _fake_message([_tool_use_block(syntheses)])
    agent = SynthesisAgent(_fake_client_seq([response]))

    result = await agent.synthesize(topic="t", papers=papers)

    assert len(result.notes) == 2
    note_ids = {n.arxiv_id for n in result.notes}
    assert note_ids == {papers[0].arxiv_id, papers[2].arxiv_id}


# ───── Per-paper confidence ─────────────────────────────────────────────


async def test_synthesize_propagates_model_emitted_confidence_per_paper() -> None:
    """Confidence values emitted by the model propagate into PaperSynthesis."""
    papers = _papers(4)
    syntheses = [
        _synthesis_dict(papers[0].arxiv_id, confidence=0.95),
        _synthesis_dict(papers[1].arxiv_id, confidence=0.60),
        _synthesis_dict(papers[2].arxiv_id, confidence=0.40),
        _synthesis_dict(papers[3].arxiv_id, confidence=0.85),
    ]
    response = _fake_message([_tool_use_block(syntheses)])
    agent = SynthesisAgent(_fake_client_seq([response]))

    result = await agent.synthesize(topic="t", papers=papers)

    confidence_by_id = {s.arxiv_id: s.confidence for s in result.syntheses}
    assert confidence_by_id[papers[0].arxiv_id] == 0.95
    assert confidence_by_id[papers[1].arxiv_id] == 0.60
    assert confidence_by_id[papers[2].arxiv_id] == 0.40
    assert confidence_by_id[papers[3].arxiv_id] == 0.85


async def test_synthesize_disclaimer_synthesis_has_zero_confidence() -> None:
    """A paper that fails both rounds gets a disclaimer with confidence=0.0."""
    papers = _papers(4)
    missing_id = papers[2].arxiv_id
    other_ids = [p.arxiv_id for p in papers if p.arxiv_id != missing_id]
    responses: list[MagicMock | Exception] = [
        # Round 1: covers 3 of 4
        _fake_message(
            [_tool_use_block([_synthesis_dict(aid) for aid in other_ids])]
        ),
        # Round 2: retry also fails
        ConnectionError("retry also failed"),
    ]
    agent = SynthesisAgent(_fake_client_seq(responses))

    result = await agent.synthesize(topic="t", papers=papers)

    # The disclaimer-filled paper has confidence=0.0 (code-emitted floor).
    confidence_by_id = {s.arxiv_id: s.confidence for s in result.syntheses}
    assert confidence_by_id[missing_id] == 0.0
    # The other three carry the model-emitted 0.85 from _synthesis_dict default.
    for other in other_ids:
        assert confidence_by_id[other] == 0.85


def test_paper_synthesis_rejects_confidence_out_of_range() -> None:
    """confidence must be in [0.0, 1.0]."""
    base = _synthesis_dict("1706.00001")
    for bad in (-0.1, 1.1, 1.5, -10.0):
        broken = base.copy()
        broken["confidence"] = bad
        with pytest.raises(ValidationError):
            PaperSynthesis.model_validate(broken)


def test_paper_synthesis_requires_confidence_field() -> None:
    """confidence is not optional."""
    bad = {
        k: v
        for k, v in _synthesis_dict("1706.00001").items()
        if k != "confidence"
    }
    with pytest.raises(ValidationError):
        PaperSynthesis.model_validate(bad)


# ───── Token + cost accounting ──────────────────────────────────────────


async def test_synthesize_sums_tokens_across_all_batch_calls() -> None:
    """input/output tokens accumulate across both batches."""
    papers = _papers(8)
    responses = [
        _fake_message(
            [_tool_use_block([_synthesis_dict(p.arxiv_id) for p in papers[:4]])],
            input_tokens=2000,
            output_tokens=600,
        ),
        _fake_message(
            [_tool_use_block([_synthesis_dict(p.arxiv_id) for p in papers[4:]])],
            input_tokens=2500,
            output_tokens=700,
        ),
    ]
    agent = SynthesisAgent(_fake_client_seq(responses))
    result = await agent.synthesize(topic="t", papers=papers)

    assert result.usage.input_tokens == 4500
    assert result.usage.output_tokens == 1300
    expected = 4500 / 1e6 * 1.0 + 1300 / 1e6 * 5.0  # Haiku rates
    assert result.usage.cost_usd == pytest.approx(expected)


# ───── Retry round: batch-level failure recovered ───────────────────────


async def test_synthesize_retries_when_a_batch_raises() -> None:
    """Round 1 batch raises; round 2 covers the missing papers."""
    papers = _papers(8)
    batch1_ids = [p.arxiv_id for p in papers[:4]]
    batch2_ids = [p.arxiv_id for p in papers[4:]]
    responses: list[MagicMock | Exception] = [
        # Round 1: batch 1 succeeds.
        _fake_message(
            [_tool_use_block([_synthesis_dict(aid) for aid in batch1_ids])]
        ),
        # Round 1: batch 2 raises (e.g. network).
        ConnectionError("network blip"),
        # Round 2: retry of batch 2 succeeds.
        _fake_message(
            [_tool_use_block([_synthesis_dict(aid) for aid in batch2_ids])]
        ),
    ]
    client = _fake_client_seq(responses)
    agent = SynthesisAgent(client)

    result = await agent.synthesize(topic="t", papers=papers)

    # All 8 papers recovered.
    assert len(result.syntheses) == 8
    assert [s.arxiv_id for s in result.syntheses] == [p.arxiv_id for p in papers]
    # One ErrorRecord captured the round-1 failure; retry succeeded so no
    # per-paper "still missing" records.
    assert len(result.error_records) == 1
    assert result.error_records[0].step_index == 3
    assert result.error_records[0].recovered is True
    assert "ConnectionError" in result.error_records[0].message


# ───── Retry round: missing papers within a successful batch ────────────


async def test_synthesize_retries_when_a_batch_skips_papers() -> None:
    """Batch returns successfully but omits one paper -> retry covers it."""
    papers = _papers(4)  # one batch of 4
    missing_id = papers[2].arxiv_id
    responses = [
        # Round 1: model emits 3 of 4 (omits papers[2]).
        _fake_message(
            [_tool_use_block(
                [_synthesis_dict(p.arxiv_id) for p in papers if p.arxiv_id != missing_id]
            )]
        ),
        # Round 2: retry of just papers[2] succeeds.
        _fake_message([_tool_use_block([_synthesis_dict(missing_id)])]),
    ]
    agent = SynthesisAgent(_fake_client_seq(responses))

    result = await agent.synthesize(topic="t", papers=papers)

    assert len(result.syntheses) == 4
    # Order preserved; missing paper's synthesis is the retry one.
    assert result.syntheses[2].arxiv_id == missing_id
    # No error records — both rounds succeeded eventually.
    assert result.error_records == ()


# ───── Permanently missing: disclaimer fill ─────────────────────────────


async def test_synthesize_disclaimer_fills_papers_missing_after_retry() -> None:
    """Both rounds fail to cover papers[2] -> disclaimer synthesis + ErrorRecord."""
    papers = _papers(4)
    missing_id = papers[2].arxiv_id
    other_ids = [p.arxiv_id for p in papers if p.arxiv_id != missing_id]
    responses: list[MagicMock | Exception] = [
        # Round 1: covers 3 of 4.
        _fake_message(
            [_tool_use_block([_synthesis_dict(aid) for aid in other_ids])]
        ),
        # Round 2: also fails to return the missing one (raises).
        ConnectionError("retry also failed"),
    ]
    agent = SynthesisAgent(_fake_client_seq(responses))

    result = await agent.synthesize(topic="t", papers=papers)

    # Output still has 4 entries — three real + one disclaimer.
    assert len(result.syntheses) == 4
    missing_synth = result.syntheses[2]
    assert missing_synth.arxiv_id == missing_id
    assert "Stage 3 synthesis failed" in missing_synth.summary_about
    # Distinct disclaimer per dimension (Evaluator attribution).
    assert missing_synth.summary_about != missing_synth.summary_problem
    # Two ErrorRecords: one for the round-2 batch failure, one per-paper.
    assert len(result.error_records) == 2
    per_paper_record = next(
        r for r in result.error_records if missing_id in r.message
    )
    assert per_paper_record.recovered is True
    assert per_paper_record.step_index == 3


# ───── Total failure: SynthesisError raised ─────────────────────────────


async def test_synthesize_raises_when_every_batch_fails_both_rounds() -> None:
    """No synthesised papers across both rounds -> SynthesisError."""
    papers = _papers(4)
    responses: list[MagicMock | Exception] = [
        ConnectionError("round 1 fail"),
        ConnectionError("round 2 fail"),
    ]
    agent = SynthesisAgent(_fake_client_seq(responses))

    with pytest.raises(SynthesisError, match="synthesised 0 of 4"):
        await agent.synthesize(topic="t", papers=papers)


# ───── Fabricated / duplicate ids fail the batch ────────────────────────


async def test_synthesize_treats_fabricated_arxiv_id_as_batch_failure() -> None:
    """A batch emitting an unknown arxiv_id is a batch-level failure -> retry."""
    papers = _papers(4)
    fake_id = "9999.99999"
    responses = [
        # Round 1: includes a fabricated id (and omits one real one).
        _fake_message(
            [_tool_use_block(
                [_synthesis_dict(papers[0].arxiv_id), _synthesis_dict(fake_id),
                 _synthesis_dict(papers[2].arxiv_id), _synthesis_dict(papers[3].arxiv_id)]
            )]
        ),
        # Round 2: retries the missing one.
        _fake_message(
            [_tool_use_block([_synthesis_dict(papers[1].arxiv_id)])]
        ),
    ]
    agent = SynthesisAgent(_fake_client_seq(responses))

    # Round 1's fabricated id makes the whole batch fail; only round 2's
    # retry contributes successfully. End result: 1 real + 3 disclaimers.
    result = await agent.synthesize(topic="t", papers=papers)

    # The full batch failed in round 1 (so papers 0, 2, 3 are missing).
    # Round 2 retries the missing ids — but round 2 only retries the
    # paper we marked missing after round 1 (papers[1]). Papers 0, 2, 3
    # are missing because the round-1 batch failed entirely.
    # So round 2 *should* retry all 4 (papers 0, 1, 2, 3) — actually no,
    # only papers[1] was missing after round 1 conceptually...
    # Re-reading the code: when a batch fails, none of its papers are
    # marked as synthesised. So round 2 retries papers 0, 1, 2, 3.
    # Round 2's _fake_message only provides papers[1] -> 3 still missing
    # after retry.
    assert result.syntheses[1].arxiv_id == papers[1].arxiv_id
    # Papers 0, 2, 3 get disclaimers.
    for idx in (0, 2, 3):
        assert "Stage 3 synthesis failed" in result.syntheses[idx].summary_about


async def test_synthesize_treats_duplicate_arxiv_id_as_batch_failure() -> None:
    """A batch emitting a duplicate arxiv_id is a batch-level failure."""
    papers = _papers(4)
    responses = [
        # Round 1: duplicate of papers[0].
        _fake_message(
            [_tool_use_block(
                [
                    _synthesis_dict(papers[0].arxiv_id),
                    _synthesis_dict(papers[0].arxiv_id),  # duplicate
                    _synthesis_dict(papers[2].arxiv_id),
                    _synthesis_dict(papers[3].arxiv_id),
                ]
            )]
        ),
        # Round 2: retry covers all 4 (assume it succeeds).
        _fake_message(
            [_tool_use_block([_synthesis_dict(p.arxiv_id) for p in papers])]
        ),
    ]
    agent = SynthesisAgent(_fake_client_seq(responses))

    result = await agent.synthesize(topic="t", papers=papers)

    # Round 2 covered all 4 cleanly.
    assert len(result.syntheses) == 4
    for s in result.syntheses:
        assert "Stage 3 synthesis failed" not in s.summary_about


# ───── Refusal stop_reason ──────────────────────────────────────────────


async def test_synthesize_treats_refusal_as_batch_failure() -> None:
    """A batch with no tool_use block is treated like any other batch failure."""
    papers = _papers(4)
    responses = [
        # Round 1: model emits only text, no tool_use.
        _fake_message([_text_block("I cannot complete this.")], stop_reason="refusal"),
        # Round 2: succeeds.
        _fake_message(
            [_tool_use_block([_synthesis_dict(p.arxiv_id) for p in papers])]
        ),
    ]
    agent = SynthesisAgent(_fake_client_seq(responses))

    result = await agent.synthesize(topic="t", papers=papers)
    assert len(result.syntheses) == 4
    # One ErrorRecord for the round-1 refusal.
    assert any("refusal" in r.message.lower() or "no tool_use" in r.message for r in result.error_records)


# ───── SDK wiring ───────────────────────────────────────────────────────


async def test_synthesize_passes_haiku_and_forces_tool_choice() -> None:
    """First batch call goes to Haiku with forced tool_choice."""
    papers = _papers(4)
    response = _fake_message(
        [_tool_use_block([_synthesis_dict(p.arxiv_id) for p in papers])]
    )
    client = _fake_client_seq([response])
    agent = SynthesisAgent(client)

    await agent.synthesize(topic="t", papers=papers)

    kwargs = client.messages.create.await_args.kwargs
    assert kwargs["model"] == "claude-haiku-4-5"
    assert kwargs["tool_choice"] == {"type": "tool", "name": "submit_synthesis"}
    assert len(kwargs["tools"]) == 1


async def test_synthesize_user_message_carries_full_abstracts() -> None:
    """Each batch's user message includes the full abstract for every paper in it."""
    papers = _papers(4)
    response = _fake_message(
        [_tool_use_block([_synthesis_dict(p.arxiv_id) for p in papers])]
    )
    client = _fake_client_seq([response])
    agent = SynthesisAgent(client)

    await agent.synthesize(topic="the topic", papers=papers)

    kwargs = client.messages.create.await_args.kwargs
    body = kwargs["messages"][0]["content"]
    assert "the topic" in body
    for p in papers:
        assert f"arxiv_id={p.arxiv_id}" in body
        assert p.abstract in body


# ───── Defensive: empty input ───────────────────────────────────────────


async def test_synthesize_raises_immediately_on_empty_papers() -> None:
    """Empty papers list is a stage-2 bug; surface without an API call."""
    client = _fake_client_seq([])
    agent = SynthesisAgent(client)

    with pytest.raises(SynthesisError, match="no papers"):
        await agent.synthesize(topic="t", papers=[])

    client.messages.create.assert_not_called()


# ───── PaperSynthesis schema ────────────────────────────────────────────


def test_paper_synthesis_rejects_missing_required_field() -> None:
    """Each of the 5 summary fields is required at min_length=1."""
    base = _synthesis_dict("1706.00001")
    for field in (
        "summary_about",
        "summary_relation_to_topic",
        "summary_problem",
        "summary_approach",
        "summary_impact",
    ):
        broken = base.copy()
        broken[field] = ""
        with pytest.raises(ValidationError):
            PaperSynthesis.model_validate(broken)


def test_paper_synthesis_accepts_omitted_notes() -> None:
    """``notes`` is optional and defaults to None."""
    p = PaperSynthesis.model_validate(_synthesis_dict("1706.00001"))
    assert p.notes is None
