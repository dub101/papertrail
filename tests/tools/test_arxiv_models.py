"""Round-trip + validation tests for ``ArxivPaper``.

We cover the four behaviours that justify having a pydantic model at all
instead of a plain ``dict``:

1. Valid construction succeeds and round-trips through ``model_dump`` /
   ``model_validate`` losslessly.
2. ``extra='forbid'`` rejects unknown field names (typo guard).
3. Required fields are actually required.
4. Empty-authors and empty-title are rejected (those are XML-parse smells,
   not legitimate data).
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from papertrail.tools.arxiv import ArxivPaper


def _valid_payload() -> dict[str, object]:
    """One canonical, valid payload reused across tests.

    Helper rather than a fixture because we want each test to mutate a fresh
    copy without fixture-scoping surprises.
    """
    return {
        "arxiv_id": "1706.03762",
        "title": "Attention Is All You Need",
        "authors": ["Ashish Vaswani", "Noam Shazeer"],
        "categories": ["cs.CL", "cs.LG"],
        "abstract": "The dominant sequence transduction models are based on...",
        "entry_url": "https://arxiv.org/abs/1706.03762",
        "pdf_url": "https://arxiv.org/pdf/1706.03762",
        "published": datetime(2017, 6, 12, tzinfo=UTC),
        "updated": datetime(2017, 12, 6, tzinfo=UTC),
    }


def test_valid_payload_constructs() -> None:
    paper = ArxivPaper(**_valid_payload())  # type: ignore[arg-type]
    assert paper.arxiv_id == "1706.03762"
    assert paper.authors == ["Ashish Vaswani", "Noam Shazeer"]
    assert paper.categories == ["cs.CL", "cs.LG"]


def test_model_round_trips_through_dump_and_validate() -> None:
    """Serializing then re-validating must yield an equal model.

    This guards the contract that an ``ArxivPaper`` can be persisted (e.g.
    written to a benchmark artifact on disk) and re-loaded without loss.
    """
    original = ArxivPaper(**_valid_payload())  # type: ignore[arg-type]
    dumped = original.model_dump(mode="json")
    rehydrated = ArxivPaper.model_validate(dumped)
    assert rehydrated == original


def test_unknown_field_is_rejected() -> None:
    """``extra='forbid'`` catches typos like ``abstact=`` for ``abstract=``."""
    payload = _valid_payload()
    payload["abstact"] = "typo"  # missing 'r'
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ArxivPaper(**payload)  # type: ignore[arg-type]


def test_empty_authors_is_rejected() -> None:
    """Zero authors signals a parse failure, not a legitimate paper."""
    payload = _valid_payload()
    payload["authors"] = []
    with pytest.raises(ValidationError, match="authors"):
        ArxivPaper(**payload)  # type: ignore[arg-type]


def test_empty_title_is_rejected() -> None:
    payload = _valid_payload()
    payload["title"] = ""
    with pytest.raises(ValidationError, match="title"):
        ArxivPaper(**payload)  # type: ignore[arg-type]


def test_empty_categories_is_allowed() -> None:
    """Pre-taxonomy or unusual entries may legitimately have no categories."""
    payload = _valid_payload()
    payload["categories"] = []
    paper = ArxivPaper(**payload)  # type: ignore[arg-type]
    assert paper.categories == []


def test_malformed_url_is_rejected() -> None:
    """``HttpUrl`` catches malformed links at the boundary, not at render time."""
    payload = _valid_payload()
    payload["entry_url"] = "not-a-url"
    with pytest.raises(ValidationError, match="entry_url"):
        ArxivPaper(**payload)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "missing_field",
    ["arxiv_id", "title", "authors", "abstract", "entry_url", "pdf_url", "published", "updated"],
)
def test_missing_required_field_is_rejected(missing_field: str) -> None:
    """Each required field must actually be required (no silent defaults)."""
    payload = _valid_payload()
    payload.pop(missing_field)
    with pytest.raises(ValidationError, match=missing_field):
        ArxivPaper(**payload)  # type: ignore[arg-type]
