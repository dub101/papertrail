"""Tests for ``format_citation`` — author-count branches, surname extraction,
year derivation, arxiv_id pass-through, and the documented particle-name
limitation.

No HTTP, no fixtures — ``format_citation`` is pure sync. Tests construct
``ArxivPaper`` instances directly with a small helper.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from papertrail.tools.arxiv import ArxivPaper
from papertrail.tools.citation import _surname, format_citation


def _make_paper(
    *,
    authors: list[str],
    title: str = "Some Paper",
    arxiv_id: str = "1706.03762v5",
    published_year: int = 2017,
) -> ArxivPaper:
    """Construct an ``ArxivPaper`` for citation tests.

    Citation only cares about authors, title, arxiv_id, and published year.
    The other required fields get plausible defaults so each test can stay
    focused on the field under examination.
    """
    return ArxivPaper(
        arxiv_id=arxiv_id,
        title=title,
        authors=authors,
        categories=["cs.LG"],
        abstract="Irrelevant for citation tests.",
        entry_url="https://arxiv.org/abs/" + arxiv_id,  # type: ignore[arg-type]
        pdf_url="https://arxiv.org/pdf/" + arxiv_id,  # type: ignore[arg-type]
        published=datetime(published_year, 6, 1, tzinfo=UTC),
        updated=datetime(published_year, 12, 1, tzinfo=UTC),
    )


# ───── Author-count branches ────────────────────────────────────────────


def test_single_author_no_et_al() -> None:
    """One author -> just the surname, no 'and' or 'et al.'."""
    paper = _make_paper(authors=["Plato Aristocles"])
    assert format_citation(paper).startswith("Aristocles (")


def test_two_authors_uses_and() -> None:
    """Two authors -> 'A and B', never 'et al.'."""
    paper = _make_paper(authors=["Ashish Vaswani", "Noam Shazeer"])
    citation = format_citation(paper)
    assert "Vaswani and Shazeer" in citation
    assert "et al." not in citation


def test_three_authors_uses_et_al() -> None:
    """Three authors -> 'First et al.', second and third names dropped."""
    paper = _make_paper(authors=["Ashish Vaswani", "Noam Shazeer", "Niki Parmar"])
    citation = format_citation(paper)
    assert "Vaswani et al." in citation
    assert "Shazeer" not in citation
    assert "Parmar" not in citation


def test_many_authors_uses_et_al_unchanged() -> None:
    """Form doesn't change with author count above 3 — always 'First et al.'."""
    paper = _make_paper(authors=[f"Author {i} Surname{i}" for i in range(10)])
    citation = format_citation(paper)
    assert "Surname0 et al." in citation


# ───── Full output shape ────────────────────────────────────────────────


def test_full_citation_format() -> None:
    """End-to-end format string matches the documented shape exactly."""
    paper = _make_paper(
        authors=["Ashish Vaswani", "Noam Shazeer", "Niki Parmar"],
        title="Attention Is All You Need",
        arxiv_id="1706.03762v5",
        published_year=2017,
    )
    assert (
        format_citation(paper)
        == "Vaswani et al. (2017). Attention Is All You Need. arXiv:1706.03762v5"
    )


def test_arxiv_id_with_version_is_preserved() -> None:
    """We deliberately keep the version suffix; same length, more info."""
    paper = _make_paper(authors=["Tri Dao"], arxiv_id="2205.14135v2")
    assert "arXiv:2205.14135v2" in format_citation(paper)


def test_arxiv_id_without_version_passes_through() -> None:
    """Bare ids are echoed verbatim too — the tool doesn't synthesise a version."""
    paper = _make_paper(authors=["Tri Dao"], arxiv_id="2205.14135")
    assert "arXiv:2205.14135" in format_citation(paper)
    # Make sure we don't accidentally append something.
    assert "arXiv:2205.14135v" not in format_citation(paper)


def test_year_comes_from_published_not_updated() -> None:
    """Year for the citation is the original submission, not the last revision.
    A paper submitted in 2017 and revised in 2024 should still cite as 2017."""
    paper = ArxivPaper(
        arxiv_id="1706.03762v5",
        title="Attention Is All You Need",
        authors=["Ashish Vaswani"],
        categories=[],
        abstract="Irrelevant.",
        entry_url="https://arxiv.org/abs/1706.03762v5",  # type: ignore[arg-type]
        pdf_url="https://arxiv.org/pdf/1706.03762v5",  # type: ignore[arg-type]
        published=datetime(2017, 6, 12, tzinfo=UTC),
        updated=datetime(2024, 1, 1, tzinfo=UTC),
    )
    assert "(2017)" in format_citation(paper)
    assert "(2024)" not in format_citation(paper)


# ───── Surname extraction edge cases ────────────────────────────────────


@pytest.mark.parametrize(
    ("full_name", "expected_surname"),
    [
        ("Ashish Vaswani", "Vaswani"),
        ("Yann LeCun", "LeCun"),
        ("Plato", "Plato"),  # single-word name
        ("  John   Doe  ", "Doe"),  # whitespace tolerated
        ("J. K. Rowling", "Rowling"),  # initials
    ],
)
def test_surname_extraction(full_name: str, expected_surname: str) -> None:
    """Standard names: last whitespace-token is the surname."""
    assert _surname(full_name) == expected_surname


def test_surname_particle_limitation_is_documented() -> None:
    """KNOWN LIMITATION: names with particles ('van der X') resolve to just
    the last token, not the particle-prefixed compound. This test pins the
    current behaviour — if we ever upgrade to a name-parsing lib, the
    assertion flips and we know to update the docstring."""
    assert _surname("Carolyn van der Schaar") == "Schaar"


def test_surname_empty_string_fallback() -> None:
    """All-whitespace input falls back to the original string rather than
    returning empty — empty surnames would produce nonsense citations."""
    assert _surname("   ") == "   "
