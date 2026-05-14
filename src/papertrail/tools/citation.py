"""Citation formatting tool.

Public surface:

- ``format_citation(paper) -> str``  — pure sync, returns one minimal but
  correct citation line.

Format::

    {author_part} ({year}). {title}. arXiv:{arxiv_id}

Where ``author_part`` is the surname (or surnames) of the author list:

- 1 author:   ``"Vaswani"``
- 2 authors:  ``"Vaswani and Shazeer"``
- 3+ authors: ``"Vaswani et al."``

Surname extraction uses a last-whitespace-token heuristic — fast and
~90% correct on real arXiv data. The known weakness is names with
particles (``"van der Schaar"`` is cited as ``"Schaar"`` rather than the
correct ``"van der Schaar"``); accepted because the alternatives are a
hardcoded particle list to maintain or a new dependency, and the output
is consumed by the benchmark Evaluator rather than a published journal.

The arxiv_id is preserved as arXiv returned it — including a version
suffix if present (e.g. ``"1706.03762v5"``). Same precision cost as
stripping, slightly more information.

Cert mapping: D2 (Tool Design), TS 2.1 — narrow output contract for a
pure function with no I/O; the tool's signature is its full contract.
"""

from __future__ import annotations

from papertrail.tools.arxiv import ArxivPaper


def _surname(full_name: str) -> str:
    """Return the citation surname for an arXiv author string.

    arXiv gives author names as a single full string (``"Ashish Vaswani"``).
    We split on whitespace and take the last token. This is the standard
    cheap heuristic for cite-by-surname; particles and compound surnames
    are documented as a known limitation in the module docstring.
    """
    parts = full_name.split()
    # ``full_name`` itself is the fallback for the (extremely unlikely)
    # all-whitespace input — better than returning empty string.
    return parts[-1] if parts else full_name


def format_citation(paper: ArxivPaper) -> str:
    """Format one paper as a minimal-but-correct citation line.

    Pure sync function — no I/O, no exceptions for normal input (all
    necessary fields are guaranteed-present by ``ArxivPaper`` itself).

    Parameters
    ----------
    paper:
        The paper to cite. All required fields come from ``ArxivPaper``,
        which is already validated, so we trust them here.

    Returns
    -------
    str
        A single-line citation like
        ``"Vaswani et al. (2017). Attention Is All You Need. arXiv:1706.03762v5"``.
    """
    authors = paper.authors
    year = paper.published.year

    if len(authors) == 1:
        author_part = _surname(authors[0])
    elif len(authors) == 2:
        author_part = f"{_surname(authors[0])} and {_surname(authors[1])}"
    else:
        # 3 or more: by convention only the first author is named, "et al."
        # acknowledges the rest. We do not pluralize or change form by count.
        author_part = f"{_surname(authors[0])} et al."

    return f"{author_part} ({year}). {paper.title}. arXiv:{paper.arxiv_id}"
