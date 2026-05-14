"""Versioned prompt store for PaperTrail.

This package is the project's "prompts as data" layer: each agent's system
prompt lives here as a ``.md`` file named ``<agent>_<version>.md``. Bumping
a prompt is a code change with a visible diff, and the version string is
recorded in ``Telemetry.prompt_versions`` so a stored ``BenchmarkResult``
can be replayed against the exact prompt that produced it.

Convention:
    - One file per (agent, version) pair: ``evaluator_v1.md``, ``evaluator_v2.md``, ...
    - Filenames are lowercase, version strings start with ``v``
    - Body is markdown — Claude parses markdown natively and a human can read it

Lookup:
    >>> from papertrail.prompts import load_prompt
    >>> text = load_prompt("evaluator", "v1")

Cert mapping: D5 TS 5.3 (structured handoff / prompt provenance) and D4 TS 4.6
(versioned, independent grader prompts so reviews are reproducible).
"""

from __future__ import annotations

import importlib.resources


def load_prompt(name: str, version: str) -> str:
    """Load a versioned prompt by name and version.

    Args:
        name: The agent name, e.g. ``"evaluator"``.
        version: The version slug, e.g. ``"v1"``.

    Returns:
        The prompt body as a string, UTF-8 decoded.

    Raises:
        FileNotFoundError: If ``<name>_<version>.md`` does not exist in this
            package. We let the underlying filesystem error propagate rather
            than wrapping it — the missing-file path is a bug, not a runtime
            condition to recover from.

    Why ``importlib.resources`` instead of ``open()``:
        Works whether the project is installed editable (current setup) or
        as a built wheel later. Wheels package non-``.py`` files only if the
        build backend is told to; ``importlib.resources`` is the canonical
        Python way to read them either way.
    """
    filename = f"{name}_{version}.md"
    return importlib.resources.files(__name__).joinpath(filename).read_text(encoding="utf-8")


__all__ = ["load_prompt"]
