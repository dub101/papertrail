# 0001 — Use uv for dependency management

- **Status:** Accepted
- **Date:** 2026-05-12
- **Deciders:** dub101

## Context

PaperTrail needs a Python dependency manager that supports `pyproject.toml`, locks reproducibly, manages the virtualenv, and runs fast enough that CI and pre-commit feel responsive across a multi-month project.

Candidates considered:

- `uv` — single Rust binary; resolves and locks in one tool; written by the ruff authors so the toolchain is coherent
- `poetry` — well-established, opinionated, slower; some friction with PEP 621
- `pip` + `pip-tools` (`pip-compile`) — boring, no extra tool, but multi-step workflow and manual venv management

## Decision

Use `uv`.

## Consequences

**Easier:**
- Fast resolve + install — matters in CI and pre-commit
- Single binary, single config in `pyproject.toml`
- Cohesive toolchain with `ruff` (same vendor)
- `uv.lock` is platform-independent and committed for reproducibility
- `uv run <cmd>` removes the "did I activate the venv?" friction

**Harder / traded away:**
- User is new to `uv` — small learning curve
- Less ecosystem maturity than `poetry`; some plugins/integrations may not exist
- If `uv` development stalls, migration cost is non-zero (mitigated by `pyproject.toml` being standard)

## Alternatives considered

- **`poetry`** — rejected for slower resolution and PEP 621 friction
- **`pip` + `pip-tools`** — rejected for the multi-step workflow and lack of integrated venv management
