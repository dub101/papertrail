# 0002 — `src/` layout for the Python package

- **Status:** Accepted
- **Date:** 2026-05-12
- **Deciders:** dub101

## Context

The package can live at either:

- `src/papertrail/` (**src-layout**) — forces installation before import; protects against "works in dev, fails in CI" packaging bugs
- `papertrail/` (**flat**) — simpler navigation; what most tutorials show; no install step required for imports to resolve

For a strictly-typed (mypy strict) project with tests and CI, packaging correctness matters early. A bug where the project works locally because of `PYTHONPATH` quirks but breaks on a clean install is exactly the kind of bug we want to catch on day one.

A related question: where do **architectures** live? Inside the package (`src/papertrail/architectures/...`) or as a sibling top-level package?

## Decision

- Use `src/papertrail/` layout
- Architectures live **inside** the package as `src/papertrail/architectures/arch_NN_*/`

## Consequences

**Easier:**
- Tests cannot accidentally import from working-tree files outside the package — they import from the installed package, the way real consumers will
- Packaging correctness is verified on every `uv sync` + `pytest` cycle
- Subpackage imports are clean and package-relative: `from papertrail.architectures.arch_01 import ...`
- Architectures are first-class citizens of the project, not external scripts

**Harder / traded away:**
- One extra directory level in paths
- Some IDEs require src-layout configuration for auto-import resolution (one-time setup)

## Alternatives considered

- **Flat `papertrail/` layout** — rejected because we want packaging bugs caught early in a strict project
- **`architectures/` as a sibling top-level package** — rejected; architectures need clean package-relative imports and should share code with `papertrail.tools`, `papertrail.models`, etc.
