# Architecture Decision Records (ADRs)

Durable records of significant architectural decisions made on PaperTrail.

## Why ADRs

- They survive memory wipes and session resets — anyone reading the repo can reconstruct the *why*
- They prevent re-litigating decisions in future sessions
- They are referenced from [`CLAUDE.md`](../../CLAUDE.md) and from per-architecture READMEs

## Format

Each ADR is a markdown file named `NNNN-short-title.md` where `NNNN` is a zero-padded sequence number. Template:

```markdown
# NNNN — Short Title

- **Status:** Proposed | Accepted | Superseded by ADR-XXXX | Deprecated
- **Date:** YYYY-MM-DD
- **Deciders:** names

## Context

The forces at play and the issue that motivates the decision.

## Decision

What we decided.

## Consequences

What becomes easier or harder. What we traded away.

## Alternatives considered

What else was on the table and why it was not chosen.
```

## Index

- [0001 — Use uv for dependency management](0001-use-uv.md)
- [0002 — `src/` layout for the Python package](0002-src-layout.md)
- [0003 — Model policy: Haiku default, Sonnet for Evaluator, never Opus](0003-haiku-default.md)
