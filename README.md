# PaperTrail

Multi-architecture arxiv research intelligence system. Given a research topic, produces a top-10 paper list, a topic-evolution timeline, and an impact analysis — implemented across **7 architectures** so orchestration patterns can be compared head-to-head.

Built as cert prep for the **Claude Certified Architect – Foundations** exam.

## Quickstart

```bash
# Install uv (one time): https://docs.astral.sh/uv/
curl -LsSf https://astral.sh/uv/install.sh | sh

# Set up the project
uv sync                              # install deps + create .venv
cp .env.example .env                 # fill in ANTHROPIC_API_KEY
docker compose up -d qdrant          # start vector store
uv run pytest                        # run tests

# Pre-commit hooks (one time per clone)
uv run pre-commit install
```

## Documentation

- **[`docs/PROJECT_BRIEF.md`](docs/PROJECT_BRIEF.md)** — canonical project specification (verbatim original brief)
- **[`CLAUDE.md`](CLAUDE.md)** — operational guide for Claude sessions
- **[`docs/decisions/`](docs/decisions/)** — Architecture Decision Records (ADRs)

## Architectures

| ID | Pattern | Status |
|---|---|---|
| arch_00 | Naive baseline (no LLM) — adapted from legacy `apps/` | not yet adapted |
| arch_01 | Sequential Pipeline | not started |
| arch_02 | Map-Reduce | not started |
| arch_03 | Parallel Fan-Out | not started |
| arch_04 | Plan-and-Execute | not started |
| arch_05 | Critic-Revisor Loop | not started |
| arch_06 | Hierarchical Orchestration + MCP server | not started |

Every architecture produces the same `BenchmarkResult` schema and is scored by a dedicated Evaluator Agent.

## Stack

Python 3.11+ · `uv` · `anthropic` SDK (raw) · `mcp` SDK · pydantic v2 · httpx async · asyncio · structlog → JSON · rich · pytest + pytest-asyncio · mypy strict · ruff · Docker · `act` for local CI
