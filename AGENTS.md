# AGENTS.md — comfy-api-graphs

> Guide for contributors and coding agents. Tool-agnostic.

## What this is
Offline Python helper to load, mutate and validate ComfyUI API-format graphs, with no runtime dependencies.

## Layout
- `src/comfy_api_graphs/core.py` — `ComfyWorkflow` (load, mutate, introspect, diff, Mermaid, seed batches).
- `src/comfy_api_graphs/validation.py` — link, `/object_info` schema, orphan and duplicate-seed checks.
- `src/comfy_api_graphs/templates/` — optional generic FLUX/SDXL/utility templates (video ones are stubs).
- `src/comfy_api_graphs/__main__.py` — stdlib CLI: `validate | seeds | show`.
- `tests/` (+ `tests/fixtures/`) — pytest suite; `examples/` — runnable scripts; `docs/` — API reference and scope policy.

## Build, test, run
- Install: `pip install -e ".[dev]"`
- Test: `pytest` (must pass before any PR; 142 tests)
- Lint: `ruff check src tests && black --check src tests && mypy src` (or `make lint`)
- Run: `python -m comfy_api_graphs show tests/fixtures/minimal_api_graph.json --mermaid`

## Conventions
- Branching: `main` is stable; work on `feat/*`, `fix/*`, `docs/*`; open a PR.
- Style: black + ruff + mypy (config in `pyproject.toml`). Keep the core free of runtime dependencies (stdlib only).
- Scope: API-format graphs only; no ComfyUI runtime, no editor-format conversion, no private workflows or model stacks (see `docs/maintainers/SYNC_POLICY.md`).
- Use real ComfyUI `class_type` names; do not invent custom nodes.
- Do not commit secrets, local paths, hostnames or IP addresses; use placeholders (`example.local`, `<HOST>`).
- Add a CHANGELOG entry under `[Unreleased]` for user-visible changes.

## Pointers
- Docs: `docs/` · Changelog: `CHANGELOG.md` · Contributing: `CONTRIBUTING.md`
