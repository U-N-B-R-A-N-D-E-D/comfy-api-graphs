# Contributing

Contributions welcome — especially if you already build graphs in ComfyUI and care about API-format correctness.

Maintained by **[ U N B R A N D E D ]**.

## Setup

This package is not assumed to be on PyPI. Work from a local clone / tree:

```bash
cd /path/to/comfy-api-graphs
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev,examples]"
```

Optional: `pre-commit install` if you use the project hooks.

## Workflow for changes

1. Branch for the change.
2. Implement against the public core (`ComfyWorkflow`, validators, generic templates).
3. Run tests and linters.
4. Open a PR with what you verified (ideally against a live ComfyUI).

```bash
pytest
ruff check src tests
black --check src tests
mypy src
```

## ComfyUI-minded rules

This is a graph-construction library. Treat node identity like an API contract.

- **Verify `class_type` names.** Guessing custom-node class names is a common failure mode. Prefer checking `GET http://localhost:8188/object_info` (or your instance URL) before submitting a template or example that introduces new nodes.
- **API format vs UI format.** Ship and test **API-format** graphs (`to_api_format` / Save API Format). Do not PR “it looked right in the UI JSON” without an API-format payload that queues.
- **Test against live Comfy when you can.** Unit tests catch broken links; only a running ComfyUI + `/prompt` proves checkpoint paths, LoRA loaders, and custom nodes resolve. Mark live checks clearly; do not require CI to reach a GPU box.
- **Do not submit unverified node names.** If you cannot run `/object_info` or queue a minimal graph, say so in the PR and keep the change scoped (docs-only, link-validator fix, etc.).
- **Keep proprietary stacks out.** Institutional checkpoints, locked video recipes, private orchestration wiring, and private worker JSON do not belong here. Generic FLUX/SDXL/utility patterns only. See `SYNC_POLICY.md` / `PERSONALIZATION.md`.
- **No bidirectional “sync with studio” PRs.** This package is not the source of truth for production workers. Do not land private pipeline dumps “for parity.”
- **Video templates stay quarantined stubs.** `create()` raises `NotImplementedError`. Do not market LTX/Wan as working recipes. Never paste locked production knobs.

## Adding a template

1. Put it in the right module under `src/comfy_api_graphs/templates/`.
2. Static `create()` (or `create_batch()`), typed params, sensible defaults, Google-style docstring.
3. Export from `templates/__init__.py` and package `__init__.py` `__all__`.
4. Add tests in `tests/test_templates.py` (at least: builds, `validate_node_references` clean, ends with something like `SaveImage` where applicable).
5. Add or update an example under `examples/` if useful.
6. Update `README.md` and `docs/api-reference.md`.

Pattern:

```python
class MyNewTemplate:
    """One-line: what graph this builds."""

    @staticmethod
    def create(
        prompt: str,
        width: int = 1024,
        height: int = 1024,
        seed: int = -1,
        model: str = "sd_xl_base_1.0.safetensors",
    ) -> ComfyWorkflow:
        wf = ComfyWorkflow("my_template")
        # CheckpointLoaderSimple → CLIPTextEncode → EmptyLatent → KSampler → ...
        return wf
```

Wire links with `node.get_output_ref(index)` — same `[node_id, output_index]` links Comfy uses in API format.

## Project layout

```
src/comfy_api_graphs/
  core.py           # WorkflowNode, ComfyWorkflow
  validation.py     # validate_node_references, estimate_vram_usage, ...
  templates/        # FLUX, SDXL, utility (+ video_templates stubs, not public API)
tests/
examples/
docs/
```

## Style

- Black (100 cols), Ruff, MyPy strict on `src`
- Type hints on public functions
- Docstrings on public APIs
- Conventional Commits: `feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `chore:`

## PR checklist

- [ ] Tests pass locally
- [ ] Docs updated if the public API or templates changed
- [ ] New `class_type`s checked against `/object_info` or explicitly called out as unverified
- [ ] No proprietary model names or locked production recipes
- [ ] Breaking changes called out in the PR body

## License

Contributions are under the MIT License. Copyright (c) 2026 [ U N B R A N D E D ].
