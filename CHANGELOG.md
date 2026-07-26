# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `ComfyWorkflow.iter_links()` — yield `(src_id, slot, dst_id, dst_input)` for
  every socket link (walks nested conditioning bags; skips `["0", 0]`).
- `ComfyWorkflow.remove_node(node_id)` — drop a node and clean up dangling links
  pointing at it (downstream inputs are removed, not left as broken refs).
- `ComfyWorkflow.topological_order()` — deterministic dependency order
  (Kahn's algorithm); raises `ValueError` on cycles.
- `ComfyWorkflow.to_mermaid(show_class=True)` — render the graph as a Mermaid
  `flowchart` for README / PR / issue diffs (renders inline on GitHub).
- `ComfyWorkflow.diff(other)` — structured diff (added/removed/changed_class/
  added_inputs/removed_inputs/changed_inputs) between two workflows.
- `ComfyWorkflow.batch_seeds(n, mode="bump", start=None, ...)` — generator
  yielding N mutated copies with distinct seeds; the seed-batch primitive.
- `validate_against_object_info(..., warn_orphans=True)` now warns on
  link-vs-scalar type mismatches and on orphan nodes whose output no other
  node consumes.
- `find_orphan_nodes(workflow, output_class_types=None)` — standalone helper.
- `find_duplicate_seeds_across_batch(workflows, input_keys=None)` — detect
  seed collisions across a batch before queuing (ComfyUI cache-skips
  identical prompts).
- `python -m comfy_api_graphs` — minimal stdlib CLI: `validate`, `seeds`,
  `show` subcommands. No GPU / no runtime deps; fits in a CI step.

### Fixed

- `requires-python` bumped to `>=3.9`. The codebase already used PEP 585
  generic aliases (`set[str]`, `list[tuple[...]]`) which require 3.9+; the
  previous `>=3.8` claim would have broken on a 3.8 install. Python 3.8 is
  EOL since Oct 2024.
- `make lint` now passes on a clean checkout (it previously failed with 67
  ruff errors, 7 black reformats, 1 mypy `no-untyped-def`). The documented
  Black/Ruff/MyPy gate is now real.
- ruff config: moved deprecated top-level `select`/`ignore` to
  `[tool.ruff.lint]` (required by ruff 0.15+).

### Changed

- README badge: Python 3.8+ → 3.9+. Classifiers drop 3.8.
- black / ruff / mypy `target-version` / `python_version` → py39.

## [0.3.0] - 2026-07-11

Close honesty gaps: mutate-first story, object_info schema validation, VRAM
disclaimer, deprecate fake UI graph export, honest install front door.

### Added

- `validate_against_object_info(workflow, object_info)` — catch missing /
  renamed `class_type`, missing required inputs, out-of-range output slots.
- `load_object_info(path_or_dict)` / `fetch_object_info(url)` (stdlib urllib;
  still zero runtime deps). Prefer cached snapshots in CI.
- `estimate_vram_usage` now returns `is_estimate=True` and a hard `disclaimer`.

### Changed

- Package version **0.3.0**.
- README / `__init__` lead with load → mutate → validate → re-queue (templates
  are secondary).
- Install docs: **not on PyPI** — `pip install git+https://...` or local
  `pip install -e .`. `make publish` still refuses silent uploads.
- `validate_workflow_complete` warns that link checks alone are incomplete
  without object_info.

### Deprecated

- `ComfyWorkflow.to_graph_format()` / `save_graph_json()` — empty-links skeleton
  is **not** a UI↔API converter; raises `DeprecationWarning`. Use
  `to_api_format` / `save_api_json`.

## [0.2.0] - 2026-07-11

Minor release: load → mutate → seed batch for existing Save (API Format) graphs.
Install from source (`pip install -e .`). `make publish` stays blocked until a
real PyPI publish is configured.

### Added

- `ComfyWorkflow.from_api_json` / `from_api_dict`: load Save (API Format) graphs
  (or `{"prompt": ...}` wrappers) without rebuilding from templates.
- Mutators: `set_input`, `set_inputs`, `set_input_by_class_type`, `set_input_by_title`.
- `randomize_seeds()` / `bump_seeds(n)` for API batch loops.
- Cookbook: `examples/seed_batch_cookbook.py` (dry-run write or live `/prompt`).
- Fixture + tests under `tests/fixtures/minimal_api_graph.json`.

### Changed

- Package version **0.2.0** (`pyproject.toml`, `__version__`).
- README soft-announce matches the load → mutate → seed-batch story (no oversell).
- Maintainer-only scope notes moved under `docs/maintainers/` (off the community
  front door). CONTRIBUTING keeps the public contribution rules.
- Dropped decorative Black badge, internal release checklist, and similar
  front-door noise. Acknowledgments stay ComfyUI / BFL / Stability only.

### Removed

- Public `ROADMAP.md` and all public pointers (planning stays internal).

### Notes (carried from 0.1.x posture work)

- Public package rename: **comfyui-workflow-builder** → **comfy-api-graphs**
  (import: `comfy_api_graphs`).
- Attribution: **[ U N B R A N D E D ]**.
- Public vs private boundary documented for maintainers only (no bidirectional
  sync fiction, no fake PyPI pins).
- Template curation: FLUX / SDXL / utility patterns; video stubs stay
  quarantined (`NotImplementedError`).

## [0.1.0] - 2026-05-12

Early public-core release. Install locally with `pip install -e .` until a real
PyPI / GitHub publish exists. This is a small ComfyUI API workflow builder
utility — not a full studio pipeline.

### Added

#### Core
- `ComfyWorkflow` — build ComfyUI API-format workflow JSON in Python
- `WorkflowNode` — individual node with output references
- Export helpers: `to_api_format()`, `to_graph_format()`, `save_api_json()`,
  `save_graph_json()`

#### Image templates (generic community patterns)
- FLUX: `FluxTxt2ImgTemplate`, `FluxImg2ImgTemplate`, `FluxInpaintTemplate`
- SDXL: `SdxlTxt2ImgTemplate`, `SdxlImg2ImgTemplate`
- Utility: `CharacterReferenceTemplate`, `UpscaleTemplate`, `InpaintTemplate`,
  `OutpaintTemplate`

#### Video templates (quarantined; not in public ``__all__``)
- ``LtxVideoTemplate``, ``LtxImg2VideoTemplate``, ``WanVideoTemplate`` live under
  ``comfy_api_graphs.templates.video_templates`` as **stubs**:
  ``create()`` raises ``NotImplementedError``. Import them only if you need the
  explicit refusal message. Build real video graphs in the ComfyUI UI (Save API
  Format), then wrap with ``ComfyWorkflow``. Not production recipes.

#### Validation
- `validate_node_references()`, `estimate_vram_usage()`,
  `validate_workflow_complete()`

#### Examples
- `basic_txt2img.py`, `img2img_with_lora.py`, `custom_workflow.py`,
  `execution_example.py`, `utility_templates_example.py`

#### Testing / docs
- pytest suite for core + image/utility templates
- README, API reference notes, CONTRIBUTING, MIT License

### Notes

- Package status: Alpha (`0.1.0`). No claim of PyPI publish or public GitHub
  releases until those channels exist.
- No locked production video recipes ship in this package. Prototype LTX/Wan
  graphs in the ComfyUI UI; this utility does not publish studio video recipes.
