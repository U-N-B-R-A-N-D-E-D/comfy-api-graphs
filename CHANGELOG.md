# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
