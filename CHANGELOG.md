# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed — public package rename

- Renamed **comfyui-workflow-builder** → **comfy-api-graphs** (import: `comfy_api_graphs`).
  Sharper wedge: offline API-format graph builders, not a generic “workflow builder” /
  runtime competitor. PyPI + GitHub slug free at rename time.

### Changed — public attribution

- Credit / copyright / authors / maintainers → **[ U N B R A N D E D ]**
- Removed studio homepage URL from `pyproject.toml` (no fake publish links)
- Docs speak experience-informed ComfyAPI automation — no “extracted from private
  production” marketing; private stacks named generically in sync/scope docs

### Changed — product scope honesty (public core posture)

- Rewrote `PERSONALIZATION.md` and `SYNC_POLICY.md`: public package = **minimal shared
  core**; private production stacks stay separate (private JSON/workers). No bidirectional
  sync, no production pin-to-PyPI fiction, no public inventory of institutional model filenames.
- README Scope block points maintainers at that boundary; UI → Save (API Format) remains king.
- Version remains **0.1.0** until a real publish decision — this is documentation /
  posture, not a feature release.

### Changed — template gold-mine curation

- FLUX T2I/I2I/inpaint: empty negative via real `CLIPTextEncode("")`; `ModelSamplingFlux`
  includes width/height; inpaint uses `ImageToMask` + `SetLatentNoiseMask`
- SDXL refiner path re-encodes prompts with the refiner CLIP
- Outpaint: `ImagePadForOutpaint` mask → `SetLatentNoiseMask`; latent upscale uses
  `LatentUpscaleBy`
- Public package `__all__` exports FLUX + SDXL + utilities only (no LTX/Wan)
- Examples: `utility_templates_example.py` replaces video demo; README lists real files
- Root sample JSON regenerated from curated FLUX templates; stale LTX/Wan JSON removed

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
