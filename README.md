# Comfy API Graphs

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Zero-dependency Python → ComfyUI **API-format** JSON by **[ U N B R A N D E D ]**.

Build, validate, and export `/prompt` graphs offline: FLUX/SDXL templates + wiring checks. The ComfyUI node editor still wins for complex graphs — this is not a runtime, not a ComfyScript competitor, and not a dump of private worker stacks.

```python
from comfy_api_graphs import FluxTxt2ImgTemplate, ComfyWorkflow

workflow = FluxTxt2ImgTemplate.create(
    prompt="a serene mountain landscape at golden hour",
    width=1024,
    height=1024,
    steps=20,
)

# Export ComfyUI API format
api_payload = workflow.to_api_format()
workflow.save_api_json("my_workflow.json")
```

## Scope

Public core for ComfyUI **API-format graphs**: nodes, latents, checkpoints, LoRAs, `/object_info`, `/prompt`, queue. Generic FLUX/SDXL/utility templates teach wiring — they are not locked production video recipes. Design in the UI → Save (API Format) remains king. Private worker stacks stay private; this package does not ship them.

## Features

- **Programmatic workflows**: Build ComfyUI graphs in pure Python
- **Load and mutate**: `from_api_json` / mutators / seed helpers for Save (API Format) exports
- **Seed batch cookbook**: dry-run write or live `/prompt` loop (`examples/seed_batch_cookbook.py`)
- **Community-safe templates**: FLUX.1 and SDXL T2I / I2I / inpaint, plus utility patterns (upscale, outpaint, character refs)
- **Validation**: Check node references and estimate VRAM before queueing
- **Multiple formats**: API format (execution) or simplified graph format (UI sketch)
- **Honest scope**: Templates teach correct wiring; you still need matching checkpoints, CLIP, VAE, LoRAs, and custom nodes locally
- **Zero runtime dependencies**: Pure Python core

## Installation

From a local clone (recommended until packaging is finalized):

```bash
pip install -e .
pip install -e ".[dev]"        # pytest, etc.
pip install -e ".[examples]"   # optional example extras
```

## Quick Start

### Basic text-to-image (FLUX)

```python
from comfy_api_graphs import FluxTxt2ImgTemplate

workflow = FluxTxt2ImgTemplate.create(
    prompt="a futuristic cityscape with flying cars",
    width=1024,
    height=768,
    steps=20,
    cfg_scale=3.5,
    seed=42,
)
workflow.save_api_json("my_workflow.json")
```

### Image-to-image with LoRA

```python
from comfy_api_graphs import FluxImg2ImgTemplate

workflow = FluxImg2ImgTemplate.create(
    prompt="transform into watercolor painting",
    image_path="input.png",  # under ComfyUI input/
    lora="watercolor_style_v1.safetensors",
    lora_strength=0.8,
    denoise=0.75,
)
```

### Character reference sheets

```python
from comfy_api_graphs import CharacterReferenceTemplate

workflows = CharacterReferenceTemplate.create_batch(
    appearance="young woman with short brown hair",
    style="wearing casual blue sweater",
    base_seed=42,
)
for wf in workflows:
    wf.save_api_json(f"{wf.name}.json")
```

### Custom graph from scratch

```python
from comfy_api_graphs import ComfyWorkflow, validate_node_references

wf = ComfyWorkflow("custom_pipeline")
loader = wf.add_node(
    "CheckpointLoaderSimple",
    {"ckpt_name": "sd_xl_base_1.0.safetensors"},
    title="Load Checkpoint",
)
positive = wf.add_node(
    "CLIPTextEncode",
    {"text": "beautiful landscape", "clip": loader.get_output_ref(1)},
)
errors = validate_node_references(wf)
```

## Available templates

| Template | Purpose | Notes |
|----------|---------|--------|
| `FluxTxt2ImgTemplate` | FLUX.1 T2I | DualCLIP + UNET + VAE + ModelSamplingFlux + KSampler |
| `FluxImg2ImgTemplate` | FLUX.1 I2I | LoadImage → VAEEncode → denoise |
| `FluxInpaintTemplate` | FLUX.1 inpaint | ImageToMask + SetLatentNoiseMask |
| `SdxlTxt2ImgTemplate` | SDXL T2I | Optional refiner second pass |
| `SdxlImg2ImgTemplate` | SDXL I2I | Checkpoint + denoise |
| `UpscaleTemplate` | Upscale | UpscaleModelLoader or LatentUpscaleBy |
| `InpaintTemplate` | SDXL-class inpaint | SetLatentNoiseMask |
| `OutpaintTemplate` | Outpaint | ImagePadForOutpaint + mask |
| `CharacterReferenceTemplate` | Multi-angle refs | SDXL T2I prompt batch |

**Not shipped as working recipes:** LTX / Wan video templates. Stubs under `templates/video_templates.py` raise `NotImplementedError`. Video graphs change with custom nodes and model revisions — build them in the ComfyUI UI, Save (API Format), then wrap or edit with `ComfyWorkflow`. This package does **not** ship studio video production recipes.

## How a community user should use these

1. Install ComfyUI and place the matching model files (checkpoint / UNET / CLIP / VAE / LoRA / upscaler).
2. `pip install -e .` from this repo.
3. Run an example under `examples/` to emit API JSON, or call a template `create()` in your own script.
4. Queue with `POST http://127.0.0.1:8188/prompt` and `{"prompt": workflow.to_api_format()}`.
5. If a node `class_type` is missing on your server, install the custom node pack or redesign that hop in the UI — templates use well-known core nodes on purpose.

## API sketch

### `ComfyWorkflow`

```python
wf = ComfyWorkflow(name="my_workflow")
node = wf.add_node("KSampler", {...}, title="My Sampler")
api_format = wf.to_api_format()
wf.save_api_json("workflow_api.json")
```

### `WorkflowNode.get_output_ref(i)`

Returns `[node_id, output_index]` for wiring into another node's inputs.

### `validate_node_references(workflow)` / `estimate_vram_usage(workflow)`

Sanity-check wiring and rough VRAM before you burn a queue slot.

## Executing workflows

This library builds JSON; it does not run diffusion. See `examples/execution_example.py` for a localhost `/prompt` + history poll loop.

Load an existing Save (API Format) export, mutate seeds, and batch:

```python
from comfy_api_graphs import ComfyWorkflow

wf = ComfyWorkflow.from_api_json("my_graph_api.json")
wf.set_input_by_class_type("CLIPTextEncode", "text", "a new prompt")
wf.randomize_seeds()
wf.save_api_json("my_graph_api.seeded.json")
```

Or run `examples/seed_batch_cookbook.py` for a dry-run / live queue loop.

## Examples

Files that exist under `examples/`:

- `basic_txt2img.py` — FLUX T2I
- `img2img_with_lora.py` — FLUX I2I + LoRA stack
- `custom_workflow.py` — low-level `ComfyWorkflow` build
- `utility_templates_example.py` — upscale / inpaint / outpaint / character refs
- `execution_example.py` — queue on local ComfyUI
- `seed_batch_cookbook.py` — load API JSON → mutate seeds → write or `/prompt`
- `video_generation.py` — shows quarantined stubs raise `NotImplementedError`

Examples write `*.json` at the repo root (gitignored). Regenerate anytime by running them.

## Creating custom templates

```python
from comfy_api_graphs import ComfyWorkflow

class MyCustomTemplate:
    @staticmethod
    def create(prompt: str, width: int = 1024, height: int = 1024) -> ComfyWorkflow:
        wf = ComfyWorkflow("my_custom_workflow")
        # add CheckpointLoaderSimple / CLIPTextEncode / KSampler / SaveImage ...
        return wf
```

Prefer real ComfyUI `class_type` names from `/object_info`. Do not invent speculative custom nodes.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT License — Copyright (c) 2026 [ U N B R A N D E D ]. See [LICENSE](LICENSE).

## Acknowledgments

Thanks to the ComfyUI team and the wider custom-node community. FLUX.1 is by Black Forest Labs; SDXL is by Stability AI — templates here only teach common ComfyUI wiring patterns around those models.
