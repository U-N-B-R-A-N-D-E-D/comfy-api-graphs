# API Reference

## Core Classes

### `ComfyWorkflow`

The main container for building workflows.

```python
class ComfyWorkflow:
    def __init__(self, name: str = "workflow")
    def add_node(self, class_type: str, inputs: Dict[str, Any],
                 node_id: Optional[str] = None,
                 title: Optional[str] = None) -> WorkflowNode
    def get_node(self, node_id: str) -> Optional[WorkflowNode]
    @classmethod
    def from_api_json(cls, path_or_dict, name: Optional[str] = None) -> "ComfyWorkflow"
    @classmethod
    def from_api_dict(cls, data: Dict[str, Any], name: str = "workflow") -> "ComfyWorkflow"
    def set_input(self, node_id: str, key: str, value: Any) -> "ComfyWorkflow"
    def set_inputs(self, node_id: str, **inputs: Any) -> "ComfyWorkflow"
    def set_input_by_class_type(self, class_type: str, key: str, value: Any,
                                *, all_matches: bool = True) -> int
    def set_input_by_title(self, title: str, key: str, value: Any,
                           *, all_matches: bool = True) -> int
    def randomize_seeds(self, ...) -> List[str]
    def bump_seeds(self, n: int = 1, ...) -> List[str]
    def iter_links(self) -> Iterable[Tuple[str, int, str, str]]
    def remove_node(self, node_id: str) -> "ComfyWorkflow"
    def topological_order(self) -> List[str]
    def to_mermaid(self, *, show_class: bool = True) -> str
    def diff(self, other: "ComfyWorkflow") -> Dict[str, Any]
    def batch_seeds(self, n: int, *, mode: str = "bump",
                    start: Optional[int] = None, ...) -> Iterable["ComfyWorkflow"]
    def to_api_format(self) -> Dict[str, Any]
    def to_graph_format(self) -> Dict[str, Any]  # DEPRECATED — empty links
    def save_api_json(self, filepath: str) -> None
    def save_graph_json(self, filepath: str) -> None  # DEPRECATED
```

Load a Save (API Format) export, mutate prompts/seeds, then re-export or queue.
See `examples/seed_batch_cookbook.py`.

`to_graph_format` / `save_graph_json` raise `DeprecationWarning` and do **not**
produce a frontend-loadable workflow.

#### Graph introspection / mutation (v0.4)

- `iter_links()` — yield `(src_id, src_slot, dst_id, dst_input)` for every
  socket link. Walks nested conditioning bags; skips the `["0", 0]` placeholder.
- `remove_node(node_id)` — drop a node and clean up dangling links pointing at
  it (downstream inputs are removed, not left as broken refs). Raises
  `KeyError` if missing.
- `topological_order()` — deterministic dependency order (Kahn's algorithm,
  insertion-order tie-break). Raises `ValueError` on cycles.
- `to_mermaid(show_class=True)` — render the graph as a Mermaid `flowchart` for
  README / PR / issue diffs (renders inline on GitHub).
- `diff(other)` — structured diff: `added`, `removed`, `changed_class`,
  `added_inputs`, `removed_inputs`, `changed_inputs` (per node id).
- `batch_seeds(n, mode="bump", start=None, ...)` — generator yielding `n`
  mutated copies with distinct seeds (`bump` or `randomize`). The seed-batch
  primitive for `/prompt` loops; the original workflow is unchanged.

### `WorkflowNode`

Represents a single node in a workflow.

```python
@dataclass
class WorkflowNode:
    node_id: str
    class_type: str
    inputs: Dict[str, Any]
    title: Optional[str]

    def to_api_dict(self) -> Dict[str, Any]
    def get_output_ref(self, output_index: int = 0) -> List[Union[str, int]]
```

## Templates (public)

### FLUX

#### `FluxTxt2ImgTemplate`

```python
@staticmethod
def create(
    prompt: str,
    width: int = 1024,
    height: int = 1024,
    seed: int = -1,
    steps: int = 20,
    cfg_scale: float = 3.5,
    model: str = "flux1-dev.safetensors",
    vae: str = "ae.safetensors",
    clip1: str = "t5xxl_fp16.safetensors",
    clip2: str = "clip_l.safetensors",
    lora: Optional[str] = None,
    lora_strength: float = 1.0,
    filename_prefix: str = "flux_",
) -> ComfyWorkflow
```

#### `FluxImg2ImgTemplate`

```python
@staticmethod
def create(
    prompt: str,
    image_path: str,
    width: int = 1024,
    height: int = 1024,
    seed: int = -1,
    steps: int = 20,
    cfg_scale: float = 3.5,
    denoise: float = 0.75,
    model: str = "flux1-dev.safetensors",
    vae: str = "ae.safetensors",
    clip1: str = "t5xxl_fp16.safetensors",
    clip2: str = "clip_l.safetensors",
    lora: Optional[str] = None,
    lora_strength: float = 1.0,
    filename_prefix: str = "flux_i2i_",
) -> ComfyWorkflow
```

#### `FluxInpaintTemplate`

```python
@staticmethod
def create(
    prompt: str,
    image_path: str,
    mask_path: str,
    width: int = 1024,
    height: int = 1024,
    seed: int = -1,
    steps: int = 20,
    cfg_scale: float = 3.5,
    denoise: float = 1.0,
    model: str = "flux1-dev.safetensors",
    vae: str = "ae.safetensors",
    clip1: str = "t5xxl_fp16.safetensors",
    clip2: str = "clip_l.safetensors",
    filename_prefix: str = "flux_inpaint_",
) -> ComfyWorkflow
```

### SDXL

#### `SdxlTxt2ImgTemplate`

```python
@staticmethod
def create(
    prompt: str,
    negative_prompt: str = "",
    width: int = 1024,
    height: int = 1024,
    seed: int = -1,
    steps: int = 30,
    cfg_scale: float = 7.0,
    base_model: str = "sd_xl_base_1.0.safetensors",
    refiner_model: Optional[str] = None,
    vae: Optional[str] = "sdxl_vae.safetensors",
    lora: Optional[str] = None,
    lora_strength: float = 1.0,
    filename_prefix: str = "sdxl_",
) -> ComfyWorkflow
```

#### `SdxlImg2ImgTemplate`

```python
@staticmethod
def create(
    prompt: str,
    image_path: str,
    negative_prompt: str = "",
    width: int = 1024,
    height: int = 1024,
    seed: int = -1,
    steps: int = 30,
    cfg_scale: float = 7.0,
    denoise: float = 0.65,
    model: str = "sd_xl_base_1.0.safetensors",
    vae: Optional[str] = "sdxl_vae.safetensors",
    lora: Optional[str] = None,
    lora_strength: float = 1.0,
    filename_prefix: str = "sdxl_i2i_",
) -> ComfyWorkflow
```

### Utility

- `UpscaleTemplate.create(...)` — UpscaleModelLoader or LatentUpscaleBy
- `InpaintTemplate.create(...)` — SetLatentNoiseMask
- `OutpaintTemplate.create(...)` — ImagePadForOutpaint
- `CharacterReferenceTemplate.create_batch(...)` / `create_single(...)` — SDXL T2I angles

### Video (quarantined)

`LtxVideoTemplate`, `LtxImg2VideoTemplate`, and `WanVideoTemplate` live under
`comfy_api_graphs.templates.video_templates` as stubs that always raise
`NotImplementedError`. They are **not** exported from the package `__init__`.
Build video graphs in the ComfyUI UI and Save (API Format).

## Validation Functions

### `validate_node_references`

```python
def validate_node_references(workflow: ComfyWorkflow) -> List[str]
```

Validates that all node references point to existing nodes.

**Returns:** List of error messages (empty if valid)

### `load_object_info` / `fetch_object_info`

```python
def load_object_info(path_or_dict) -> Dict[str, Any]
def fetch_object_info(url: str = "http://127.0.0.1:8188/object_info", *, timeout: float = 60.0) -> Dict[str, Any]
```

Load a cached `/object_info` snapshot or fetch from a running ComfyUI (stdlib).

### `validate_against_object_info`

```python
def validate_against_object_info(
    workflow: ComfyWorkflow,
    object_info: Dict[str, Any],
    *,
    unknown_inputs_as_errors: bool = False,
    warn_orphans: bool = True,
) -> Dict[str, Any]
```

Schema-check `class_type`, required inputs, output-slot bounds, and (as
warnings) link-vs-scalar type mismatches and orphan nodes whose output is
never consumed.

**Returns:** `valid`, `errors`, `warnings`, `info`, `node_count`.

### `find_orphan_nodes`

```python
def find_orphan_nodes(
    workflow: ComfyWorkflow,
    *,
    output_class_types: Optional[Iterable[str]] = None,
) -> List[str]
```

Return sorted node ids whose output no other node consumes and which are not
recognized output nodes (default: SaveImage, PreviewImage, SaveVideo,
VHS_VideoCombine, SaveAnimatedPNG).

### `find_duplicate_seeds_across_batch`

```python
def find_duplicate_seeds_across_batch(
    workflows: Sequence[ComfyWorkflow],
    *,
    input_keys: Optional[Iterable[str]] = None,
) -> Dict[int, List[Tuple[int, str]]]
```

Detect seed collisions across a batch before queuing. ComfyUI cache-skips
identical prompts, so a collision silently wastes queue slots. Returns
`{seed_value: [(workflow_index, node_id), ...]}` for colliding values only.

### `estimate_vram_usage`

```python
def estimate_vram_usage(workflow: ComfyWorkflow) -> Dict[str, Any]
```

**Planning hint only** — not a VRAM predictor (GGUF/NVFP4/resolution dominate).

**Returns:** Dictionary with:
- `estimated_vram_gb`: Hint number from class_type heuristics
- `is_estimate`: Always `True`
- `disclaimer`: Hard warning string — do not size hardware on this
- `model_vram_gb`, `overhead_gb`, `model_loaders`, `details`
- `recommendation`: Soft hint language only

### `validate_workflow_complete`

```python
def validate_workflow_complete(workflow: ComfyWorkflow) -> Dict[str, Any]
```

Link + completeness heuristics only. Prefer `validate_against_object_info`
before production queues.

## Command-line interface

A minimal, stdlib-only CLI for scripting in CI without a GPU:

```bash
# Validate links + optional object_info schema
python -m comfy_api_graphs validate graph_api.json [--object-info snap.json]

# Mutate seeds and write N copies (no queue, no GPU)
python -m comfy_api_graphs seeds graph_api.json --bump 100 --count 5 --out-dir out/
python -m comfy_api_graphs seeds graph_api.json --randomize --count 5 --out-dir out/

# Render the graph (topological order or Mermaid for PR/issue diffs)
python -m comfy_api_graphs show graph_api.json
python -m comfy_api_graphs show graph_api.json --mermaid
```

`seeds --fail-on-collision` exits non-zero if any seed value collides across
the batch (ComfyUI would cache-skip the duplicates).

---

Maintained by **[ U N B R A N D E D ]**. MIT License.
