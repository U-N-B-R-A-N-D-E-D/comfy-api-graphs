"""
Comfy API Graphs
================

Offline thin Python helper for ComfyUI **API-format** graphs:
load → mutate → validate → re-queue.

Primary path (what this package is for)::

    >>> from comfy_api_graphs import ComfyWorkflow
    >>> wf = ComfyWorkflow.from_api_json("my_graph_api.json")  # Save (API Format)
    >>> wf.set_input_by_class_type("CLIPTextEncode", "text", "a new prompt")
    >>> wf.randomize_seeds()
    >>> wf.save_api_json("my_graph_api.seeded.json")

Optional FLUX/SDXL templates teach common wiring — they are not the product.
"""

from .core import ComfyWorkflow, WorkflowNode
from .templates import (
    CharacterReferenceTemplate,
    FluxImg2ImgTemplate,
    FluxInpaintTemplate,
    FluxTxt2ImgTemplate,
    InpaintTemplate,
    OutpaintTemplate,
    SdxlImg2ImgTemplate,
    SdxlTxt2ImgTemplate,
    UpscaleTemplate,
)
from .validation import (
    estimate_vram_usage,
    fetch_object_info,
    find_duplicate_seeds_across_batch,
    find_orphan_nodes,
    load_object_info,
    validate_against_object_info,
    validate_node_references,
    validate_workflow_complete,
)

__version__ = "0.3.0"

__all__ = [
    # Core
    "WorkflowNode",
    "ComfyWorkflow",
    # Validation
    "validate_node_references",
    "load_object_info",
    "fetch_object_info",
    "validate_against_object_info",
    "estimate_vram_usage",
    "validate_workflow_complete",
    "find_orphan_nodes",
    "find_duplicate_seeds_across_batch",
    # FLUX (optional templates — secondary)
    "FluxTxt2ImgTemplate",
    "FluxImg2ImgTemplate",
    "FluxInpaintTemplate",
    # SDXL
    "SdxlTxt2ImgTemplate",
    "SdxlImg2ImgTemplate",
    # Utility
    "CharacterReferenceTemplate",
    "UpscaleTemplate",
    "InpaintTemplate",
    "OutpaintTemplate",
]
