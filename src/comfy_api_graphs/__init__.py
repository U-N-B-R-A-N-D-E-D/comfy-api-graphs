"""
Comfy API Graphs
========================

Pythonic construction of ComfyUI workflows.
Build, validate, and export AI generation workflows as ComfyUI API JSON.

Basic usage:
    >>> from comfy_api_graphs import FluxTxt2ImgTemplate
    >>> workflow = FluxTxt2ImgTemplate.create(
    ...     prompt="a serene landscape",
    ...     width=1024,
    ...     height=1024,
    ... )
    >>> workflow.save_api_json("my_workflow.json")

Custom workflow:
    >>> from comfy_api_graphs import ComfyWorkflow
    >>> wf = ComfyWorkflow("custom")
    >>> node = wf.add_node("KSampler", {...})
    >>> api_format = wf.to_api_format()
"""

from .core import WorkflowNode, ComfyWorkflow
from .validation import (
    validate_node_references,
    estimate_vram_usage,
    validate_workflow_complete,
)
from .templates import (
    FluxTxt2ImgTemplate,
    FluxImg2ImgTemplate,
    FluxInpaintTemplate,
    SdxlTxt2ImgTemplate,
    SdxlImg2ImgTemplate,
    CharacterReferenceTemplate,
    UpscaleTemplate,
    InpaintTemplate,
    OutpaintTemplate,
)

__version__ = "0.2.0"

__all__ = [
    # Core
    "WorkflowNode",
    "ComfyWorkflow",
    # Validation
    "validate_node_references",
    "estimate_vram_usage",
    "validate_workflow_complete",
    # FLUX
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
