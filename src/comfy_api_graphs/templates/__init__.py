"""
Pre-built workflow templates for common ComfyUI graphs.

Each template is a static class with a ``create()`` method that returns a
``ComfyWorkflow`` whose ``to_api_format()`` payload is suitable for ``/prompt``.

Video templates (LTX, Wan) are quarantined stubs under ``video_templates`` —
they raise ``NotImplementedError`` and are not part of the public package API.
"""

from .flux_templates import FluxImg2ImgTemplate, FluxInpaintTemplate, FluxTxt2ImgTemplate
from .sdxl_templates import SdxlImg2ImgTemplate, SdxlTxt2ImgTemplate
from .utility_templates import (
    CharacterReferenceTemplate,
    InpaintTemplate,
    OutpaintTemplate,
    UpscaleTemplate,
)

__all__ = [
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
