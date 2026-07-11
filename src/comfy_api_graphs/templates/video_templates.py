"""
Video workflow templates — quarantined.

LTX / Wan graphs change frequently across ComfyUI custom nodes and model
revisions. The stubs below used to emit outdated API JSON (e.g. LTX 2B v0.9
shaped like a generic CheckpointLoader + KSampler graph). That misled users.

Build video graphs in the ComfyUI UI, Save (API Format), then load or wrap
that JSON with ``ComfyWorkflow``. Do not treat these class names as working
recipes.
"""

from __future__ import annotations

import warnings
from typing import Any

from ..core import ComfyWorkflow

_VIDEO_TEMPLATE_MSG = (
    "{name} is not implemented in this package. "
    "Video model graphs (LTX, Wan, and similar) are version-sensitive and "
    "belong in the ComfyUI UI: design the graph, then Save (API Format) for "
    "/prompt. Use ComfyWorkflow to assemble or edit that API JSON in Python. "
    "Do not expect production video recipes from this public utility."
)


def _raise_not_implemented(name: str) -> None:
    warnings.warn(_VIDEO_TEMPLATE_MSG.format(name=name), UserWarning, stacklevel=3)
    raise NotImplementedError(_VIDEO_TEMPLATE_MSG.format(name=name))


class LtxVideoTemplate:
    """
    Quarantined stub. Former LTX text-to-video template (stale / misleading).

    Raises:
        NotImplementedError: Always. Build LTX graphs in the ComfyUI UI.
    """

    RECOMMENDED_RESOLUTIONS: list[tuple[int, int]] = []

    @staticmethod
    def create(*args: Any, **kwargs: Any) -> ComfyWorkflow:
        """Not implemented — build LTX T2V in the ComfyUI UI, export API JSON."""
        _raise_not_implemented("LtxVideoTemplate")
        raise AssertionError("unreachable")  # pragma: no cover


class LtxImg2VideoTemplate:
    """
    Quarantined stub. Former LTX image-to-video template (stale / misleading).

    Raises:
        NotImplementedError: Always. Build LTX I2V graphs in the ComfyUI UI.
    """

    @staticmethod
    def create(*args: Any, **kwargs: Any) -> ComfyWorkflow:
        """Not implemented — build LTX I2V in the ComfyUI UI, export API JSON."""
        _raise_not_implemented("LtxImg2VideoTemplate")
        raise AssertionError("unreachable")  # pragma: no cover


class WanVideoTemplate:
    """
    Quarantined stub. Former Wan video template (stale / misleading).

    Raises:
        NotImplementedError: Always. Build Wan graphs in the ComfyUI UI.
    """

    @staticmethod
    def create(*args: Any, **kwargs: Any) -> ComfyWorkflow:
        """Not implemented — build Wan video in the ComfyUI UI, export API JSON."""
        _raise_not_implemented("WanVideoTemplate")
        raise AssertionError("unreachable")  # pragma: no cover


__all__ = [
    "LtxVideoTemplate",
    "LtxImg2VideoTemplate",
    "WanVideoTemplate",
]
