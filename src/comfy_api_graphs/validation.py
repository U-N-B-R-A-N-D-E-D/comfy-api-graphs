"""
Validation helpers for ComfyUI API-format workflow graphs.

Checks socket links (``[node_id, output_index]``), rough VRAM from loaders,
and a simple completeness pass (output nodes, model loaders).
"""

from __future__ import annotations

from typing import Any, Dict, List

from .core import ComfyWorkflow


def validate_node_references(workflow: ComfyWorkflow) -> List[str]:
    """
    Validate that every ``[node_id, output_index]`` link targets an existing node.

    In API format, linked inputs are lists of length 2: string ``node_id`` and
    int output slot. Nested lists (e.g. some conditioning bags) are walked one
    level deep. Placeholder ``["0", 0]`` (common empty conditioning hack) is
    skipped.

    Args:
        workflow: Graph to inspect.

    Returns:
        Error strings; empty list means all links resolve.

    Example:
        >>> from comfy_api_graphs import ComfyWorkflow, validate_node_references
        >>> wf = ComfyWorkflow()
        >>> a = wf.add_node("CheckpointLoaderSimple", {})
        >>> wf.add_node("KSampler", {"model": a.get_output_ref()})
        >>> validate_node_references(wf)
        []
    """
    errors: List[str] = []

    def is_node_reference(value: Any) -> bool:
        if not isinstance(value, list) or len(value) != 2:
            return False
        ref_id, ref_idx = value
        return isinstance(ref_id, str) and isinstance(ref_idx, int)

    for node_id, node in workflow.nodes.items():
        for input_name, input_value in node.inputs.items():
            if is_node_reference(input_value):
                ref_id, ref_idx = input_value
                if ref_id == "0" and ref_idx == 0:
                    continue
                if ref_id not in workflow.nodes:
                    errors.append(
                        f"Node {node_id} input '{input_name}' references "
                        f"non-existent node: {ref_id}"
                    )
            elif isinstance(input_value, list):
                for item in input_value:
                    if is_node_reference(item):
                        ref_id, ref_idx = item
                        if ref_id == "0" and ref_idx == 0:
                            continue
                        if ref_id not in workflow.nodes:
                            errors.append(
                                f"Node {node_id} input '{input_name}' contains "
                                f"reference to non-existent node: {ref_id}"
                            )

    return errors


def estimate_vram_usage(workflow: ComfyWorkflow) -> Dict[str, Any]:
    """
    Rough VRAM estimate from loader ``class_type`` nodes in the graph.

    Heuristic only: resolution, batch size, attention backend, and quantization
    dominate real usage. Treat the result as a planning hint, not a guarantee.

    Args:
        workflow: Graph to analyze.

    Returns:
        Dict with ``estimated_vram_gb``, ``model_vram_gb``, ``overhead_gb``,
        ``model_loaders``, ``details``, and ``recommendation``.
    """
    vram_gb = 0.0
    model_count = 0
    details: List[str] = []

    vram_estimates: Dict[str, Any] = {
        "CheckpointLoaderSimple": {
            "flux": 22.0,
            "flux-schnell": 22.0,
            "default": 12.0,
        },
        "UNETLoader": 12.0,
        "VAELoader": 0.5,
        "CLIPLoader": 2.0,
        "CLIPLoaderGGUF": 2.0,
        "DualCLIPLoader": 4.0,
        "LoraLoader": 0.5,
        "ControlNetLoader": 2.0,
        "UpscaleModelLoader": 0.5,
        "IPAdapterModelLoader": 2.0,
    }

    model_loader_types = {
        "CheckpointLoaderSimple",
        "UNETLoader",
        "LoraLoader",
        "ControlNetLoader",
        "UpscaleModelLoader",
        "IPAdapterModelLoader",
    }

    for node in workflow.nodes.values():
        class_type = node.class_type

        if class_type not in vram_estimates:
            continue

        estimate = vram_estimates[class_type]

        if class_type == "CheckpointLoaderSimple" and isinstance(estimate, dict):
            ckpt_name = node.inputs.get("ckpt_name", "")
            ckpt_lower = str(ckpt_name).lower()

            if "flux" in ckpt_lower:
                model_vram = (
                    estimate["flux-schnell"]
                    if "schnell" in ckpt_lower
                    else estimate["flux"]
                )
            else:
                model_vram = estimate["default"]

            details.append(f"Checkpoint '{ckpt_name}': ~{model_vram:.1f} GB")
            vram_gb += model_vram
            model_count += 1
        elif class_type in model_loader_types:
            model_vram = float(estimate) if isinstance(estimate, (int, float)) else 0.0
            details.append(f"{class_type}: ~{model_vram:.1f} GB")
            vram_gb += model_vram
            model_count += 1
        else:
            model_vram = float(estimate) if isinstance(estimate, (int, float)) else 0.0
            details.append(f"{class_type}: ~{model_vram:.1f} GB")
            vram_gb += model_vram

    generation_overhead = 2.0
    total_estimate = vram_gb + generation_overhead

    if total_estimate > 20:
        recommendation = "Use load balancer or GPU with 24GB+ VRAM"
    elif total_estimate > 12:
        recommendation = "Requires GPU with 16GB+ VRAM"
    elif total_estimate > 8:
        recommendation = "Requires GPU with 12GB+ VRAM"
    else:
        recommendation = "Should run on most GPUs (8GB+)"

    return {
        "estimated_vram_gb": round(total_estimate, 1),
        "model_vram_gb": round(vram_gb, 1),
        "overhead_gb": generation_overhead,
        "model_loaders": model_count,
        "details": details,
        "recommendation": recommendation,
    }


def validate_workflow_complete(workflow: ComfyWorkflow) -> Dict[str, Any]:
    """
    Lightweight completeness check before queueing on ``/prompt``.

    Checks broken socket links, presence of a common output node
    (``SaveImage``, ``PreviewImage``, ``SaveVideo``, ``VHS_VideoCombine``),
    and a model loader (``CheckpointLoaderSimple`` or ``UNETLoader``).

    Does **not** call ComfyUI ``/object_info`` or verify widget schemas.

    Args:
        workflow: Graph to validate.

    Returns:
        Dict with ``valid`` (no hard errors), ``errors``, ``warnings``,
        ``info``, and ``node_count``.
    """
    errors: List[str] = []
    warnings: List[str] = []
    info: List[str] = []

    errors.extend(validate_node_references(workflow))

    output_nodes = {
        "SaveImage",
        "SaveVideo",
        "PreviewImage",
        "VHS_VideoCombine",
    }
    has_output = any(
        node.class_type in output_nodes for node in workflow.nodes.values()
    )
    if not has_output:
        warnings.append(
            "No output node found (SaveImage, SaveVideo, PreviewImage, "
            "VHS_VideoCombine). Graph may run without persisting results."
        )

    has_checkpoint = any(
        node.class_type in ("CheckpointLoaderSimple", "UNETLoader")
        for node in workflow.nodes.values()
    )
    if not has_checkpoint:
        warnings.append(
            "No model loader found (CheckpointLoaderSimple or UNETLoader)."
        )

    node_count = len(workflow.nodes)
    info.append(f"Workflow contains {node_count} nodes")

    seen_ids: set[str] = set()
    for node_id in workflow.nodes.keys():
        if node_id in seen_ids:
            errors.append(f"Duplicate node ID found: {node_id}")
        seen_ids.add(node_id)

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "info": info,
        "node_count": node_count,
    }


__all__ = [
    "validate_node_references",
    "estimate_vram_usage",
    "validate_workflow_complete",
]
