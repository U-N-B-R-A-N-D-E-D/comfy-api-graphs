"""
Validation helpers for ComfyUI API-format workflow graphs.

Checks socket links (``[node_id, output_index]``), optional ``/object_info``
schema checks, a rough VRAM *planning hint* (not a predictor), and a simple
completeness pass (output nodes, model loaders).
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .core import ComfyWorkflow

# Planning-hint disclaimer — returned with every estimate_vram_usage result.
_VRAM_DISCLAIMER = (
    "PLANNING HINT ONLY — not a predictor. Resolution, batch size, "
    "attention backend, quantization (GGUF/NVFP4/FP8), dual-GPU, and custom "
    "nodes dominate real VRAM. Do not size hardware or blame OOM on this number."
)


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


def load_object_info(
    path_or_dict: Union[str, Path, Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Load a ComfyUI ``/object_info`` payload from a path or in-memory dict.

    Prefer a cached snapshot for CI (large live payloads can be multi‑MB).

    Args:
        path_or_dict: JSON file path, or an already-parsed object_info dict.

    Returns:
        Dict keyed by ``class_type`` → node schema.

    Raises:
        TypeError: If the payload is not a dict.
        ValueError: If the dict does not look like object_info.
    """
    if isinstance(path_or_dict, dict):
        data = path_or_dict
    else:
        path = Path(path_or_dict)
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

    if not isinstance(data, dict):
        raise TypeError(
            f"object_info must be a dict, got {type(data).__name__}"
        )
    if not data:
        return {}

    sample_key, sample = next(iter(data.items()))
    if not isinstance(sample, dict) or "input" not in sample:
        raise ValueError(
            "Expected ComfyUI /object_info shape "
            f"(class_type → {{'input': ...}}); sample key {sample_key!r} "
            "is missing 'input'."
        )
    return data


def fetch_object_info(
    url: str = "http://127.0.0.1:8188/object_info",
    *,
    timeout: float = 60.0,
) -> Dict[str, Any]:
    """
    Fetch ``/object_info`` from a running ComfyUI (stdlib ``urllib`` only).

    Still zero runtime dependencies. For CI, prefer ``load_object_info(path)``
    with a checked-in snapshot so you do not hammer a live GPU box.

    Args:
        url: Full object_info URL (default localhost:8188).
        timeout: Socket timeout in seconds.

    Returns:
        Parsed object_info dict.

    Raises:
        urllib.error.URLError: On network / HTTP failure.
        ValueError: If the response is not valid object_info.
    """
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8")
    data = json.loads(raw)
    return load_object_info(data)


def _input_schema_names(node_schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Flatten required + optional input names from one object_info node entry.

    Returns:
        Dict with ``required`` (set), ``optional`` (set), ``all`` (set).
    """
    input_block = node_schema.get("input") or {}
    if not isinstance(input_block, dict):
        return {"required": set(), "optional": set(), "all": set()}

    required = input_block.get("required") or {}
    optional = input_block.get("optional") or {}
    req_names = set(required.keys()) if isinstance(required, dict) else set()
    opt_names = set(optional.keys()) if isinstance(optional, dict) else set()
    return {
        "required": req_names,
        "optional": opt_names,
        "all": req_names | opt_names,
    }


def validate_against_object_info(
    workflow: ComfyWorkflow,
    object_info: Dict[str, Any],
    *,
    unknown_inputs_as_errors: bool = False,
) -> Dict[str, Any]:
    """
    Schema-check an API-format graph against a ComfyUI ``/object_info`` snapshot.

    Catches the failure class that link-only validation misses: missing /
    renamed ``class_type``, unknown widget names, missing required inputs,
    and output-slot indexes past what the source node declares.

    Args:
        workflow: Graph to validate.
        object_info: Payload from ``load_object_info`` / ``fetch_object_info``.
        unknown_inputs_as_errors: If True, unknown input keys are errors;
            default False keeps them as warnings (dynamic custom nodes exist).

    Returns:
        Dict with ``valid``, ``errors``, ``warnings``, ``info``, ``node_count``.

    Example:
        >>> info = load_object_info("object_info_snapshot.json")
        >>> report = validate_against_object_info(wf, info)
        >>> assert report["valid"]
    """
    errors: List[str] = []
    warnings: List[str] = []
    info: List[str] = []

    errors.extend(validate_node_references(workflow))

    if not object_info:
        errors.append("object_info is empty; cannot schema-check class_types")
        return {
            "valid": False,
            "errors": errors,
            "warnings": warnings,
            "info": info,
            "node_count": len(workflow.nodes),
        }

    known_types = set(object_info.keys())
    info.append(f"object_info contains {len(known_types)} class_types")

    def is_node_reference(value: Any) -> bool:
        if not isinstance(value, list) or len(value) != 2:
            return False
        ref_id, ref_idx = value
        return isinstance(ref_id, str) and isinstance(ref_idx, int)

    for node_id, node in workflow.nodes.items():
        class_type = node.class_type
        if class_type not in object_info:
            errors.append(
                f"Node {node_id}: unknown class_type {class_type!r} "
                "(not in object_info — missing custom node or renamed type)"
            )
            continue

        schema = object_info[class_type]
        names = _input_schema_names(schema)
        present = set(node.inputs.keys())

        for missing in sorted(names["required"] - present):
            errors.append(
                f"Node {node_id} ({class_type}): missing required input "
                f"{missing!r}"
            )

        for unknown in sorted(present - names["all"]):
            msg = (
                f"Node {node_id} ({class_type}): unknown input {unknown!r} "
                "(not in object_info required/optional)"
            )
            if unknown_inputs_as_errors:
                errors.append(msg)
            else:
                warnings.append(msg)

        # Output-slot bounds when the source node is known
        output_len: Optional[int] = None
        outputs = schema.get("output")
        if isinstance(outputs, list):
            output_len = len(outputs)

        for input_name, input_value in node.inputs.items():
            refs: List[Any] = []
            if is_node_reference(input_value):
                refs = [input_value]
            elif isinstance(input_value, list):
                refs = [item for item in input_value if is_node_reference(item)]

            for ref_id, ref_idx in refs:
                if ref_id == "0" and ref_idx == 0:
                    continue
                src = workflow.nodes.get(ref_id)
                if src is None:
                    continue  # already reported by validate_node_references
                src_schema = object_info.get(src.class_type)
                if not isinstance(src_schema, dict):
                    continue
                src_outputs = src_schema.get("output")
                if not isinstance(src_outputs, list):
                    continue
                if ref_idx < 0 or ref_idx >= len(src_outputs):
                    errors.append(
                        f"Node {node_id} input {input_name!r}: "
                        f"output index {ref_idx} out of range for "
                        f"{src.class_type!r} "
                        f"(declares {len(src_outputs)} outputs)"
                    )

        if output_len is not None:
            info.append(
                f"Node {node_id} ({class_type}): schema OK "
                f"({len(names['required'])} required, "
                f"{len(names['optional'])} optional, "
                f"{output_len} outputs)"
            )

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "info": info,
        "node_count": len(workflow.nodes),
    }


def estimate_vram_usage(workflow: ComfyWorkflow) -> Dict[str, Any]:
    """
    Rough VRAM **planning hint** from loader ``class_type`` nodes.

    This is **not** a predictor. GGUF / NVFP4 / FP8, resolution, batch,
    attention backends, and dual-GPU layouts dominate real usage. The returned
    number must not be used to size production hardware or explain OOM.

    Args:
        workflow: Graph to analyze.

    Returns:
        Dict including ``estimated_vram_gb``, ``is_estimate`` (always True),
        ``disclaimer``, soft ``recommendation``, and loader details.
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

            details.append(f"Checkpoint '{ckpt_name}': ~{model_vram:.1f} GB (hint)")
            vram_gb += model_vram
            model_count += 1
        elif class_type in model_loader_types:
            model_vram = float(estimate) if isinstance(estimate, (int, float)) else 0.0
            details.append(f"{class_type}: ~{model_vram:.1f} GB (hint)")
            vram_gb += model_vram
            model_count += 1
        else:
            model_vram = float(estimate) if isinstance(estimate, (int, float)) else 0.0
            details.append(f"{class_type}: ~{model_vram:.1f} GB (hint)")
            vram_gb += model_vram

    generation_overhead = 2.0
    total_estimate = vram_gb + generation_overhead

    if total_estimate > 20:
        recommendation = (
            "Hint only: this class_type mix often needs 24GB-class VRAM — "
            "measure on your box; ignore this if you use quantized loaders."
        )
    elif total_estimate > 12:
        recommendation = (
            "Hint only: this class_type mix often needs 16GB-class VRAM — "
            "not a guarantee."
        )
    elif total_estimate > 8:
        recommendation = (
            "Hint only: this class_type mix often needs 12GB-class VRAM — "
            "not a guarantee."
        )
    else:
        recommendation = (
            "Hint only: light class_type mix; still verify on your hardware."
        )

    return {
        "estimated_vram_gb": round(total_estimate, 1),
        "model_vram_gb": round(vram_gb, 1),
        "overhead_gb": generation_overhead,
        "model_loaders": model_count,
        "details": details,
        "recommendation": recommendation,
        "is_estimate": True,
        "disclaimer": _VRAM_DISCLAIMER,
    }


def validate_workflow_complete(workflow: ComfyWorkflow) -> Dict[str, Any]:
    """
    Lightweight completeness check before queueing on ``/prompt``.

    Checks broken socket links, presence of a common output node
    (``SaveImage``, ``PreviewImage``, ``SaveVideo``, ``VHS_VideoCombine``),
    and a model loader (``CheckpointLoaderSimple`` or ``UNETLoader``).

    Does **not** verify widget schemas — use ``validate_against_object_info``
    with a live or cached ``/object_info`` for that.

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

    warnings.append(
        "Link/completeness check only — run validate_against_object_info() "
        "for class_type / required-input schema checks before production queues."
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
    "load_object_info",
    "fetch_object_info",
    "validate_against_object_info",
    "estimate_vram_usage",
    "validate_workflow_complete",
]
