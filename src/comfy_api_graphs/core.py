"""
Core ComfyUI workflow graph builders.

ComfyUI executes a directed graph of nodes. In **API format** (what ``/prompt``
accepts), each node is a dict with ``class_type`` and ``inputs``. Cross-node
wiring uses socket links of the form ``[node_id, output_index]`` — for example
``CheckpointLoaderSimple`` output 0 is MODEL, 1 is CLIP, 2 is VAE.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field


@dataclass
class WorkflowNode:
    """
    One node in a ComfyUI API-format graph.

    Mirrors a single entry under the prompt dict keyed by ``node_id``:
    ``{"class_type": "...", "inputs": {...}, "_meta": {"title": "..."}}``.

    Socket links from this node into another node's ``inputs`` are produced
    with ``get_output_ref(output_index)`` → ``[node_id, output_index]``.

    Attributes:
        node_id: String id in the prompt graph (e.g. ``"1"``, ``"10"``).
        class_type: Registered ComfyUI node type (e.g. ``KSampler``,
            ``CheckpointLoaderSimple``, ``CLIPTextEncode``, ``EmptyLatentImage``).
        inputs: Widget values and linked inputs (primitives or ``[id, slot]``).
        title: Optional UI title stored under ``_meta`` (ignored by the backend).

    Example:
        >>> node = WorkflowNode(
        ...     node_id="5",
        ...     class_type="CheckpointLoaderSimple",
        ...     inputs={"ckpt_name": "model.safetensors"},
        ...     title="Load Checkpoint",
        ... )
        >>> node.to_api_dict()["class_type"]
        'CheckpointLoaderSimple'
        >>> node.get_output_ref(1)  # CLIP socket
        ['5', 1]
    """

    node_id: str
    class_type: str
    inputs: Dict[str, Any] = field(default_factory=dict)
    title: Optional[str] = None

    def to_api_dict(self) -> Dict[str, Any]:
        """
        Serialize this node for inclusion in an API-format prompt.

        Returns:
            Dict with ``class_type`` and ``inputs``; adds ``_meta.title`` when set.
        """
        result: Dict[str, Any] = {
            "class_type": self.class_type,
            "inputs": self.inputs,
        }
        if self.title:
            result["_meta"] = {"title": self.title}
        return result

    def get_output_ref(self, output_index: int = 0) -> List[Union[str, int]]:
        """
        Build a socket link to one of this node's outputs.

        ComfyUI API format wires graphs by embedding
        ``[source_node_id, output_index]`` inside a downstream node's inputs
        (conditioning, MODEL, CLIP, VAE, LATENT, IMAGE, etc.).

        Args:
            output_index: Output slot index (0 = first output). For
                ``CheckpointLoaderSimple``: 0=MODEL, 1=CLIP, 2=VAE.

        Returns:
            Link list ``[node_id, output_index]``.

        Example:
            >>> loader = WorkflowNode("1", "CheckpointLoaderSimple", {})
            >>> clip_ref = loader.get_output_ref(1)
            >>> # later: CLIPTextEncode inputs={"text": "...", "clip": clip_ref}
        """
        return [self.node_id, output_index]


class ComfyWorkflow:
    """
    Mutable container for a ComfyUI API-format workflow graph.

    Owns a map of ``node_id`` → ``WorkflowNode``. Typical pipeline shape:

    - Loaders (``CheckpointLoaderSimple``, ``UNETLoader``, ``VAELoader``, …)
    - Conditioning (``CLIPTextEncode`` → CONDITIONING)
    - Latent (``EmptyLatentImage`` / encode path → LATENT)
    - Sampler (``KSampler`` / ``KSamplerAdvanced``)
    - Decode + save (``VAEDecode`` → ``SaveImage`` / ``PreviewImage``)

    Call ``to_api_format()`` and POST ``{"prompt": ...}`` to ``/prompt``.

    Attributes:
        name: Label for debugging only (not sent to ComfyUI).
        nodes: ``node_id`` → ``WorkflowNode``.

    Example:
        >>> wf = ComfyWorkflow("txt2img")
        >>> ckpt = wf.add_node(
        ...     "CheckpointLoaderSimple",
        ...     {"ckpt_name": "model.safetensors"},
        ... )
        >>> sampler = wf.add_node(
        ...     "KSampler",
        ...     {"model": ckpt.get_output_ref(0), "seed": 0},
        ... )
        >>> prompt = wf.to_api_format()
    """

    def __init__(self, name: str = "workflow"):
        """
        Create an empty graph.

        Args:
            name: Local identifier; not part of API JSON.
        """
        self.name = name
        self.nodes: Dict[str, WorkflowNode] = {}
        self._next_id = 1

    def add_node(
        self,
        class_type: str,
        inputs: Dict[str, Any],
        node_id: Optional[str] = None,
        title: Optional[str] = None,
    ) -> WorkflowNode:
        """
        Append a node to the graph.

        Args:
            class_type: ComfyUI ``class_type`` string.
            inputs: Input dict — scalars for widgets, ``[node_id, slot]`` for links.
            node_id: Optional fixed id; otherwise auto-incremented ``"1"``, ``"2"``, …
            title: Optional display title (``_meta`` only).

        Returns:
            The new ``WorkflowNode`` (use ``get_output_ref`` to wire it further).
        """
        if node_id is None:
            node_id = str(self._next_id)
            self._next_id += 1

        node = WorkflowNode(
            node_id=node_id, class_type=class_type, inputs=inputs, title=title
        )
        self.nodes[node_id] = node
        return node

    def get_node(self, node_id: str) -> Optional[WorkflowNode]:
        """Return the node for ``node_id``, or ``None`` if missing."""
        return self.nodes.get(node_id)

    def to_api_format(self) -> Dict[str, Any]:
        """
        Export the graph as ComfyUI **API format** (prompt dict).

        Shape: ``{node_id: {"class_type": ..., "inputs": ...}, ...}``.
        This is what ComfyUI expects under the ``prompt`` key on ``POST /prompt``.

        Returns:
            Flat dict keyed by node id strings.
        """
        return {
            node_id: node.to_api_dict() for node_id, node in self.nodes.items()
        }

    def to_graph_format(self) -> Dict[str, Any]:
        """
        Export a **minimal** UI-oriented graph skeleton.

        Honesty note: this does **not** reconstruct full frontend ``links``
        arrays or layout from API-format socket refs. Nodes are listed with
        default ``pos``/``size`` and an empty ``links`` list. Prefer
        ``to_api_format()`` for execution; use Save (API Format) / Save from
        the ComfyUI UI when you need a complete editable graph.

        Returns:
            Dict with ``nodes``, empty ``links``, and version metadata.
        """
        nodes_array = []
        links_array: List[Any] = []

        for node_id, node in self.nodes.items():
            node_entry: Dict[str, Any] = {
                "id": int(node_id) if node_id.isdigit() else hash(node_id) & 0xFFFFFF,
                "type": node.class_type,
                "pos": [100, 100],
                "size": [200, 100],
                "flags": {},
                "order": 0,
                "mode": 0,
                "inputs": [],
                "outputs": [],
                "widgets_values": [],
                "properties": {"Node name for S&R": node.class_type},
            }
            if node.title:
                node_entry["title"] = node.title
            nodes_array.append(node_entry)

        return {
            "last_node_id": len(nodes_array),
            "last_link_id": len(links_array),
            "nodes": nodes_array,
            "links": links_array,
            "groups": [],
            "version": 0.4,
        }

    def save_api_json(self, filepath: str) -> None:
        """Write API-format JSON (suitable for ``/prompt``) to ``filepath``."""
        with open(filepath, "w") as f:
            json.dump(self.to_api_format(), f, indent=2)

    def save_graph_json(self, filepath: str) -> None:
        """Write the minimal graph skeleton from ``to_graph_format()``."""
        with open(filepath, "w") as f:
            json.dump(self.to_graph_format(), f, indent=2)

    def __len__(self) -> int:
        """Number of nodes in the graph."""
        return len(self.nodes)

    def __repr__(self) -> str:
        return f"ComfyWorkflow(name='{self.name}', nodes={len(self.nodes)})"
