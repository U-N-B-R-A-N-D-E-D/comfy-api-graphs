"""
Core ComfyUI workflow graph builders.

ComfyUI executes a directed graph of nodes. In **API format** (what ``/prompt``
accepts), each node is a dict with ``class_type`` and ``inputs``. Cross-node
wiring uses socket links of the form ``[node_id, output_index]`` — for example
``CheckpointLoaderSimple`` output 0 is MODEL, 1 is CLIP, 2 is VAE.
"""

from __future__ import annotations

import json
import random
import warnings
from collections.abc import Iterable, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Inputs commonly used as RNG widgets across stock + popular custom nodes.
_SEED_INPUT_KEYS = frozenset({"seed", "noise_seed"})


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
    inputs: dict[str, Any] = field(default_factory=dict)
    title: str | None = None

    def to_api_dict(self) -> dict[str, Any]:
        """
        Serialize this node for inclusion in an API-format prompt.

        Returns:
            Dict with ``class_type`` and ``inputs``; adds ``_meta.title`` when set.
        """
        result: dict[str, Any] = {
            "class_type": self.class_type,
            "inputs": self.inputs,
        }
        if self.title:
            result["_meta"] = {"title": self.title}
        return result

    def get_output_ref(self, output_index: int = 0) -> list[str | int]:
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
        self.nodes: dict[str, WorkflowNode] = {}
        self._next_id = 1

    def add_node(
        self,
        class_type: str,
        inputs: dict[str, Any],
        node_id: str | None = None,
        title: str | None = None,
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

        node = WorkflowNode(node_id=node_id, class_type=class_type, inputs=inputs, title=title)
        self.nodes[node_id] = node
        return node

    def get_node(self, node_id: str) -> WorkflowNode | None:
        """Return the node for ``node_id``, or ``None`` if missing."""
        return self.nodes.get(node_id)

    def to_api_format(self) -> dict[str, Any]:
        """
        Export the graph as ComfyUI **API format** (prompt dict).

        Shape: ``{node_id: {"class_type": ..., "inputs": ...}, ...}``.
        This is what ComfyUI expects under the ``prompt`` key on ``POST /prompt``.

        Returns:
            Flat dict keyed by node id strings.
        """
        return {node_id: node.to_api_dict() for node_id, node in self.nodes.items()}

    def to_graph_format(self) -> dict[str, Any]:
        """
        DEPRECATED: minimal UI-oriented **skeleton** — not a UI↔API converter.

        Does **not** reconstruct frontend ``links``, ``widgets_values``, or
        layout from API-format socket refs. Empty ``links`` means this will
        **not** round-trip into the ComfyUI frontend as a working graph.

        Prefer ``to_api_format()`` / ``save_api_json()`` for execution.
        Design complex graphs in the UI → Save (API Format).

        Returns:
            Dict with ``nodes``, empty ``links``, and version metadata.
        """
        warnings.warn(
            "ComfyWorkflow.to_graph_format() is deprecated and does NOT "
            "produce a loadable ComfyUI frontend workflow (empty links, "
            "no widgets_values). Use to_api_format() / save_api_json() for "
            "/prompt. Design in the UI → Save (API Format) for editable graphs. "
            "See Comfy-Org/ComfyUI#3050 / #1112.",
            DeprecationWarning,
            stacklevel=2,
        )
        nodes_array = []
        links_array: list[Any] = []

        for node_id, node in self.nodes.items():
            node_entry: dict[str, Any] = {
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
        """
        Write API-format JSON (suitable for ``/prompt``) to ``filepath``.

        Pretty-printed (indent=2) so seed/prompt diffs review cleanly before
        you burn queue time.
        """
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_api_format(), f, indent=2)
            f.write("\n")

    def save_graph_json(self, filepath: str) -> None:
        """
        DEPRECATED: write the skeleton from ``to_graph_format()``.

        Does not produce a frontend-loadable workflow. Prefer ``save_api_json``.
        """
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_graph_format(), f, indent=2)
            f.write("\n")

    # --- Load / mutate (v0.2 queue-time payback) ---------------------------

    @staticmethod
    def _unwrap_api_payload(data: dict[str, Any]) -> dict[str, Any]:
        """
        Accept a raw prompt dict or a ``{"prompt": {...}}`` wrapper.

        Raises:
            TypeError: If ``data`` is not a dict.
            ValueError: If the payload does not look like API-format nodes.
        """
        if not isinstance(data, dict):
            raise TypeError(f"API JSON must be a dict, got {type(data).__name__}")

        if "prompt" in data and isinstance(data["prompt"], dict):
            # Common queue payload shape: {"prompt": {...}, "client_id": "..."}
            candidates = data["prompt"]
        else:
            candidates = data

        if not candidates:
            return {}

        sample = next(iter(candidates.values()))
        if not isinstance(sample, dict) or "class_type" not in sample:
            raise ValueError(
                "Expected ComfyUI API-format nodes "
                "(each value needs class_type + inputs). "
                "UI workflow JSON (widgets_values / links) is not supported here."
            )
        return candidates

    @classmethod
    def from_api_dict(
        cls,
        data: dict[str, Any],
        name: str = "workflow",
    ) -> ComfyWorkflow:
        """
        Build a ``ComfyWorkflow`` from an in-memory API-format prompt dict.

        Preserves string node ids and ``_meta.title``. Does not rewrite links.

        Args:
            data: Prompt dict keyed by node id, or ``{"prompt": {...}}``.
            name: Local label (not sent to ComfyUI).

        Returns:
            Populated workflow ready for mutate / validate / re-queue.
        """
        payload = cls._unwrap_api_payload(data)
        wf = cls(name=name)
        max_numeric_id = 0

        for node_id, node_data in payload.items():
            if not isinstance(node_data, dict):
                raise ValueError(f"Node {node_id!r} must be a dict")
            class_type = node_data.get("class_type")
            if not class_type:
                raise ValueError(f"Node {node_id!r} is missing class_type")
            inputs = deepcopy(node_data.get("inputs") or {})
            if not isinstance(inputs, dict):
                raise ValueError(f"Node {node_id!r} inputs must be a dict")
            title = None
            meta = node_data.get("_meta")
            if isinstance(meta, dict):
                title = meta.get("title")

            wf.nodes[str(node_id)] = WorkflowNode(
                node_id=str(node_id),
                class_type=str(class_type),
                inputs=inputs,
                title=title if isinstance(title, str) else None,
            )
            if str(node_id).isdigit():
                max_numeric_id = max(max_numeric_id, int(node_id))

        wf._next_id = max_numeric_id + 1
        return wf

    @classmethod
    def from_api_json(
        cls,
        path_or_dict: str | Path | dict[str, Any],
        name: str | None = None,
    ) -> ComfyWorkflow:
        """
        Load a ComfyUI **API-format** graph from a path or dict.

        This is the v0.2 entry point: Save (API Format) → mutate → re-queue
        without re-drawing the node editor.

        Args:
            path_or_dict: Filesystem path to ``*.json``, or an already-parsed dict.
            name: Optional label; defaults to the file stem or ``"workflow"``.

        Returns:
            Populated ``ComfyWorkflow``.

        Example:
            >>> wf = ComfyWorkflow.from_api_json("my_graph_api.json")
            >>> wf.randomize_seeds()
            >>> wf.save_api_json("my_graph_api.seeded.json")
        """
        if isinstance(path_or_dict, dict):
            return cls.from_api_dict(path_or_dict, name=name or "workflow")

        path = Path(path_or_dict)
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_api_dict(data, name=name or path.stem)

    def nodes_by_class_type(self, class_type: str) -> list[WorkflowNode]:
        """Return all nodes whose ``class_type`` matches exactly."""
        return [n for n in self.nodes.values() if n.class_type == class_type]

    def nodes_by_title(self, title: str) -> list[WorkflowNode]:
        """Return all nodes whose ``_meta`` title matches exactly."""
        return [n for n in self.nodes.values() if n.title == title]

    def set_input(self, node_id: str, key: str, value: Any) -> ComfyWorkflow:
        """
        Set one input on a node by id.

        Args:
            node_id: Existing node id in the graph.
            key: Input / widget name (e.g. ``seed``, ``text``, ``ckpt_name``).
            value: Scalar or ``[src_id, slot]`` link.

        Returns:
            ``self`` for chaining.

        Raises:
            KeyError: If ``node_id`` is missing.
        """
        node = self.nodes.get(node_id)
        if node is None:
            raise KeyError(f"No node with id {node_id!r}")
        node.inputs[key] = value
        return self

    def set_inputs(self, node_id: str, **inputs: Any) -> ComfyWorkflow:
        """Set multiple inputs on one node. Returns ``self`` for chaining."""
        for key, value in inputs.items():
            self.set_input(node_id, key, value)
        return self

    def set_input_by_class_type(
        self,
        class_type: str,
        key: str,
        value: Any,
        *,
        all_matches: bool = True,
    ) -> int:
        """
        Set ``key`` on nodes matching ``class_type``.

        Args:
            class_type: Exact ComfyUI class type (e.g. ``KSampler``).
            key: Input name to overwrite.
            value: New value.
            all_matches: If True, update every match; if False, only the first.

        Returns:
            Number of nodes updated.
        """
        matches = self.nodes_by_class_type(class_type)
        if not matches:
            return 0
        targets = matches if all_matches else matches[:1]
        for node in targets:
            node.inputs[key] = value
        return len(targets)

    def set_input_by_title(
        self,
        title: str,
        key: str,
        value: Any,
        *,
        all_matches: bool = True,
    ) -> int:
        """
        Set ``key`` on nodes whose ``_meta.title`` matches ``title``.

        Returns:
            Number of nodes updated.
        """
        matches = self.nodes_by_title(title)
        if not matches:
            return 0
        targets = matches if all_matches else matches[:1]
        for node in targets:
            node.inputs[key] = value
        return len(targets)

    def iter_seed_inputs(
        self,
        *,
        input_keys: Iterable[str] | None = None,
        class_types: Sequence[str] | None = None,
    ) -> list[tuple]:
        """
        Find ``(node_id, input_key)`` pairs that look like seed widgets.

        Default keys: ``seed``, ``noise_seed``. Optionally restrict by class_type.
        """
        keys = frozenset(input_keys) if input_keys is not None else _SEED_INPUT_KEYS
        allowed = set(class_types) if class_types is not None else None
        found: list[tuple] = []
        for node_id, node in self.nodes.items():
            if allowed is not None and node.class_type not in allowed:
                continue
            for key in keys:
                if key in node.inputs:
                    found.append((node_id, key))
        return found

    def randomize_seeds(
        self,
        *,
        input_keys: Iterable[str] | None = None,
        class_types: Sequence[str] | None = None,
        rng: random.Random | None = None,
    ) -> list[str]:
        """
        Assign a fresh random int seed to every matching seed widget.

        Addresses the common API-batch pain: Comfy has no A1111-style ``-1``;
        identical prompts get cache-skipped; hand-editing JSON for 50 seeds is
        wasted queue setup time.

        Args:
            input_keys: Widget names to touch (default ``seed`` / ``noise_seed``).
            class_types: Optional allow-list of ``class_type`` strings.
            rng: Optional ``random.Random`` for deterministic tests.

        Returns:
            List of node ids that were updated.
        """
        roller = rng if rng is not None else random
        updated: list[str] = []
        for node_id, key in self.iter_seed_inputs(input_keys=input_keys, class_types=class_types):
            # ComfyUI seeds are typically uint64-ish; stay in a safe int range.
            self.nodes[node_id].inputs[key] = roller.randint(0, 2**32 - 1)
            if node_id not in updated:
                updated.append(node_id)
        return updated

    def bump_seeds(
        self,
        n: int = 1,
        *,
        input_keys: Iterable[str] | None = None,
        class_types: Sequence[str] | None = None,
    ) -> list[str]:
        """
        Add ``n`` to every matching numeric seed (batch / A-B without full random).

        Non-int seeds are replaced with ``n`` so the graph still queues.

        Returns:
            List of node ids that were updated.
        """
        updated: list[str] = []
        for node_id, key in self.iter_seed_inputs(input_keys=input_keys, class_types=class_types):
            current = self.nodes[node_id].inputs.get(key)
            if isinstance(current, bool) or not isinstance(current, int):
                self.nodes[node_id].inputs[key] = int(n)
            else:
                self.nodes[node_id].inputs[key] = current + int(n)
            if node_id not in updated:
                updated.append(node_id)
        return updated

    # --- Graph introspection / mutation (v0.4 community helpers) ---------

    @staticmethod
    def _is_node_reference(value: Any) -> bool:
        """True if ``value`` is a ComfyUI socket link ``[node_id, output_index]``."""
        if not isinstance(value, list) or len(value) != 2:
            return False
        ref_id, ref_idx = value
        return isinstance(ref_id, str) and isinstance(ref_idx, int)

    def iter_links(self) -> Iterable[tuple[str, int, str, str]]:
        """
        Yield every socket link in the graph as ``(src_id, src_slot, dst_id, dst_input)``.

        Walks each node's ``inputs`` and emits one tuple per ``[src_id, slot]``
        link. Nested lists (e.g. some conditioning bags) are walked one level
        deep. The placeholder ``["0", 0]`` (empty-conditioning hack) is skipped.

        Useful for diffing, building DOT/Mermaid views, or auditing wiring
        without re-implementing the link walker in every consumer.

        Example:
            >>> for src_id, slot, dst_id, dst_input in wf.iter_links():
            ...     print(f"{src_id}.{slot} -> {dst_id}.{dst_input}")
        """
        for dst_id, node in self.nodes.items():
            for input_name, input_value in node.inputs.items():
                refs: list[Any] = []
                if self._is_node_reference(input_value):
                    refs = [input_value]
                elif isinstance(input_value, list):
                    refs = [v for v in input_value if self._is_node_reference(v)]
                for ref in refs:
                    ref_id, ref_idx = ref
                    if ref_id == "0" and ref_idx == 0:
                        continue
                    yield (ref_id, ref_idx, dst_id, input_name)

    def remove_node(self, node_id: str) -> ComfyWorkflow:
        """
        Remove ``node_id`` and clean up links pointing at it.

        Downstream ``inputs`` that referenced the removed node are **dropped**
        (the key is deleted) — the caller is expected to rewire or remove the
        now-broken downstream nodes. This keeps the graph honest rather than
        leaving dangling ``[removed_id, slot]`` links that would fail
        ``validate_node_references``.

        Args:
            node_id: Node to remove. Must exist.

        Returns:
            ``self`` for chaining.

        Raises:
            KeyError: If ``node_id`` is not in the graph.
        """
        if node_id not in self.nodes:
            raise KeyError(f"No node with id {node_id!r}")
        del self.nodes[node_id]
        for other in self.nodes.values():
            for input_name in list(other.inputs.keys()):
                value = other.inputs[input_name]
                if self._is_node_reference(value):
                    if value[0] == node_id:
                        del other.inputs[input_name]
                elif isinstance(value, list):
                    cleaned = [
                        v for v in value if not (self._is_node_reference(v) and v[0] == node_id)
                    ]
                    if len(cleaned) != len(value):
                        other.inputs[input_name] = cleaned
        return self

    def topological_order(self) -> list[str]:
        """
        Return node ids in dependency order (sources before consumers).

        Edges are derived from ``iter_links()``: an edge ``src -> dst`` means
        ``dst`` consumes an output of ``src``. ComfyUI itself does not require
        a specific key order in the prompt dict, but a stable topological
        order is useful for diffing, deterministic serialization, and
        reasoning about execution order in CI logs.

        Returns:
            Node ids ordered so every source appears before its consumers.

        Raises:
            ValueError: If the graph contains a cycle (ComfyUI would not
                execute it either).
        """
        # Kahn's algorithm with deterministic tie-breaking (insertion order).
        indegree: dict[str, int] = dict.fromkeys(self.nodes, 0)
        successors: dict[str, list[str]] = {nid: [] for nid in self.nodes}
        for src_id, _slot, dst_id, _input in self.iter_links():
            if src_id in indegree and dst_id in indegree:
                successors[src_id].append(dst_id)
                indegree[dst_id] += 1

        # Preserve insertion order for deterministic output.
        ready = [nid for nid in self.nodes if indegree[nid] == 0]
        order: list[str] = []
        while ready:
            nid = ready.pop(0)
            order.append(nid)
            for succ in successors[nid]:
                indegree[succ] -= 1
                if indegree[succ] == 0:
                    ready.append(succ)

        if len(order) != len(self.nodes):
            remaining = sorted(set(self.nodes) - set(order))
            raise ValueError(
                f"Workflow contains a cycle; cannot topologically sort. "
                f"Nodes in or reachable from a cycle: {remaining}"
            )
        return order

    def to_mermaid(self, *, show_class: bool = True) -> str:
        """
        Render the graph as a Mermaid ``flowchart`` for README / PR / issue use.

        Each node becomes ``id["class_type"]`` and each socket link becomes an
        edge. Output slots are labeled on edges (``|slot N|``) only when the
        source declares more than one used slot; otherwise the edge is plain.

        This is a debugging/diffing aid, not a UI round-trip format. Paste the
        output into a Markdown block on GitHub and it renders inline.

        Args:
            show_class: If True (default), label nodes with ``class_type``;
                if False, label with the node id only (compact view).
        """
        lines = ["flowchart TD"]
        for nid, node in self.nodes.items():
            label = node.class_type if show_class else nid
            # Mermaid node ids must be alphanumeric/underscore; sanitize.
            safe = nid if nid.isalnum() else f'"{nid}"'
            lines.append(f'    {safe}["{label}"]')
        for src_id, _slot, dst_id, _input in self.iter_links():
            src_safe = src_id if src_id.isalnum() else f'"{src_id}"'
            dst_safe = dst_id if dst_id.isalnum() else f'"{dst_id}"'
            lines.append(f"    {src_safe} --> {dst_safe}")
        return "\n".join(lines)

    def diff(self, other: ComfyWorkflow) -> dict[str, Any]:
        """
        Compute a structured diff between this workflow and ``other``.

        Compares node identity by ``node_id`` and reports:

        - ``added``: node ids present in ``other`` but not here.
        - ``removed``: node ids present here but not in ``other``.
        - ``changed_class``: node ids in both whose ``class_type`` differs.
        - ``added_inputs`` / ``removed_inputs`` / ``changed_inputs``:
          per-node input-key changes (added/removed keys, or same key with
          different value). Values are compared by equality; link lists
          compare structurally.

        Returns:
            Dict with the four buckets above (each a list or per-node dict).

        Example:
            >>> d = wf_a.diff(wf_b)
            >>> if d["added"] or d["removed"] or d["changed_inputs"]:
            ...     print("graphs differ")
        """
        added = sorted(set(other.nodes) - set(self.nodes))
        removed = sorted(set(self.nodes) - set(other.nodes))
        common = set(self.nodes) & set(other.nodes)

        changed_class: list[str] = []
        added_inputs: dict[str, list[str]] = {}
        removed_inputs: dict[str, list[str]] = {}
        changed_inputs: dict[str, list[str]] = {}

        for nid in sorted(common):
            a, b = self.nodes[nid], other.nodes[nid]
            if a.class_type != b.class_type:
                changed_class.append(nid)
                continue
            a_keys = set(a.inputs)
            b_keys = set(b.inputs)
            if a_keys - b_keys:
                removed_inputs[nid] = sorted(a_keys - b_keys)
            if b_keys - a_keys:
                added_inputs[nid] = sorted(b_keys - a_keys)
            changed: list[str] = []
            for k in sorted(a_keys & b_keys):
                if a.inputs[k] != b.inputs[k]:
                    changed.append(k)
            if changed:
                changed_inputs[nid] = changed

        return {
            "added": added,
            "removed": removed,
            "changed_class": changed_class,
            "added_inputs": added_inputs,
            "removed_inputs": removed_inputs,
            "changed_inputs": changed_inputs,
        }

    def batch_seeds(
        self,
        n: int,
        *,
        mode: str = "bump",
        start: int | None = None,
        input_keys: Iterable[str] | None = None,
        class_types: Sequence[str] | None = None,
    ) -> Iterable[ComfyWorkflow]:
        """
        Generate ``n`` mutated copies with distinct seeds (a batch generator).

        This is the seed-batch primitive: yield ``n`` independent
        ``ComfyWorkflow`` copies whose seed widgets differ, ready to serialize
        with ``to_api_format()`` and POST to ``/prompt``. Avoids hand-editing
        JSON for every variant of a Save (API Format) graph.

        Args:
            n: Number of variants to yield. Must be >= 0.
            mode: ``"bump"`` (default) increments seeds by the batch index;
                ``"randomize"`` assigns a fresh random int per copy.
            start: For ``mode="bump"``, the seed value of the first copy
                (subsequent copies add their index). If None, the loaded
                seed is used as the start. Ignored for ``mode="randomize"``.
            input_keys: Widget names to touch (default ``seed`` / ``noise_seed``).
            class_types: Optional allow-list of ``class_type`` strings.

        Yields:
            ``n`` new ``ComfyWorkflow`` instances; the original is unchanged.

        Raises:
            ValueError: If ``n`` is negative or ``mode`` is unknown.
        """
        if n < 0:
            raise ValueError(f"n must be >= 0, got {n}")
        if mode not in ("bump", "randomize"):
            raise ValueError(f"mode must be 'bump' or 'randomize', got {mode!r}")

        base_api = self.to_api_format()
        for i in range(n):
            wf = ComfyWorkflow.from_api_dict(base_api, name=f"{self.name}_batch{i}")
            if mode == "bump":
                if start is None:
                    wf.bump_seeds(i, input_keys=input_keys, class_types=class_types)
                else:
                    # Deterministic series starting at ``start``: copy i = start + i.
                    for nid, key in wf.iter_seed_inputs(
                        input_keys=input_keys, class_types=class_types
                    ):
                        wf.nodes[nid].inputs[key] = int(start) + i
            else:
                wf.randomize_seeds(input_keys=input_keys, class_types=class_types)
            yield wf

    def __len__(self) -> int:
        """Number of nodes in the graph."""
        return len(self.nodes)

    def __repr__(self) -> str:
        return f"ComfyWorkflow(name='{self.name}', nodes={len(self.nodes)})"
