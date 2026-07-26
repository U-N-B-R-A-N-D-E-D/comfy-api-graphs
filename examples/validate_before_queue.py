#!/usr/bin/env python3
"""
Validate before queue
=====================

The highest-value pattern for CI: load a Save (API Format) graph, run link
checks, then schema-check it against a cached ``/object_info`` snapshot, and
fail loudly *before* anything reaches a GPU queue.

This is the same logic the ``python -m comfy_api_graphs validate`` CLI runs,
shown here as a library call so you can wire it into your own scripts.

Usage:
    python examples/validate_before_queue.py path/to/graph_api.json path/to/object_info.json
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from comfy_api_graphs import (
    ComfyWorkflow,
    find_orphan_nodes,
    load_object_info,
    validate_against_object_info,
    validate_node_references,
)


def main(argv: Optional[list[str]] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 2:
        print(
            "usage: validate_before_queue.py <graph_api.json> <object_info.json>",
            file=sys.stderr,
        )
        return 2
    graph_path, info_path = Path(args[0]), Path(args[1])

    wf = ComfyWorkflow.from_api_json(graph_path)
    print(f"Loaded {wf.name!r} ({len(wf)} nodes) from {graph_path}")

    link_errors = validate_node_references(wf)
    if link_errors:
        print("\nLink errors (fix before queue):")
        for e in link_errors:
            print(f"  - {e}")
        return 1
    print("Link check: OK")

    info = load_object_info(info_path)
    report = validate_against_object_info(wf, info)
    print(f"\nobject_info schema check ({len(info)} class_types): valid={report['valid']}")
    for e in report["errors"]:
        print(f"  error: {e}")
    for w in report["warnings"]:
        print(f"  warn: {w}")

    orphans = find_orphan_nodes(wf)
    if orphans:
        print("\nOrphan nodes (output discarded):")
        for o in orphans:
            print(f"  - {o}")

    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
