#!/usr/bin/env python3
"""
Diff two workflows
==================

Compare two Save (API Format) graphs and print a structured diff plus a
Mermaid view. Useful in code review when a graph was edited in the ComfyUI
UI and re-exported: see exactly which nodes/inputs changed before you
re-queue, and paste the Mermaid block into the PR for inline rendering.

Usage:
    python examples/diff_workflows.py a_api.json b_api.json
    python examples/diff_workflows.py a_api.json b_api.json --mermaid
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from comfy_api_graphs import ComfyWorkflow


def main(argv: Optional[list[str]] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) < 2:
        print("usage: diff_workflows.py <a_api.json> <b_api.json> [--mermaid]", file=sys.stderr)
        return 2

    show_mermaid = "--mermaid" in args
    paths = [a for a in args if not a.startswith("-")]
    if len(paths) != 2:
        print("usage: diff_workflows.py <a_api.json> <b_api.json> [--mermaid]", file=sys.stderr)
        return 2

    a = ComfyWorkflow.from_api_json(Path(paths[0]))
    b = ComfyWorkflow.from_api_json(Path(paths[1]))
    d = a.diff(b)

    print(f"A: {a.name!r} ({len(a)} nodes)")
    print(f"B: {b.name!r} ({len(b)} nodes)")
    print(f"added nodes:         {d['added']}")
    print(f"removed nodes:       {d['removed']}")
    print(f"changed class_type:  {d['changed_class']}")
    print(f"added inputs:        {d['added_inputs']}")
    print(f"removed inputs:      {d['removed_inputs']}")
    print(f"changed inputs:      {d['changed_inputs']}")

    if show_mermaid:
        print("\n--- Mermaid (B) ---")
        print(b.to_mermaid())

    changed = bool(
        d["added"]
        or d["removed"]
        or d["changed_class"]
        or d["added_inputs"]
        or d["removed_inputs"]
        or d["changed_inputs"]
    )
    return 1 if changed else 0


if __name__ == "__main__":
    raise SystemExit(main())
