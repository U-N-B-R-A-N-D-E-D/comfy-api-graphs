#!/usr/bin/env python3
"""
Seed batch cookbook
===================

Load a Save (API Format) graph, mutate seeds, optionally POST /prompt in a loop.

Queue time saved: no hand-editing JSON and no re-export from the node editor
for every seed variation (ComfyUI issues #850 / #1085 class of pain).

Usage:
    # Offline: write N mutated copies (default)
    python examples/seed_batch_cookbook.py path/to/graph_api.json --count 5 --dry-run

    # Live queue (ComfyUI must be up):
    python examples/seed_batch_cookbook.py path/to/graph_api.json --count 5 --queue

Stdlib only for the HTTP path (urllib) — matches examples/execution_example.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from urllib import error, request

from comfy_api_graphs import ComfyWorkflow, validate_node_references

DEFAULT_URL = "http://127.0.0.1:8188"


def queue_prompt(api_graph: dict, base_url: str) -> str:
    """POST ``{"prompt": ...}`` to ComfyUI; return prompt_id."""
    payload = json.dumps({"prompt": api_graph}).encode("utf-8")
    req = request.Request(
        f"{base_url.rstrip('/')}/prompt",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req) as response:
        result = json.loads(response.read().decode("utf-8"))
    return result["prompt_id"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Load API-format JSON, mutate seeds, batch write or queue."
    )
    parser.add_argument(
        "api_json",
        type=Path,
        help="Path to ComfyUI Save (API Format) JSON",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=5,
        help="How many seed variants to produce (default: 5)",
    )
    parser.add_argument(
        "--mode",
        choices=("randomize", "bump"),
        default="randomize",
        help="randomize = fresh seeds each pass; bump = seed += index",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("seed_batch_out"),
        help="Where to write mutated JSON when not --queue-only",
    )
    parser.add_argument(
        "--queue",
        action="store_true",
        help="POST each variant to ComfyUI /prompt",
    )
    parser.add_argument(
        "--queue-only",
        action="store_true",
        help="With --queue, skip writing JSON files",
    )
    parser.add_argument(
        "--url",
        default=DEFAULT_URL,
        help=f"ComfyUI base URL (default: {DEFAULT_URL})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned seeds; do not write or queue",
    )
    args = parser.parse_args(argv)

    if not args.api_json.is_file():
        print(f"File not found: {args.api_json}", file=sys.stderr)
        return 1

    base = ComfyWorkflow.from_api_json(args.api_json)
    errors = validate_node_references(base)
    if errors:
        print("Link validation failed before batch:")
        for err in errors:
            print(f"  - {err}")
        return 1

    if not args.queue_only and not args.dry_run:
        args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loaded {base.name!r} ({len(base)} nodes) from {args.api_json}")
    seed_slots = base.iter_seed_inputs()
    if not seed_slots:
        print("No seed / noise_seed inputs found — nothing to batch.", file=sys.stderr)
        return 1
    print(f"Seed slots: {seed_slots}")

    for i in range(args.count):
        wf = ComfyWorkflow.from_api_dict(base.to_api_format(), name=f"{base.name}_s{i}")
        if args.mode == "randomize":
            wf.randomize_seeds()
        else:
            # Deterministic series: bump by batch index from the loaded baseline.
            wf.bump_seeds(i)

        seeds_now = {
            nid: wf.get_node(nid).inputs.get(key) for nid, key in seed_slots
        }
        print(f"[{i + 1}/{args.count}] seeds={seeds_now}")

        if args.dry_run:
            continue

        if not args.queue_only:
            out_path = args.out_dir / f"{args.api_json.stem}_seed_{i:03d}.json"
            wf.save_api_json(str(out_path))
            print(f"  wrote {out_path}")

        if args.queue:
            try:
                prompt_id = queue_prompt(wf.to_api_format(), args.url)
                print(f"  queued prompt_id={prompt_id}")
            except error.URLError as exc:
                print(f"  queue failed: {exc}", file=sys.stderr)
                return 1

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
