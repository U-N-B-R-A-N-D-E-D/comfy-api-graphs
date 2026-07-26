"""
Command-line entry point: ``python -m comfy_api_graphs``.

A minimal, stdlib-only CLI for scripting ComfyUI API-format graphs in CI
without a GPU. Three subcommands:

- ``validate <file> [--object-info <snap.json>]`` — link check, optional
  object_info schema check, orphan + duplicate-seed warnings.
- ``seeds <file> --randomize|--bump N -o <out.json>`` — mutate seeds and
  write a new API-format file (no queue, no GPU).
- ``show <file> --mermaid`` — print a Mermaid flowchart of the graph (paste
  into a PR/issue for inline rendering on GitHub).

Argparse only. No runtime dependencies. This is the same surface the
library exposes; the CLI just wraps it for shell use.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from .core import ComfyWorkflow
from .validation import (
    find_duplicate_seeds_across_batch,
    find_orphan_nodes,
    load_object_info,
    validate_against_object_info,
    validate_node_references,
    validate_workflow_complete,
)


def _load_workflow(path: Path) -> ComfyWorkflow:
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    return ComfyWorkflow.from_api_json(path)


def _cmd_validate(args: argparse.Namespace) -> int:
    wf = _load_workflow(Path(args.file))
    print(f"Loaded {wf.name!r} ({len(wf)} nodes)")

    link_errors = validate_node_references(wf)
    if link_errors:
        print("\nLink errors:")
        for e in link_errors:
            print(f"  - {e}")
    else:
        print("Link check: OK")

    complete = validate_workflow_complete(wf)
    for w in complete["warnings"]:
        print(f"  warn: {w}")

    if args.object_info:
        info_path = Path(args.object_info)
        if not info_path.is_file():
            print(f"object_info snapshot not found: {info_path}", file=sys.stderr)
            return 2
        info = load_object_info(info_path)
        report = validate_against_object_info(wf, info)
        print(f"\nobject_info schema check ({len(info)} class_types):")
        if report["valid"]:
            print("  valid: True")
        else:
            print("  valid: False")
            for e in report["errors"]:
                print(f"  error: {e}")
            link_errors.extend(report["errors"])
        for w in report["warnings"]:
            print(f"  warn: {w}")

    orphans = find_orphan_nodes(wf)
    if orphans:
        print("\nOrphan nodes (output discarded, not an output node):")
        for o in orphans:
            print(f"  - {o}")

    if link_errors:
        return 1
    return 0


def _cmd_seeds(args: argparse.Namespace) -> int:
    wf = _load_workflow(Path(args.file))
    seed_slots = wf.iter_seed_inputs()
    if not seed_slots:
        print("No seed / noise_seed inputs found — nothing to mutate.", file=sys.stderr)
        return 1
    print(f"Seed slots: {seed_slots}")

    if args.randomize and args.bump is not None:
        print("Use --randomize OR --bump N, not both.", file=sys.stderr)
        return 2

    if args.count < 1:
        print("--count must be >= 1.", file=sys.stderr)
        return 2

    out_dir = Path(args.out_dir) if args.out_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    copies = list(
        wf.batch_seeds(
            args.count,
            mode="randomize" if args.randomize else "bump",
            start=args.bump if args.bump is not None else None,
        )
    )

    collisions = find_duplicate_seeds_across_batch(copies)
    if collisions:
        print("\nWARNING: seed collisions across the batch (ComfyUI will cache-skip):")
        for value, slots in collisions.items():
            print(f"  seed={value}: {slots}")
        if args.fail_on_collision:
            return 1

    for i, c in enumerate(copies):
        seeds_now = {nid: c.nodes[nid].inputs.get(k) for nid, k in seed_slots}
        print(f"[{i + 1}/{args.count}] seeds={seeds_now}")
        if out_dir:
            out_path = out_dir / f"{Path(args.file).stem}_seed_{i:03d}.json"
            c.save_api_json(str(out_path))
            print(f"  wrote {out_path}")
    return 0


def _cmd_show(args: argparse.Namespace) -> int:
    wf = _load_workflow(Path(args.file))
    if args.mermaid:
        print(wf.to_mermaid(show_class=not args.ids))
        return 0
    # Default: print a compact node list.
    print(f"# {wf.name} ({len(wf)} nodes)")
    for nid in wf.topological_order():
        node = wf.nodes[nid]
        title = f"  title={node.title!r}" if node.title else ""
        print(f"  {nid}: {node.class_type}{title}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m comfy_api_graphs",
        description=(
            "Offline stdlib CLI for ComfyUI API-format graphs: validate, "
            "mutate seeds, or render a graph. No GPU / no live ComfyUI needed."
        ),
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_val = sub.add_parser("validate", help="Validate links + optional object_info schema")
    p_val.add_argument("file", type=Path, help="Path to a Save (API Format) JSON")
    p_val.add_argument(
        "--object-info",
        type=Path,
        default=None,
        help="Optional /object_info snapshot JSON for schema checks",
    )
    p_val.set_defaults(func=_cmd_validate)

    p_seeds = sub.add_parser("seeds", help="Mutate seeds and write N copies")
    p_seeds.add_argument("file", type=Path, help="Path to a Save (API Format) JSON")
    p_seeds.add_argument(
        "--randomize",
        action="store_true",
        help="Assign a fresh random int seed to each copy",
    )
    p_seeds.add_argument(
        "--bump",
        type=int,
        default=None,
        help="Bump mode: first copy seed = N, later copies = N + index",
    )
    p_seeds.add_argument(
        "--count",
        type=int,
        default=1,
        help="Number of seed variants to produce (default: 1)",
    )
    p_seeds.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory to write mutated JSON (omit to dry-run only)",
    )
    p_seeds.add_argument(
        "--fail-on-collision",
        action="store_true",
        help="Exit non-zero if any seed value collides across the batch",
    )
    p_seeds.set_defaults(func=_cmd_seeds)

    p_show = sub.add_parser("show", help="Print the graph (topological order or Mermaid)")
    p_show.add_argument("file", type=Path, help="Path to a Save (API Format) JSON")
    p_show.add_argument(
        "--mermaid",
        action="store_true",
        help="Emit a Mermaid flowchart (renders inline on GitHub)",
    )
    p_show.add_argument(
        "--ids",
        action="store_true",
        help="With --mermaid: label nodes by id only (compact)",
    )
    p_show.set_defaults(func=_cmd_show)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
