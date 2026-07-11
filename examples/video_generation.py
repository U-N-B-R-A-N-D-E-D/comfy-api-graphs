#!/usr/bin/env python3
"""
Video Generation Example (quarantined)
======================================

Video templates in this package are stubs: create() raises NotImplementedError.
Prototype LTX / Wan graphs in the ComfyUI UI, Save (API Format), then load or
wrap that JSON with ComfyWorkflow. This package does not ship video production
recipes.
"""

from comfy_api_graphs.templates.video_templates import (
    LtxVideoTemplate,
    WanVideoTemplate,
)


def main():
    print("=" * 60)
    print("Video templates are quarantined (not implemented)")
    print("=" * 60)
    print()
    print("Recommended path:")
    print("  1. Design the graph in ComfyUI UI")
    print("  2. Save (API Format)")
    print("  3. Load or wrap that JSON with ComfyWorkflow in Python")
    print()

    for name, cls in (
        ("LtxVideoTemplate", LtxVideoTemplate),
        ("WanVideoTemplate", WanVideoTemplate),
    ):
        try:
            cls.create(prompt="demo")
        except NotImplementedError as exc:
            print(f"{name}: {exc}")
            print()

    print("=" * 60)


if __name__ == "__main__":
    main()
