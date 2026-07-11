#!/usr/bin/env python3
"""
Execution Example
=================

Shows how to build a workflow AND execute it on a running ComfyUI server.
This requires ComfyUI to be running and accessible.

Prerequisites:
    pip install requests websockets
    # Or from a local clone: pip install -e ".[examples]"
"""

import json
import time
from urllib import request, error

from comfy_api_graphs import FluxTxt2ImgTemplate


# Configuration
COMFYUI_URL = "http://127.0.0.1:8188"


def queue_prompt(workflow_dict: dict) -> str:
    """
    Queue a workflow to ComfyUI server.

    Args:
        workflow_dict: Workflow in ComfyUI API format

    Returns:
        prompt_id for tracking
    """
    data = json.dumps({"prompt": workflow_dict}).encode("utf-8")
    req = request.Request(
        f"{COMFYUI_URL}/prompt",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with request.urlopen(req) as response:
            result = json.loads(response.read().decode("utf-8"))
            return result["prompt_id"]
    except error.HTTPError as e:
        print(f"Error queueing prompt: {e.read().decode()}")
        raise


def get_history(prompt_id: str) -> dict:
    """Get execution history for a prompt."""
    req = request.Request(f"{COMFYUI_URL}/history/{prompt_id}")
    with request.urlopen(req) as response:
        return json.loads(response.read().decode("utf-8"))


def check_server() -> bool:
    """Check if ComfyUI server is running."""
    try:
        req = request.Request(f"{COMFYUI_URL}/system_stats")
        with request.urlopen(req, timeout=5) as response:
            stats = json.loads(response.read().decode("utf-8"))
            print(f"Connected to ComfyUI")
            print(f"  - Device: {stats.get('devices', [{}])[0].get('name', 'Unknown')}")
            return True
    except Exception as e:
        print(f"Cannot connect to ComfyUI at {COMFYUI_URL}")
        print(f"Error: {e}")
        return False


def main():
    print("=" * 60)
    print("ComfyUI Workflow Execution Example")
    print("=" * 60)

    # Check server
    if not check_server():
        print("\nPlease start ComfyUI first:")
        print("  cd /path/to/ComfyUI")
        print("  python main.py")
        return

    # Create workflow
    print("\nCreating workflow...")
    workflow = FluxTxt2ImgTemplate.create(
        prompt="a cute robot reading a book in a library",
        width=1024,
        height=1024,
        seed=42,
        steps=20,
    )

    # Get API format
    workflow_api = workflow.to_api_format()
    print(f"Workflow: {len(workflow)} nodes")

    # Queue the workflow
    print("\nQueueing to ComfyUI...")
    try:
        prompt_id = queue_prompt(workflow_api)
        print(f"Prompt queued with ID: {prompt_id}")
    except Exception as e:
        print(f"Failed to queue: {e}")
        return

    # Poll for completion
    print("\nWaiting for completion...")
    max_wait = 300  # 5 minutes
    waited = 0

    while waited < max_wait:
        time.sleep(2)
        waited += 2

        history = get_history(prompt_id)
        if prompt_id in history:
            outputs = history[prompt_id].get("outputs", {})
            if outputs:
                print("\n" + "=" * 60)
                print("Generation complete!")
                print("=" * 60)

                # Show outputs
                for node_id, node_output in outputs.items():
                    if "images" in node_output:
                        for img in node_output["images"]:
                            print(f"Output image: {img['filename']}")
                            print(f"  Subfolder: {img.get('subfolder', 'none')}")
                            print(f"  Type: {img.get('type', 'output')}")
                return

        if waited % 10 == 0:
            print(f"  Still waiting... ({waited}s)")

    print("\nTimeout waiting for completion")


if __name__ == "__main__":
    main()
