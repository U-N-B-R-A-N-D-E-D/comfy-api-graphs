#!/usr/bin/env python3
"""
Basic Text-to-Image Example
===========================

Demonstrates the simplest possible workflow: generating an image from text.
"""

from comfy_api_graphs import FluxTxt2ImgTemplate


def main():
    print("Creating FLUX text-to-image workflow...")

    # Create a basic workflow
    workflow = FluxTxt2ImgTemplate.create(
        prompt="a serene mountain landscape at golden hour, dramatic clouds, "
               "snow-capped peaks reflected in a crystal clear lake",
        width=1024,
        height=1024,
        seed=42,  # Fixed seed for reproducibility
        steps=20,
    )

    # Save the workflow to JSON
    output_file = "basic_txt2img.json"
    workflow.save_api_json(output_file)
    print(f"Workflow saved to: {output_file}")

    # Show workflow stats
    print(f"\nWorkflow stats:")
    print(f"  - Nodes: {len(workflow)}")
    print(f"  - Format: ComfyUI API")

    # Validate the workflow
    from comfy_api_graphs import validate_node_references

    errors = validate_node_references(workflow)
    if errors:
        print(f"  - Validation: {len(errors)} errors found")
        for error in errors:
            print(f"    - {error}")
    else:
        print("  - Validation: OK")

    print("\nTo execute this workflow:")
    print("1. Start ComfyUI server")
    print("2. Load the JSON in ComfyUI or send via API")
    print("3. Queue the prompt")


if __name__ == "__main__":
    main()
