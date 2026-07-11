#!/usr/bin/env python3
"""
Image-to-Image with LoRA Example
=================================

Shows how to transform an existing image using a style LoRA.
"""

from comfy_api_graphs import FluxImg2ImgTemplate, validate_node_references


def main():
    print("Creating Image-to-Image workflow with LoRA...")

    # Example: Transform a photo to anime style
    workflow = FluxImg2ImgTemplate.create(
        # The transformation prompt
        prompt="anime style artwork, vibrant colors, detailed illustration, "
               "studio ghibli aesthetic, clean linework",

        # Input image path (relative to ComfyUI's input directory)
        image_path="source_photo.png",

        # Output dimensions
        width=1024,
        height=1024,

        # Transformation strength
        # 0.0 = no change, 1.0 = complete regeneration
        denoise=0.75,

        # LoRA to apply
        lora="anime_v1.safetensors",  # Your LoRA file
        lora_strength=0.8,  # How strongly to apply the style

        # Generation parameters
        steps=25,
        seed=12345,
    )

    # Validate
    errors = validate_node_references(workflow)
    if errors:
        print("Validation errors:")
        for e in errors:
            print(f"  - {e}")
    else:
        print("Validation: OK")

    # Save
    output_file = "img2img_lora.json"
    workflow.save_api_json(output_file)
    print(f"\nWorkflow saved to: {output_file}")

    print("\nNotes:")
    print("- The image_path should be in ComfyUI's input directory")
    print("- The LoRA file should be in ComfyUI's models/loras directory")
    print("- Higher denoise = more transformation")
    print("- Higher lora_strength = stronger style influence")


if __name__ == "__main__":
    main()
