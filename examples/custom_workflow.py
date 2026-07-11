#!/usr/bin/env python3
"""
Custom Workflow Building Example
=================================

Shows how to build workflows from scratch using the low-level API.
This gives you full control over node connections.
"""

from comfy_api_graphs import ComfyWorkflow
from comfy_api_graphs import validate_node_references, estimate_vram_usage


def main():
    print("Building custom workflow from scratch...")
    print()

    # Create empty workflow
    wf = ComfyWorkflow("custom_flux_pipeline")

    # Step 1: Load the FLUX models
    print("Step 1: Loading CLIP encoders...")
    clip = wf.add_node(
        "DualCLIPLoader",
        {
            "clip_name1": "t5xxl_fp16.safetensors",
            "clip_name2": "clip_l.safetensors",
            "type": "flux",
        },
        title="Load Dual CLIP",
    )

    print("Step 2: Loading FLUX diffusion model...")
    unet = wf.add_node(
        "UNETLoader",
        {"unet_name": "flux1-dev.safetensors", "weight_dtype": "default"},
        title="Load FLUX UNET",
    )

    print("Step 3: Loading VAE...")
    vae = wf.add_node(
        "VAELoader",
        {"vae_name": "ae.safetensors"},
        title="Load VAE",
    )

    # Encode positive + empty negative (FLUX distilled guidance)
    print("Step 4: Encoding prompts...")
    positive = wf.add_node(
        "CLIPTextEncode",
        {
            "text": "a futuristic cyberpunk city at night, neon lights, "
                    "flying vehicles, detailed architecture",
            "clip": clip.get_output_ref(),
        },
        title="Positive Prompt",
    )
    negative = wf.add_node(
        "CLIPTextEncode",
        {"text": "", "clip": clip.get_output_ref()},
        title="Empty Negative (FLUX)",
    )

    print("Step 5: Creating latent space...")
    latent = wf.add_node(
        "EmptySD3LatentImage",
        {"width": 1024, "height": 1024, "batch_size": 1},
        title="Empty Latent",
    )

    print("Step 6: Configuring ModelSamplingFlux...")
    model_sampling = wf.add_node(
        "ModelSamplingFlux",
        {
            "max_shift": 1.15,
            "base_shift": 0.5,
            "width": 1024,
            "height": 1024,
            "model": unet.get_output_ref(),
        },
        title="ModelSamplingFlux",
    )

    print("Step 7: Running KSampler...")
    sampler = wf.add_node(
        "KSampler",
        {
            "seed": 42,
            "steps": 20,
            "cfg": 3.5,
            "sampler_name": "euler",
            "scheduler": "simple",
            "denoise": 1.0,
            "model": model_sampling.get_output_ref(),
            "positive": positive.get_output_ref(),
            "negative": negative.get_output_ref(),
            "latent_image": latent.get_output_ref(),
        },
        title="KSampler",
    )

    # Step 6: Decode to image
    print("Step 8: Decoding latent to image...")
    decoded = wf.add_node(
        "VAEDecode",
        {"samples": sampler.get_output_ref(), "vae": vae.get_output_ref()},
        title="VAE Decode",
    )

    # Step 7: Save the result
    print("Step 9: Saving output...")
    wf.add_node(
        "SaveImage",
        {
            "filename_prefix": "custom_flux_",
            "images": decoded.get_output_ref(),
        },
        title="Save Image",
    )

    # Print workflow stats
    print()
    print("=" * 50)
    print("Workflow Complete!")
    print("=" * 50)
    print(f"Name: {wf.name}")
    print(f"Total nodes: {len(wf)}")

    # Validate
    errors = validate_node_references(wf)
    print(f"Validation: {'OK' if not errors else f'{len(errors)} errors'}")
    if errors:
        for e in errors:
            print(f"  - {e}")

    # VRAM estimate
    vram = estimate_vram_usage(wf)
    print(f"VRAM estimate: {vram['estimated_vram_gb']:.1f} GB")
    print(f"  ({vram['recommendation']})")

    # Save
    output_file = "custom_workflow.json"
    wf.save_api_json(output_file)
    print()
    print(f"Saved to: {output_file}")

    # Show the API format structure
    print()
    print("API Format Structure:")
    api_format = wf.to_api_format()
    for node_id in list(api_format.keys())[:3]:  # First 3 nodes
        node = api_format[node_id]
        print(f"  Node {node_id}: {node['class_type']}")


if __name__ == "__main__":
    main()
