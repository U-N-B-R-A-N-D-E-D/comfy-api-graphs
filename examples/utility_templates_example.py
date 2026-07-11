#!/usr/bin/env python3
"""
Utility templates example
=========================

Upscale, inpaint, outpaint, and character reference sheets using core ComfyUI
nodes. Requires matching checkpoints / upscaler weights / input images on your
local ComfyUI install — these graphs teach structure, not magic production.
"""

from comfy_api_graphs import (
    UpscaleTemplate,
    InpaintTemplate,
    OutpaintTemplate,
    CharacterReferenceTemplate,
    validate_node_references,
)


def main():
    print("=" * 60)
    print("Utility template examples (public-safe patterns)")
    print("=" * 60)

    # 1) Model upscale (UpscaleModelLoader + ImageUpscaleWithModel)
    print("\n1. Upscale")
    upscale = UpscaleTemplate.create(
        image_path="render.png",
        upscaler="4x-UltraSharp.pth",
    )
    assert validate_node_references(upscale) == []
    upscale.save_api_json("upscale_example.json")
    print(f"  Saved upscale_example.json ({len(upscale)} nodes)")

    # 2) Inpaint (SetLatentNoiseMask)
    print("\n2. Inpaint")
    inpaint = InpaintTemplate.create(
        prompt="a red rose in a glass vase",
        image_path="scene.png",
        mask_path="mask.png",
    )
    assert validate_node_references(inpaint) == []
    inpaint.save_api_json("inpaint_example.json")
    print(f"  Saved inpaint_example.json ({len(inpaint)} nodes)")

    # 3) Outpaint (ImagePadForOutpaint)
    print("\n3. Outpaint")
    outpaint = OutpaintTemplate.create(
        prompt="continue the forest landscape into misty hills",
        image_path="forest.png",
        expand_left=256,
        expand_right=256,
    )
    assert validate_node_references(outpaint) == []
    outpaint.save_api_json("outpaint_example.json")
    print(f"  Saved outpaint_example.json ({len(outpaint)} nodes)")

    # 4) Character reference batch (SDXL T2I prompts)
    print("\n4. Character reference sheets")
    refs = CharacterReferenceTemplate.create_batch(
        appearance="young woman with short brown hair",
        style="wearing a casual blue sweater",
        angles=["front", "profile"],
        base_seed=42,
    )
    for wf in refs:
        assert validate_node_references(wf) == []
        wf.save_api_json(f"{wf.name}.json")
        print(f"  Saved {wf.name}.json")

    print("\n" + "=" * 60)
    print("Notes:")
    print("- Put images under ComfyUI input/; checkpoints under models/")
    print("- Video (LTX / Wan) is intentionally not shipped as templates")
    print("- Design complex graphs in the UI, then Save (API Format)")
    print("=" * 60)


if __name__ == "__main__":
    main()
