"""
Utility workflow templates (community-safe patterns).

Upscale, inpaint, outpaint, and multi-angle character reference sheets using
well-known ComfyUI core nodes (CheckpointLoaderSimple, KSampler, LoadImage,
SaveImage, UpscaleModelLoader, ImagePadForOutpaint, SetLatentNoiseMask, etc.).

These are teaching graphs. You still need matching checkpoints / upscaler
``.pth`` files / images under your local ComfyUI install. They do not encode
private studio stacks.
"""

from typing import List, Optional

from ..core import ComfyWorkflow


class CharacterReferenceTemplate:
    """
    Multi-angle character reference sheets via SDXL T2I.

    Builds several prompts (front / three-quarter / profile / smile) and
    returns one ComfyWorkflow per angle. Useful for collecting consistent
    reference images — not a claim about any private training pipeline.

    Example:
        >>> workflows = CharacterReferenceTemplate.create_batch(
        ...     appearance="young woman with short brown hair and green eyes",
        ...     style="wearing a casual blue sweater",
        ...     base_seed=42,
        ... )
        >>> for wf in workflows:
        ...     wf.save_api_json(f"{wf.name}.json")
    """

    REFERENCE_ANGLES = [
        {
            "suffix": "front",
            "name": "Front View",
            "prompt_add": (
                "front view, looking directly at camera, neutral expression, "
                "portrait, centered composition, clean background, "
                "professional studio lighting, high quality, detailed"
            ),
        },
        {
            "suffix": "threequarter",
            "name": "Three-Quarter View",
            "prompt_add": (
                "three-quarter view, slight angle from camera, "
                "neutral expression, portrait, professional studio lighting, "
                "high quality, detailed"
            ),
        },
        {
            "suffix": "profile",
            "name": "Profile View",
            "prompt_add": (
                "side profile view, facing left, neutral expression, "
                "portrait, professional studio lighting, "
                "high quality, detailed"
            ),
        },
        {
            "suffix": "expression_smile",
            "name": "Smiling Expression",
            "prompt_add": (
                "front view, smiling, happy expression, warm and friendly, "
                "portrait, professional studio lighting, "
                "high quality, detailed"
            ),
        },
    ]

    @classmethod
    def create_batch(
        cls,
        appearance: str,
        style: str = "",
        base_seed: int = 42,
        width: int = 512,
        height: int = 768,
        angles: Optional[List[str]] = None,
        lora: Optional[str] = None,
        lora_strength: float = 1.0,
    ) -> List[ComfyWorkflow]:
        """
        Create one SDXL T2I workflow per reference angle.

        Args:
            appearance: Physical appearance description
            style: Clothing / wardrobe description
            base_seed: Seed for the first angle; incremented per angle
            width: Image width
            height: Image height
            angles: Subset of suffix names, or None for all
            lora: Optional LoRA for style consistency
            lora_strength: LoRA strength

        Returns:
            List of ComfyWorkflow instances
        """
        from .sdxl_templates import SdxlTxt2ImgTemplate

        angle_configs = cls.REFERENCE_ANGLES
        if angles:
            angle_configs = [
                a for a in cls.REFERENCE_ANGLES if a["suffix"] in angles
            ]

        workflows: List[ComfyWorkflow] = []
        for i, angle_config in enumerate(angle_configs):
            prompt_parts = [appearance]
            if style:
                prompt_parts.append(style)
            prompt_parts.append(angle_config["prompt_add"])
            prompt = ", ".join(prompt_parts)

            wf = SdxlTxt2ImgTemplate.create(
                prompt=prompt,
                width=width,
                height=height,
                seed=base_seed + i,
                steps=30,
                cfg_scale=7.0,
                lora=lora,
                lora_strength=lora_strength,
                filename_prefix=f"character_{angle_config['suffix']}_",
            )
            wf.name = f"character_ref_{angle_config['suffix']}"
            workflows.append(wf)

        return workflows

    @classmethod
    def create_single(
        cls,
        appearance: str,
        angle: str = "front",
        style: str = "",
        seed: int = 42,
        width: int = 512,
        height: int = 768,
        lora: Optional[str] = None,
        lora_strength: float = 1.0,
    ) -> ComfyWorkflow:
        """Create a single-angle character reference workflow."""
        workflows = cls.create_batch(
            appearance=appearance,
            style=style,
            base_seed=seed,
            width=width,
            height=height,
            angles=[angle],
            lora=lora,
            lora_strength=lora_strength,
        )
        return workflows[0]


class UpscaleTemplate:
    """
    Image upscale template.

    Default path: UpscaleModelLoader + ImageUpscaleWithModel (ESRGAN-class
    ``.pth`` under ``models/upscale_models/``).

    Optional latent path: VAEEncode → LatentUpscaleBy → light KSampler denoise
    → VAEDecode. Requires an SDXL (or compatible) checkpoint locally.

    Example:
        >>> workflow = UpscaleTemplate.create(
        ...     image_path="render.png",
        ...     upscaler="4x-UltraSharp.pth",
        ... )
    """

    @staticmethod
    def create(
        image_path: str,
        scale: float = 2.0,
        upscaler: str = "4x-UltraSharp.pth",
        seed: int = -1,
        denoise: float = 0.2,
        steps: int = 20,
        model: str = "sd_xl_base_1.0.safetensors",
        use_latent_upscale: bool = False,
        filename_prefix: str = "upscaled_",
    ) -> ComfyWorkflow:
        """
        Build an upscale graph.

        Args:
            image_path: LoadImage path under ComfyUI ``input/``
            scale: Scale factor for LatentUpscaleBy mode
            upscaler: Upscale model filename (model mode)
            seed: KSampler seed (latent mode)
            denoise: Light denoise for latent refine (latent mode)
            steps: KSampler steps (latent mode)
            model: Checkpoint for latent mode
            use_latent_upscale: If True, use LatentUpscaleBy + KSampler
            filename_prefix: SaveImage prefix

        Returns:
            Configured ComfyWorkflow
        """
        wf = ComfyWorkflow("upscale")

        load_image = wf.add_node(
            "LoadImage", {"image": image_path}, title="LoadImage"
        )

        if use_latent_upscale:
            checkpoint = wf.add_node(
                "CheckpointLoaderSimple",
                {"ckpt_name": model},
                title="Load Checkpoint",
            )
            vae_ref = checkpoint.get_output_ref(2)

            encoded = wf.add_node(
                "VAEEncode",
                {"pixels": load_image.get_output_ref(), "vae": vae_ref},
                title="VAEEncode",
            )

            upscaled_latent = wf.add_node(
                "LatentUpscaleBy",
                {
                    "samples": encoded.get_output_ref(),
                    "upscale_method": "nearest-exact",
                    "scale_by": scale,
                },
                title="LatentUpscaleBy",
            )

            positive = wf.add_node(
                "CLIPTextEncode",
                {
                    "text": "detailed, high quality, sharp",
                    "clip": checkpoint.get_output_ref(1),
                },
                title="Positive Prompt",
            )
            negative = wf.add_node(
                "CLIPTextEncode",
                {
                    "text": "blurry, low quality, artifacts",
                    "clip": checkpoint.get_output_ref(1),
                },
                title="Negative Prompt",
            )

            sampler = wf.add_node(
                "KSampler",
                {
                    "seed": seed,
                    "steps": steps,
                    "cfg": 7.0,
                    "sampler_name": "dpmpp_2m",
                    "scheduler": "karras",
                    "denoise": denoise,
                    "model": checkpoint.get_output_ref(),
                    "positive": positive.get_output_ref(),
                    "negative": negative.get_output_ref(),
                    "latent_image": upscaled_latent.get_output_ref(),
                },
                title="Refine Upscaled",
            )

            decoded = wf.add_node(
                "VAEDecode",
                {"samples": sampler.get_output_ref(), "vae": vae_ref},
                title="VAE Decode",
            )
        else:
            upscaler_node = wf.add_node(
                "UpscaleModelLoader",
                {"model_name": upscaler},
                title="UpscaleModelLoader",
            )
            decoded = wf.add_node(
                "ImageUpscaleWithModel",
                {
                    "image": load_image.get_output_ref(),
                    "upscale_model": upscaler_node.get_output_ref(),
                },
                title="ImageUpscaleWithModel",
            )

        wf.add_node(
            "SaveImage",
            {
                "filename_prefix": filename_prefix,
                "images": decoded.get_output_ref(),
            },
            title="SaveImage",
        )

        return wf


class InpaintTemplate:
    """
    SDXL-class inpaint via SetLatentNoiseMask.

    Generic pattern: LoadImage + mask → ImageToMask → VAEEncode →
    SetLatentNoiseMask → KSampler → VAEDecode → SaveImage.

    Specialized inpaint checkpoints may work better; pass them via ``model``.
    Default is the public SDXL base filename as a placeholder — rename to
    whatever you actually have installed.

    Example:
        >>> workflow = InpaintTemplate.create(
        ...     prompt="a red rose in a glass vase",
        ...     image_path="scene.png",
        ...     mask_path="mask.png",
        ... )
    """

    @staticmethod
    def create(
        prompt: str,
        image_path: str,
        mask_path: str,
        negative_prompt: str = "",
        width: int = 1024,
        height: int = 1024,
        seed: int = -1,
        steps: int = 25,
        cfg_scale: float = 7.0,
        denoise: float = 1.0,
        model: str = "sd_xl_base_1.0.safetensors",
        filename_prefix: str = "inpainted_",
    ) -> ComfyWorkflow:
        """
        Build an inpaint graph.

        Args:
            prompt: Content for the masked region
            image_path: Source image
            mask_path: Mask image (white = inpaint)
            negative_prompt: Negative prompt
            width: Optional resize width (0 keeps original)
            height: Optional resize height
            seed: KSampler seed
            steps: KSampler steps
            cfg_scale: CFG
            denoise: Denoise (1.0 = full masked replace)
            model: Checkpoint filename
            filename_prefix: SaveImage prefix

        Returns:
            Configured ComfyWorkflow
        """
        wf = ComfyWorkflow("inpaint")

        checkpoint = wf.add_node(
            "CheckpointLoaderSimple",
            {"ckpt_name": model},
            title="Load Checkpoint",
        )

        load_image = wf.add_node(
            "LoadImage", {"image": image_path}, title="LoadImage"
        )
        load_mask = wf.add_node(
            "LoadImage", {"image": mask_path}, title="Load Mask Image"
        )

        if width > 0 and height > 0:
            resize_img = wf.add_node(
                "ImageScale",
                {
                    "image": load_image.get_output_ref(),
                    "width": width,
                    "height": height,
                    "upscale_method": "lanczos",
                    "crop": "center",
                },
                title="ImageScale",
            )
            img_ref = resize_img.get_output_ref()

            resize_mask = wf.add_node(
                "ImageScale",
                {
                    "image": load_mask.get_output_ref(),
                    "width": width,
                    "height": height,
                    "upscale_method": "nearest-exact",
                    "crop": "center",
                },
                title="Scale Mask",
            )
            mask_image_ref = resize_mask.get_output_ref()
        else:
            img_ref = load_image.get_output_ref()
            mask_image_ref = load_mask.get_output_ref()

        mask = wf.add_node(
            "ImageToMask",
            {"image": mask_image_ref, "channel": "red"},
            title="ImageToMask",
        )

        encoded = wf.add_node(
            "VAEEncode",
            {"pixels": img_ref, "vae": checkpoint.get_output_ref(2)},
            title="VAEEncode",
        )

        masked_latent = wf.add_node(
            "SetLatentNoiseMask",
            {
                "samples": encoded.get_output_ref(),
                "mask": mask.get_output_ref(),
            },
            title="SetLatentNoiseMask",
        )

        clip_ref = checkpoint.get_output_ref(1)

        positive = wf.add_node(
            "CLIPTextEncode",
            {"text": prompt, "clip": clip_ref},
            title="Positive Prompt",
        )
        negative = wf.add_node(
            "CLIPTextEncode",
            {"text": negative_prompt, "clip": clip_ref},
            title="Negative Prompt",
        )

        sampler = wf.add_node(
            "KSampler",
            {
                "seed": seed,
                "steps": steps,
                "cfg": cfg_scale,
                "sampler_name": "dpmpp_2m",
                "scheduler": "karras",
                "denoise": denoise,
                "model": checkpoint.get_output_ref(),
                "positive": positive.get_output_ref(),
                "negative": negative.get_output_ref(),
                "latent_image": masked_latent.get_output_ref(),
            },
            title="KSampler",
        )

        decoded = wf.add_node(
            "VAEDecode",
            {
                "samples": sampler.get_output_ref(),
                "vae": checkpoint.get_output_ref(2),
            },
            title="VAE Decode",
        )

        wf.add_node(
            "SaveImage",
            {
                "filename_prefix": filename_prefix,
                "images": decoded.get_output_ref(),
            },
            title="SaveImage",
        )

        return wf


class OutpaintTemplate:
    """
    Outpaint via ImagePadForOutpaint + SetLatentNoiseMask.

    ImagePadForOutpaint expands the canvas and emits an IMAGE + MASK.
    The mask drives SetLatentNoiseMask so only the padded border is sampled.

    Example:
        >>> workflow = OutpaintTemplate.create(
        ...     prompt="continue the forest landscape",
        ...     image_path="forest.png",
        ...     expand_left=256,
        ...     expand_right=256,
        ... )
    """

    @staticmethod
    def create(
        prompt: str,
        image_path: str,
        expand_left: int = 0,
        expand_right: int = 0,
        expand_top: int = 0,
        expand_bottom: int = 0,
        negative_prompt: str = "",
        seed: int = -1,
        steps: int = 25,
        cfg_scale: float = 7.0,
        denoise: float = 1.0,
        model: str = "sd_xl_base_1.0.safetensors",
        filename_prefix: str = "outpainted_",
    ) -> ComfyWorkflow:
        """
        Build an outpaint graph.

        Args:
            prompt: Content for expanded borders
            image_path: Source image
            expand_left / expand_right / expand_top / expand_bottom: Pad pixels
            negative_prompt: Negative prompt
            seed: KSampler seed
            steps: KSampler steps
            cfg_scale: CFG
            denoise: Denoise on the padded latent (typically 1.0)
            model: Checkpoint filename
            filename_prefix: SaveImage prefix

        Returns:
            Configured ComfyWorkflow

        Raises:
            ValueError: If every expand_* value is 0
        """
        total_expand = expand_left + expand_right + expand_top + expand_bottom
        if total_expand == 0:
            raise ValueError("At least one expansion direction must be > 0")

        wf = ComfyWorkflow("outpaint")

        checkpoint = wf.add_node(
            "CheckpointLoaderSimple",
            {"ckpt_name": model},
            title="Load Checkpoint",
        )
        vae_ref = checkpoint.get_output_ref(2)
        clip_ref = checkpoint.get_output_ref(1)

        load_image = wf.add_node(
            "LoadImage", {"image": image_path}, title="LoadImage"
        )

        padded = wf.add_node(
            "ImagePadForOutpaint",
            {
                "image": load_image.get_output_ref(),
                "left": expand_left,
                "top": expand_top,
                "right": expand_right,
                "bottom": expand_bottom,
                "feathering": 40,
            },
            title="ImagePadForOutpaint",
        )

        encoded = wf.add_node(
            "VAEEncode",
            {"pixels": padded.get_output_ref(0), "vae": vae_ref},
            title="VAEEncode",
        )

        masked_latent = wf.add_node(
            "SetLatentNoiseMask",
            {
                "samples": encoded.get_output_ref(),
                "mask": padded.get_output_ref(1),
            },
            title="SetLatentNoiseMask",
        )

        positive = wf.add_node(
            "CLIPTextEncode",
            {"text": prompt, "clip": clip_ref},
            title="Positive Prompt",
        )
        negative = wf.add_node(
            "CLIPTextEncode",
            {"text": negative_prompt, "clip": clip_ref},
            title="Negative Prompt",
        )

        sampler = wf.add_node(
            "KSampler",
            {
                "seed": seed,
                "steps": steps,
                "cfg": cfg_scale,
                "sampler_name": "dpmpp_2m",
                "scheduler": "karras",
                "denoise": denoise,
                "model": checkpoint.get_output_ref(),
                "positive": positive.get_output_ref(),
                "negative": negative.get_output_ref(),
                "latent_image": masked_latent.get_output_ref(),
            },
            title="KSampler",
        )

        decoded = wf.add_node(
            "VAEDecode",
            {"samples": sampler.get_output_ref(), "vae": vae_ref},
            title="VAE Decode",
        )

        wf.add_node(
            "SaveImage",
            {
                "filename_prefix": filename_prefix,
                "images": decoded.get_output_ref(),
            },
            title="SaveImage",
        )

        return wf
