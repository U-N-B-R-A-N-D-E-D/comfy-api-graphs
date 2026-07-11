"""
FLUX.1 workflow templates (community-safe patterns).

FLUX.1 by Black Forest Labs is a text-to-image model commonly run in ComfyUI
via UNETLoader + DualCLIPLoader (T5-XXL + CLIP-L) + VAELoader (ae.safetensors).

These templates teach correct graph structure for T2I / I2I / inpaint.
You still need matching checkpoint, CLIP, VAE, and optional LoRA files under
your local ComfyUI ``models/`` tree.

Comfy slang used here: checkpoint/UNET, CLIP, VAE, KSampler, latent, denoise,
CFG, LoRA stack, LoadImage, SaveImage, API format.
"""

from typing import Optional

from ..core import ComfyWorkflow, WorkflowNode


def _flux_empty_negative(wf: ComfyWorkflow, clip_ref) -> WorkflowNode:
    """
    FLUX guidance is distilled — negative conditioning is typically empty.

    Community-correct pattern: CLIPTextEncode with an empty string, not a
    fake node ref like ``["0", 0]`` or a non-API sentinel object.
    """
    return wf.add_node(
        "CLIPTextEncode",
        {"text": "", "clip": clip_ref},
        title="Empty Negative (FLUX)",
    )


class FluxTxt2ImgTemplate:
    """
    FLUX.1 text-to-image workflow template.

    Supports both FLUX.1 [dev] (quality) and [schnell] (few steps).
    FLUX uses guidance-distilled training, so CFG is typically ~3.5.

    Requires locally: UNET (``flux1-dev.safetensors`` or similar),
    DualCLIP (``t5xxl_*.safetensors`` + ``clip_l.safetensors``),
    VAE (``ae.safetensors``). Optional LoRA under ``models/loras/``.

    Example:
        >>> workflow = FluxTxt2ImgTemplate.create(
        ...     prompt="a serene mountain landscape at sunset",
        ...     width=1024,
        ...     height=1024,
        ...     model="flux1-dev.safetensors",
        ...     steps=20,
        ... )
        >>> workflow.save_api_json("flux_t2i.json")
    """

    @staticmethod
    def create(
        prompt: str,
        width: int = 1024,
        height: int = 1024,
        seed: int = -1,
        steps: int = 20,
        cfg_scale: float = 3.5,
        model: str = "flux1-dev.safetensors",
        vae: str = "ae.safetensors",
        clip1: str = "t5xxl_fp16.safetensors",
        clip2: str = "clip_l.safetensors",
        lora: Optional[str] = None,
        lora_strength: float = 1.0,
        filename_prefix: str = "flux_",
    ) -> ComfyWorkflow:
        """
        Build a FLUX text-to-image graph (API format via ``to_api_format()``).

        Args:
            prompt: Positive prompt text for CLIPTextEncode
            width: Latent width (divisible by 16; 512–2048 typical)
            height: Latent height (divisible by 16)
            seed: KSampler seed (-1 for random if the UI allows)
            steps: KSampler steps (schnell often 4; dev often 20+)
            cfg_scale: CFG (FLUX typically ~3.5)
            model: UNET filename under ``models/unet/`` (or diffusion_models)
            vae: VAE filename (FLUX uses ``ae.safetensors``)
            clip1: T5-XXL CLIP filename
            clip2: CLIP-L filename
            lora: Optional LoRA filename for a single LoRA stack hop
            lora_strength: LoRA strength on model and CLIP (0.0–2.0)
            filename_prefix: SaveImage filename prefix

        Returns:
            Configured ComfyWorkflow ready for ``save_api_json`` / ``/prompt``
        """
        wf = ComfyWorkflow("flux_txt2img")

        clip = wf.add_node(
            "DualCLIPLoader",
            {"clip_name1": clip1, "clip_name2": clip2, "type": "flux"},
            title="Load Dual CLIP",
        )

        unet = wf.add_node(
            "UNETLoader",
            {"unet_name": model, "weight_dtype": "default"},
            title="Load FLUX UNET",
        )

        if lora:
            lora_node = wf.add_node(
                "LoraLoader",
                {
                    "lora_name": lora,
                    "strength_model": lora_strength,
                    "strength_clip": lora_strength,
                    "model": unet.get_output_ref(),
                    "clip": clip.get_output_ref(),
                },
                title="LoRA Stack",
            )
            model_ref = lora_node.get_output_ref()
            clip_ref = lora_node.get_output_ref(1)
        else:
            model_ref = unet.get_output_ref()
            clip_ref = clip.get_output_ref()

        vae_node = wf.add_node(
            "VAELoader", {"vae_name": vae}, title="Load FLUX VAE"
        )

        positive = wf.add_node(
            "CLIPTextEncode",
            {"text": prompt, "clip": clip_ref},
            title="Positive Prompt",
        )
        negative = _flux_empty_negative(wf, clip_ref)

        latent = wf.add_node(
            "EmptySD3LatentImage",
            {"width": width, "height": height, "batch_size": 1},
            title="Empty Latent",
        )

        model_sampling = wf.add_node(
            "ModelSamplingFlux",
            {
                "max_shift": 1.15,
                "base_shift": 0.5,
                "width": width,
                "height": height,
                "model": model_ref,
            },
            title="ModelSamplingFlux",
        )

        sampler = wf.add_node(
            "KSampler",
            {
                "seed": seed,
                "steps": steps,
                "cfg": cfg_scale,
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

        decoded = wf.add_node(
            "VAEDecode",
            {
                "samples": sampler.get_output_ref(),
                "vae": vae_node.get_output_ref(),
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


class FluxImg2ImgTemplate:
    """
    FLUX.1 image-to-image workflow template.

    LoadImage → (optional ImageScale) → VAEEncode → KSampler(denoise) →
    VAEDecode → SaveImage. Denoise controls how far the latent drifts from
    the source image (0.0 ≈ keep, 1.0 ≈ full regenerate).

    Example:
        >>> workflow = FluxImg2ImgTemplate.create(
        ...     prompt="convert to watercolor painting style",
        ...     image_path="input.png",
        ...     denoise=0.75,
        ...     steps=20,
        ... )
    """

    @staticmethod
    def create(
        prompt: str,
        image_path: str,
        width: int = 1024,
        height: int = 1024,
        seed: int = -1,
        steps: int = 20,
        cfg_scale: float = 3.5,
        denoise: float = 0.75,
        model: str = "flux1-dev.safetensors",
        vae: str = "ae.safetensors",
        clip1: str = "t5xxl_fp16.safetensors",
        clip2: str = "clip_l.safetensors",
        lora: Optional[str] = None,
        lora_strength: float = 1.0,
        filename_prefix: str = "flux_i2i_",
    ) -> ComfyWorkflow:
        """
        Build a FLUX image-to-image graph.

        Args:
            prompt: Transformation prompt
            image_path: Filename relative to ComfyUI ``input/`` (LoadImage)
            width: Target width after ImageScale (0 skips resize)
            height: Target height after ImageScale (0 skips resize)
            seed: KSampler seed
            steps: KSampler steps
            cfg_scale: CFG
            denoise: Denoise strength (0.0–1.0)
            model: UNET filename
            vae: VAE filename
            clip1: T5-XXL filename
            clip2: CLIP-L filename
            lora: Optional LoRA filename
            lora_strength: LoRA strength
            filename_prefix: SaveImage prefix

        Returns:
            Configured ComfyWorkflow
        """
        wf = ComfyWorkflow("flux_img2img")

        clip = wf.add_node(
            "DualCLIPLoader",
            {"clip_name1": clip1, "clip_name2": clip2, "type": "flux"},
            title="Load Dual CLIP",
        )

        unet = wf.add_node(
            "UNETLoader",
            {"unet_name": model, "weight_dtype": "default"},
            title="Load FLUX UNET",
        )

        if lora:
            lora_node = wf.add_node(
                "LoraLoader",
                {
                    "lora_name": lora,
                    "strength_model": lora_strength,
                    "strength_clip": lora_strength,
                    "model": unet.get_output_ref(),
                    "clip": clip.get_output_ref(),
                },
                title="LoRA Stack",
            )
            model_ref = lora_node.get_output_ref()
            clip_ref = lora_node.get_output_ref(1)
        else:
            model_ref = unet.get_output_ref()
            clip_ref = clip.get_output_ref()

        vae_node = wf.add_node(
            "VAELoader", {"vae_name": vae}, title="Load FLUX VAE"
        )

        load_image = wf.add_node(
            "LoadImage", {"image": image_path}, title="LoadImage"
        )

        if width > 0 and height > 0:
            resize = wf.add_node(
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
            image_ref = resize.get_output_ref()
        else:
            image_ref = load_image.get_output_ref()

        encoded = wf.add_node(
            "VAEEncode",
            {"pixels": image_ref, "vae": vae_node.get_output_ref()},
            title="VAEEncode",
        )

        positive = wf.add_node(
            "CLIPTextEncode",
            {"text": prompt, "clip": clip_ref},
            title="Positive Prompt",
        )
        negative = _flux_empty_negative(wf, clip_ref)

        model_sampling = wf.add_node(
            "ModelSamplingFlux",
            {
                "max_shift": 1.15,
                "base_shift": 0.5,
                "width": width if width > 0 else 1024,
                "height": height if height > 0 else 1024,
                "model": model_ref,
            },
            title="ModelSamplingFlux",
        )

        sampler = wf.add_node(
            "KSampler",
            {
                "seed": seed,
                "steps": steps,
                "cfg": cfg_scale,
                "sampler_name": "euler",
                "scheduler": "simple",
                "denoise": denoise,
                "model": model_sampling.get_output_ref(),
                "positive": positive.get_output_ref(),
                "negative": negative.get_output_ref(),
                "latent_image": encoded.get_output_ref(),
            },
            title="KSampler",
        )

        decoded = wf.add_node(
            "VAEDecode",
            {
                "samples": sampler.get_output_ref(),
                "vae": vae_node.get_output_ref(),
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


class FluxInpaintTemplate:
    """
    FLUX.1 inpaint template (generic SetLatentNoiseMask pattern).

    LoadImage (pixels) + LoadImage (mask) → ImageToMask → VAEEncode →
    SetLatentNoiseMask → KSampler → VAEDecode → SaveImage.

    White mask areas are regenerated; black areas stay. Requires the same
    FLUX UNET / DualCLIP / VAE files as T2I. This is a teaching graph —
    dedicated FLUX Fill / inpaint UNET checkpoints may need a different
    encode path; adjust in the ComfyUI UI if your models demand it.

    Example:
        >>> workflow = FluxInpaintTemplate.create(
        ...     prompt="a red apple on the table",
        ...     image_path="scene.png",
        ...     mask_path="mask.png",
        ...     denoise=1.0,
        ... )
    """

    @staticmethod
    def create(
        prompt: str,
        image_path: str,
        mask_path: str,
        width: int = 1024,
        height: int = 1024,
        seed: int = -1,
        steps: int = 20,
        cfg_scale: float = 3.5,
        denoise: float = 1.0,
        model: str = "flux1-dev.safetensors",
        vae: str = "ae.safetensors",
        clip1: str = "t5xxl_fp16.safetensors",
        clip2: str = "clip_l.safetensors",
        filename_prefix: str = "flux_inpaint_",
    ) -> ComfyWorkflow:
        """
        Build a FLUX inpaint graph using SetLatentNoiseMask.

        Args:
            prompt: What to generate inside the masked region
            image_path: Source image (LoadImage, under ComfyUI ``input/``)
            mask_path: Mask image (white = inpaint, black = keep)
            width: Optional ImageScale width (0 keeps original)
            height: Optional ImageScale height (0 keeps original)
            seed: KSampler seed
            steps: KSampler steps
            cfg_scale: CFG
            denoise: Denoise (1.0 = full replace in masked latent noise)
            model: UNET filename
            vae: VAE filename
            clip1: T5-XXL filename
            clip2: CLIP-L filename
            filename_prefix: SaveImage prefix

        Returns:
            Configured ComfyWorkflow
        """
        wf = ComfyWorkflow("flux_inpaint")

        clip = wf.add_node(
            "DualCLIPLoader",
            {"clip_name1": clip1, "clip_name2": clip2, "type": "flux"},
            title="Load Dual CLIP",
        )

        unet = wf.add_node(
            "UNETLoader",
            {"unet_name": model, "weight_dtype": "default"},
            title="Load FLUX UNET",
        )

        vae_node = wf.add_node(
            "VAELoader", {"vae_name": vae}, title="Load FLUX VAE"
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
            {"pixels": img_ref, "vae": vae_node.get_output_ref()},
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

        positive = wf.add_node(
            "CLIPTextEncode",
            {"text": prompt, "clip": clip.get_output_ref()},
            title="Positive Prompt",
        )
        negative = _flux_empty_negative(wf, clip.get_output_ref())

        model_sampling = wf.add_node(
            "ModelSamplingFlux",
            {
                "max_shift": 1.15,
                "base_shift": 0.5,
                "width": width if width > 0 else 1024,
                "height": height if height > 0 else 1024,
                "model": unet.get_output_ref(),
            },
            title="ModelSamplingFlux",
        )

        sampler = wf.add_node(
            "KSampler",
            {
                "seed": seed,
                "steps": steps,
                "cfg": cfg_scale,
                "sampler_name": "euler",
                "scheduler": "simple",
                "denoise": denoise,
                "model": model_sampling.get_output_ref(),
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
                "vae": vae_node.get_output_ref(),
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
