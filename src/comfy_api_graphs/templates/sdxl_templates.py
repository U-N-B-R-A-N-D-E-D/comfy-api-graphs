"""
SDXL workflow templates (community-safe patterns).

Stable Diffusion XL typically loads via CheckpointLoaderSimple (model + CLIP
+ VAE in one checkpoint). Optional separate VAELoader and a single LoRA stack
hop are included. Optional refiner is a second CheckpointLoaderSimple +
second KSampler pass — a common teaching pattern, not a studio secret.

Requires locally: SDXL base checkpoint under ``models/checkpoints/``,
optional VAE / LoRA / refiner files. Graph slang: checkpoint, CLIP, VAE,
KSampler, latent, denoise, CFG, SaveImage, API format.
"""

from typing import Optional

from ..core import ComfyWorkflow


class SdxlTxt2ImgTemplate:
    """
    SDXL text-to-image template.

    EmptyLatentImage → CLIPTextEncode (pos/neg) → KSampler → VAEDecode →
    SaveImage. With ``refiner_model``, base KSampler uses denoise < 1.0 and
    a second KSampler finishes the latent (simple two-pass teaching graph).

    Example:
        >>> workflow = SdxlTxt2ImgTemplate.create(
        ...     prompt="professional photo of an astronaut",
        ...     negative_prompt="cartoon, drawing",
        ...     width=1024,
        ...     height=1024,
        ...     refiner_model="sd_xl_refiner_1.0.safetensors",
        ... )
    """

    @staticmethod
    def create(
        prompt: str,
        negative_prompt: str = "",
        width: int = 1024,
        height: int = 1024,
        seed: int = -1,
        steps: int = 30,
        cfg_scale: float = 7.0,
        base_model: str = "sd_xl_base_1.0.safetensors",
        refiner_model: Optional[str] = None,
        vae: Optional[str] = "sdxl_vae.safetensors",
        lora: Optional[str] = None,
        lora_strength: float = 1.0,
        filename_prefix: str = "sdxl_",
    ) -> ComfyWorkflow:
        """
        Build an SDXL text-to-image graph.

        Args:
            prompt: Positive prompt
            negative_prompt: Negative prompt (CLIPTextEncode)
            width: EmptyLatentImage width (1024-class resolutions work best)
            height: EmptyLatentImage height
            seed: KSampler seed
            steps: Base KSampler steps
            cfg_scale: CFG (often 5–8 for SDXL)
            base_model: Checkpoint filename
            refiner_model: Optional refiner checkpoint filename
            vae: Optional VAE override; None uses checkpoint VAE output
            lora: Optional LoRA filename
            lora_strength: LoRA strength on model and CLIP
            filename_prefix: SaveImage prefix

        Returns:
            Configured ComfyWorkflow
        """
        wf = ComfyWorkflow("sdxl_txt2img")

        base = wf.add_node(
            "CheckpointLoaderSimple",
            {"ckpt_name": base_model},
            title="Load Checkpoint",
        )

        if vae:
            vae_node = wf.add_node(
                "VAELoader", {"vae_name": vae}, title="Load VAE"
            )
            vae_ref = vae_node.get_output_ref()
        else:
            vae_ref = base.get_output_ref(2)

        if lora:
            lora_node = wf.add_node(
                "LoraLoader",
                {
                    "lora_name": lora,
                    "strength_model": lora_strength,
                    "strength_clip": lora_strength,
                    "model": base.get_output_ref(),
                    "clip": base.get_output_ref(1),
                },
                title="LoRA Stack",
            )
            model_ref = lora_node.get_output_ref()
            clip_ref = lora_node.get_output_ref(1)
        else:
            model_ref = base.get_output_ref()
            clip_ref = base.get_output_ref(1)

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

        latent = wf.add_node(
            "EmptyLatentImage",
            {"width": width, "height": height, "batch_size": 1},
            title="Empty Latent",
        )

        if refiner_model:
            # Teaching split: base does most of the denoise, refiner finishes.
            # Real pipelines may use different step/denoise splits — tune in UI.
            base_sampler = wf.add_node(
                "KSampler",
                {
                    "seed": seed,
                    "steps": steps,
                    "cfg": cfg_scale,
                    "sampler_name": "euler",
                    "scheduler": "normal",
                    "denoise": 0.8,
                    "model": model_ref,
                    "positive": positive.get_output_ref(),
                    "negative": negative.get_output_ref(),
                    "latent_image": latent.get_output_ref(),
                },
                title="Base KSampler",
            )

            refiner = wf.add_node(
                "CheckpointLoaderSimple",
                {"ckpt_name": refiner_model},
                title="Load Refiner Checkpoint",
            )

            refiner_positive = wf.add_node(
                "CLIPTextEncode",
                {"text": prompt, "clip": refiner.get_output_ref(1)},
                title="Refiner Positive",
            )
            refiner_negative = wf.add_node(
                "CLIPTextEncode",
                {"text": negative_prompt, "clip": refiner.get_output_ref(1)},
                title="Refiner Negative",
            )

            refiner_sampler = wf.add_node(
                "KSampler",
                {
                    "seed": seed,
                    "steps": max(steps // 2, 1),
                    "cfg": cfg_scale,
                    "sampler_name": "euler",
                    "scheduler": "normal",
                    "denoise": 0.2,
                    "model": refiner.get_output_ref(),
                    "positive": refiner_positive.get_output_ref(),
                    "negative": refiner_negative.get_output_ref(),
                    "latent_image": base_sampler.get_output_ref(),
                },
                title="Refiner KSampler",
            )
            final_latent = refiner_sampler.get_output_ref()
        else:
            sampler = wf.add_node(
                "KSampler",
                {
                    "seed": seed,
                    "steps": steps,
                    "cfg": cfg_scale,
                    "sampler_name": "dpmpp_2m",
                    "scheduler": "karras",
                    "denoise": 1.0,
                    "model": model_ref,
                    "positive": positive.get_output_ref(),
                    "negative": negative.get_output_ref(),
                    "latent_image": latent.get_output_ref(),
                },
                title="KSampler",
            )
            final_latent = sampler.get_output_ref()

        decoded = wf.add_node(
            "VAEDecode",
            {"samples": final_latent, "vae": vae_ref},
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


class SdxlImg2ImgTemplate:
    """
    SDXL image-to-image template.

    LoadImage → ImageScale → VAEEncode → KSampler(denoise) → VAEDecode →
    SaveImage.

    Example:
        >>> workflow = SdxlImg2ImgTemplate.create(
        ...     prompt="convert to oil painting style",
        ...     image_path="photo.png",
        ...     denoise=0.65,
        ... )
    """

    @staticmethod
    def create(
        prompt: str,
        image_path: str,
        negative_prompt: str = "",
        width: int = 1024,
        height: int = 1024,
        seed: int = -1,
        steps: int = 30,
        cfg_scale: float = 7.0,
        denoise: float = 0.65,
        model: str = "sd_xl_base_1.0.safetensors",
        vae: Optional[str] = "sdxl_vae.safetensors",
        lora: Optional[str] = None,
        lora_strength: float = 1.0,
        filename_prefix: str = "sdxl_i2i_",
    ) -> ComfyWorkflow:
        """
        Build an SDXL image-to-image graph.

        Args:
            prompt: Transformation prompt
            image_path: LoadImage path under ComfyUI ``input/``
            negative_prompt: Negative prompt
            width: ImageScale width (0 skips resize)
            height: ImageScale height (0 skips resize)
            seed: KSampler seed
            steps: KSampler steps
            cfg_scale: CFG
            denoise: Denoise strength (0.0–1.0)
            model: Checkpoint filename
            vae: Optional VAE override
            lora: Optional LoRA filename
            lora_strength: LoRA strength
            filename_prefix: SaveImage prefix

        Returns:
            Configured ComfyWorkflow
        """
        wf = ComfyWorkflow("sdxl_img2img")

        checkpoint = wf.add_node(
            "CheckpointLoaderSimple",
            {"ckpt_name": model},
            title="Load Checkpoint",
        )

        if vae:
            vae_node = wf.add_node(
                "VAELoader", {"vae_name": vae}, title="Load VAE"
            )
            vae_ref = vae_node.get_output_ref()
        else:
            vae_ref = checkpoint.get_output_ref(2)

        if lora:
            lora_node = wf.add_node(
                "LoraLoader",
                {
                    "lora_name": lora,
                    "strength_model": lora_strength,
                    "strength_clip": lora_strength,
                    "model": checkpoint.get_output_ref(),
                    "clip": checkpoint.get_output_ref(1),
                },
                title="LoRA Stack",
            )
            model_ref = lora_node.get_output_ref()
            clip_ref = lora_node.get_output_ref(1)
        else:
            model_ref = checkpoint.get_output_ref()
            clip_ref = checkpoint.get_output_ref(1)

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
            {"pixels": image_ref, "vae": vae_ref},
            title="VAEEncode",
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
                "model": model_ref,
                "positive": positive.get_output_ref(),
                "negative": negative.get_output_ref(),
                "latent_image": encoded.get_output_ref(),
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
