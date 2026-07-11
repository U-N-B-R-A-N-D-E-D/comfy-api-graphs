"""
Tests for workflow templates (public gold-mine set).
"""

import pytest
from comfy_api_graphs import (
    FluxTxt2ImgTemplate,
    FluxImg2ImgTemplate,
    FluxInpaintTemplate,
    SdxlTxt2ImgTemplate,
    SdxlImg2ImgTemplate,
    CharacterReferenceTemplate,
    UpscaleTemplate,
    InpaintTemplate,
    OutpaintTemplate,
    validate_node_references,
)


class TestFluxTemplates:
    """Tests for FLUX workflow templates."""

    def test_flux_txt2img_basic(self):
        workflow = FluxTxt2ImgTemplate.create(
            prompt="a serene mountain landscape",
            width=1024,
            height=1024,
        )
        assert workflow.name == "flux_txt2img"
        assert len(workflow.nodes) > 0
        assert validate_node_references(workflow) == []

        # Empty negative must be a real CLIPTextEncode, not a fake ref
        negatives = [
            n
            for n in workflow.nodes.values()
            if n.class_type == "CLIPTextEncode" and n.inputs.get("text") == ""
        ]
        assert len(negatives) == 1

    def test_flux_txt2img_with_lora(self):
        workflow = FluxTxt2ImgTemplate.create(
            prompt="anime style character",
            width=1024,
            height=1024,
            lora="anime_lora.safetensors",
            lora_strength=0.8,
        )
        lora_nodes = [
            n for n in workflow.nodes.values() if n.class_type == "LoraLoader"
        ]
        assert len(lora_nodes) == 1

    def test_flux_img2img_basic(self):
        workflow = FluxImg2ImgTemplate.create(
            prompt="convert to watercolor style",
            image_path="image.png",
            width=1024,
            height=1024,
            denoise=0.75,
        )
        assert workflow.name == "flux_img2img"
        load_nodes = [
            n for n in workflow.nodes.values() if n.class_type == "LoadImage"
        ]
        assert len(load_nodes) == 1
        assert validate_node_references(workflow) == []

    def test_flux_inpaint_uses_image_to_mask(self):
        workflow = FluxInpaintTemplate.create(
            prompt="a red apple",
            image_path="scene.png",
            mask_path="mask.png",
        )
        assert any(
            n.class_type == "SetLatentNoiseMask" for n in workflow.nodes.values()
        )
        assert any(n.class_type == "ImageToMask" for n in workflow.nodes.values())
        assert validate_node_references(workflow) == []


class TestSdxlTemplates:
    """Tests for SDXL workflow templates."""

    def test_sdxl_txt2img_basic(self):
        workflow = SdxlTxt2ImgTemplate.create(
            prompt="professional photo of an astronaut",
            negative_prompt="cartoon, anime",
            width=1024,
            height=1024,
        )
        assert workflow.name == "sdxl_txt2img"
        assert validate_node_references(workflow) == []

    def test_sdxl_txt2img_with_refiner(self):
        workflow = SdxlTxt2ImgTemplate.create(
            prompt="highly detailed portrait",
            width=1024,
            height=1024,
            refiner_model="sd_xl_refiner_1.0.safetensors",
        )
        checkpoint_nodes = [
            n
            for n in workflow.nodes.values()
            if n.class_type == "CheckpointLoaderSimple"
        ]
        assert len(checkpoint_nodes) == 2
        assert validate_node_references(workflow) == []

    def test_sdxl_img2img(self):
        workflow = SdxlImg2ImgTemplate.create(
            prompt="oil painting",
            image_path="photo.png",
            denoise=0.65,
        )
        assert workflow.name == "sdxl_img2img"
        assert validate_node_references(workflow) == []


class TestUtilityTemplates:
    """Tests for utility workflow templates."""

    def test_character_reference_batch(self):
        workflows = CharacterReferenceTemplate.create_batch(
            appearance="young woman with brown hair",
            style="wearing casual clothes",
            base_seed=42,
        )
        assert len(workflows) > 0
        for wf in workflows:
            assert validate_node_references(wf) == []

    def test_character_reference_single(self):
        workflow = CharacterReferenceTemplate.create_single(
            appearance="man with beard",
            angle="front",
        )
        assert workflow.name.startswith("character_ref")

    def test_character_reference_filter_angles(self):
        workflows = CharacterReferenceTemplate.create_batch(
            appearance="test character",
            angles=["front", "profile"],
        )
        assert len(workflows) == 2

    def test_upscale_template(self):
        workflow = UpscaleTemplate.create(
            image_path="image.png",
            scale=2.0,
            upscaler="4x-UltraSharp.pth",
        )
        assert workflow.name == "upscale"
        assert any(
            n.class_type == "UpscaleModelLoader" for n in workflow.nodes.values()
        )
        assert validate_node_references(workflow) == []

    def test_upscale_latent_mode(self):
        workflow = UpscaleTemplate.create(
            image_path="image.png",
            use_latent_upscale=True,
            scale=2.0,
        )
        assert any(
            n.class_type == "LatentUpscaleBy" for n in workflow.nodes.values()
        )
        assert validate_node_references(workflow) == []

    def test_inpaint_template(self):
        workflow = InpaintTemplate.create(
            prompt="a red rose",
            image_path="scene.png",
            mask_path="mask.png",
        )
        assert validate_node_references(workflow) == []

    def test_outpaint_template(self):
        workflow = OutpaintTemplate.create(
            prompt="continue the landscape",
            image_path="forest.png",
            expand_left=128,
            expand_right=128,
        )
        assert any(
            n.class_type == "ImagePadForOutpaint" for n in workflow.nodes.values()
        )
        assert any(
            n.class_type == "SetLatentNoiseMask" for n in workflow.nodes.values()
        )
        assert validate_node_references(workflow) == []

    def test_outpaint_requires_expansion(self):
        with pytest.raises(ValueError):
            OutpaintTemplate.create(
                prompt="noop",
                image_path="x.png",
            )


class TestVideoTemplatesQuarantined:
    """Video stubs exist but are not part of the public package API."""

    def test_not_exported_from_package(self):
        import comfy_api_graphs as pkg

        assert not hasattr(pkg, "LtxVideoTemplate")
        assert not hasattr(pkg, "WanVideoTemplate")

    def test_stubs_raise(self):
        from comfy_api_graphs.templates import video_templates as vt

        with pytest.raises(NotImplementedError):
            vt.LtxVideoTemplate.create(prompt="x")
        with pytest.raises(NotImplementedError):
            vt.WanVideoTemplate.create(prompt="x")


class TestTemplateApiFormat:
    """Tests for template API format output."""

    def test_flux_api_format_structure(self):
        workflow = FluxTxt2ImgTemplate.create(
            prompt="test prompt",
            width=512,
            height=512,
        )
        api_format = workflow.to_api_format()
        assert isinstance(api_format, dict)
        assert len(api_format) > 0
        for node_data in api_format.values():
            assert "class_type" in node_data
            assert "inputs" in node_data

    def test_all_public_templates_validate(self):
        cases = [
            FluxTxt2ImgTemplate.create(prompt="test"),
            SdxlTxt2ImgTemplate.create(prompt="test"),
            FluxImg2ImgTemplate.create(prompt="test", image_path="a.png"),
            SdxlImg2ImgTemplate.create(prompt="test", image_path="a.png"),
            UpscaleTemplate.create(image_path="a.png"),
            InpaintTemplate.create(
                prompt="test", image_path="a.png", mask_path="m.png"
            ),
            OutpaintTemplate.create(
                prompt="test", image_path="a.png", expand_right=64
            ),
        ]
        for workflow in cases:
            errors = validate_node_references(workflow)
            assert errors == [], f"{workflow.name}: {errors}"
