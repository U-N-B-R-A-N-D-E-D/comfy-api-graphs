"""
Tests for core workflow building functionality.
"""

import json
import random
from pathlib import Path

import pytest
from comfy_api_graphs import ComfyWorkflow, WorkflowNode
from comfy_api_graphs import (
    validate_node_references,
    load_object_info,
    validate_against_object_info,
    estimate_vram_usage,
    validate_workflow_complete,
)

# Minimal /object_info slice for schema tests (no new fixture file).
MINIMAL_OBJECT_INFO = {
    "CheckpointLoaderSimple": {
        "input": {
            "required": {"ckpt_name": [["model.safetensors", "other.safetensors"]]},
        },
        "output": ["MODEL", "CLIP", "VAE"],
        "output_name": ["MODEL", "CLIP", "VAE"],
    },
    "CLIPTextEncode": {
        "input": {
            "required": {
                "text": ["STRING", {"multiline": True}],
                "clip": ["CLIP"],
            },
        },
        "output": ["CONDITIONING"],
        "output_name": ["CONDITIONING"],
    },
    "EmptyLatentImage": {
        "input": {
            "required": {
                "width": ["INT", {"default": 512}],
                "height": ["INT", {"default": 512}],
                "batch_size": ["INT", {"default": 1}],
            },
        },
        "output": ["LATENT"],
        "output_name": ["LATENT"],
    },
    "KSampler": {
        "input": {
            "required": {
                "model": ["MODEL"],
                "seed": ["INT", {"default": 0}],
                "steps": ["INT", {"default": 20}],
                "cfg": ["FLOAT", {"default": 8.0}],
                "sampler_name": [["euler", "euler_ancestral"]],
                "scheduler": [["normal", "karras"]],
                "positive": ["CONDITIONING"],
                "negative": ["CONDITIONING"],
                "latent_image": ["LATENT"],
                "denoise": ["FLOAT", {"default": 1.0}],
            },
        },
        "output": ["LATENT"],
        "output_name": ["LATENT"],
    },
    "VAEDecode": {
        "input": {
            "required": {
                "samples": ["LATENT"],
                "vae": ["VAE"],
            },
        },
        "output": ["IMAGE"],
        "output_name": ["IMAGE"],
    },
    "SaveImage": {
        "input": {
            "required": {
                "images": ["IMAGE"],
                "filename_prefix": ["STRING", {"default": "ComfyUI"}],
            },
        },
        "output": [],
        "output_name": [],
    },
    "VAELoader": {
        "input": {
            "required": {"vae_name": [["vae.safetensors"]]},
        },
        "output": ["VAE"],
        "output_name": ["VAE"],
    },
}

FIXTURES = Path(__file__).parent / "fixtures"
MINIMAL_API = FIXTURES / "minimal_api_graph.json"


class TestWorkflowNode:
    """Tests for WorkflowNode class."""

    def test_node_creation(self):
        """Test basic node creation."""
        node = WorkflowNode(
            node_id="1",
            class_type="CheckpointLoaderSimple",
            inputs={"ckpt_name": "model.safetensors"},
            title="Load Model",
        )

        assert node.node_id == "1"
        assert node.class_type == "CheckpointLoaderSimple"
        assert node.inputs["ckpt_name"] == "model.safetensors"
        assert node.title == "Load Model"

    def test_to_api_dict(self):
        """Test conversion to API dictionary format."""
        node = WorkflowNode(
            node_id="5",
            class_type="KSampler",
            inputs={"seed": 42, "steps": 20},
        )

        api_dict = node.to_api_dict()

        assert api_dict["class_type"] == "KSampler"
        assert api_dict["inputs"]["seed"] == 42
        assert api_dict["inputs"]["steps"] == 20
        assert "_meta" not in api_dict  # No title

    def test_to_api_dict_with_title(self):
        """Test API dict includes title when set."""
        node = WorkflowNode(
            node_id="1",
            class_type="SaveImage",
            inputs={},
            title="Save Output",
        )

        api_dict = node.to_api_dict()

        assert api_dict["_meta"]["title"] == "Save Output"

    def test_get_output_ref(self):
        """Test output reference generation."""
        node = WorkflowNode(
            node_id="10",
            class_type="CheckpointLoaderSimple",
            inputs={},
        )

        # Default output (index 0)
        ref0 = node.get_output_ref()
        assert ref0 == ["10", 0]

        # Model output (index 0)
        ref_model = node.get_output_ref(0)
        assert ref_model == ["10", 0]

        # CLIP output (index 1)
        ref_clip = node.get_output_ref(1)
        assert ref_clip == ["10", 1]


class TestComfyWorkflow:
    """Tests for ComfyWorkflow class."""

    def test_workflow_creation(self):
        """Test basic workflow creation."""
        wf = ComfyWorkflow("test_workflow")

        assert wf.name == "test_workflow"
        assert len(wf.nodes) == 0

    def test_add_node_auto_id(self):
        """Test node addition with auto-generated ID."""
        wf = ComfyWorkflow()

        node1 = wf.add_node("CheckpointLoaderSimple", {})
        node2 = wf.add_node("KSampler", {})

        assert node1.node_id == "1"
        assert node2.node_id == "2"

    def test_add_node_custom_id(self):
        """Test node addition with custom ID."""
        wf = ComfyWorkflow()

        node = wf.add_node("SaveImage", {}, node_id="output_1")

        assert node.node_id == "output_1"

    def test_add_node_with_connections(self):
        """Test adding nodes with input connections."""
        wf = ComfyWorkflow()

        # Add checkpoint loader
        checkpoint = wf.add_node(
            "CheckpointLoaderSimple",
            {"ckpt_name": "model.safetensors"},
            node_id="1",
        )

        # Add sampler using checkpoint output
        sampler = wf.add_node(
            "KSampler",
            {
                "model": checkpoint.get_output_ref(),
                "seed": 42,
            },
            node_id="2",
        )

        assert sampler.inputs["model"] == ["1", 0]

    def test_get_node(self):
        """Test retrieving nodes by ID."""
        wf = ComfyWorkflow()

        wf.add_node("CheckpointLoaderSimple", {}, node_id="loader")

        node = wf.get_node("loader")
        assert node is not None
        assert node.class_type == "CheckpointLoaderSimple"

        missing = wf.get_node("nonexistent")
        assert missing is None

    def test_to_api_format(self):
        """Test conversion to ComfyUI API format."""
        wf = ComfyWorkflow()

        wf.add_node(
            "CheckpointLoaderSimple",
            {"ckpt_name": "model.safetensors"},
            node_id="1",
        )

        api_format = wf.to_api_format()

        assert "1" in api_format
        assert api_format["1"]["class_type"] == "CheckpointLoaderSimple"
        assert api_format["1"]["inputs"]["ckpt_name"] == "model.safetensors"

    def test_to_graph_format_deprecated(self):
        """Skeleton export still works but warns — not a UI converter."""
        wf = ComfyWorkflow()

        wf.add_node("CheckpointLoaderSimple", {}, node_id="1")
        wf.add_node("KSampler", {}, node_id="2")

        with pytest.warns(DeprecationWarning, match="to_graph_format"):
            graph_format = wf.to_graph_format()

        assert "nodes" in graph_format
        assert graph_format["links"] == []
        assert len(graph_format["nodes"]) == 2

    def test_workflow_length(self):
        """Test __len__ method."""
        wf = ComfyWorkflow()

        assert len(wf) == 0

        wf.add_node("Node1", {})
        wf.add_node("Node2", {})

        assert len(wf) == 2

    def test_workflow_repr(self):
        """Test string representation."""
        wf = ComfyWorkflow("my_workflow")
        wf.add_node("TestNode", {})

        repr_str = repr(wf)

        assert "my_workflow" in repr_str
        assert "1" in repr_str  # node count


class TestValidation:
    """Tests for validation functions."""

    def test_validate_node_references_valid(self):
        """Test validation with all valid references."""
        wf = ComfyWorkflow()

        node1 = wf.add_node("CheckpointLoaderSimple", {}, node_id="1")
        wf.add_node(
            "KSampler",
            {"model": node1.get_output_ref()},
            node_id="2",
        )

        errors = validate_node_references(wf)

        assert len(errors) == 0

    def test_validate_node_references_invalid(self):
        """Test validation with broken references."""
        wf = ComfyWorkflow()

        # Reference a non-existent node
        wf.add_node(
            "KSampler",
            {"model": ["nonexistent", 0]},
            node_id="2",
        )

        errors = validate_node_references(wf)

        assert len(errors) == 1
        assert "nonexistent" in errors[0]

    def test_estimate_vram_usage_is_planning_hint(self):
        """VRAM helper is explicitly a non-predictive planning hint."""
        wf = ComfyWorkflow()

        wf.add_node(
            "CheckpointLoaderSimple",
            {"ckpt_name": "model.safetensors"},
        )
        wf.add_node("VAELoader", {"vae_name": "vae.safetensors"})

        estimate = estimate_vram_usage(wf)

        assert estimate["is_estimate"] is True
        assert "PLANNING HINT" in estimate["disclaimer"]
        assert "estimated_vram_gb" in estimate
        assert estimate["estimated_vram_gb"] > 0
        assert estimate["model_loaders"] == 1  # Just the checkpoint
        assert "Hint only" in estimate["recommendation"]

    def test_estimate_vram_flux_hint(self):
        """FLUX ckpt name bumps the hint number — still not a predictor."""
        wf = ComfyWorkflow()

        wf.add_node(
            "CheckpointLoaderSimple",
            {"ckpt_name": "flux1-dev.safetensors"},
        )

        estimate = estimate_vram_usage(wf)

        assert estimate["is_estimate"] is True
        assert estimate["estimated_vram_gb"] > 20

    def test_validate_against_object_info_ok(self):
        """Minimal API graph passes against a matching object_info slice."""
        wf = ComfyWorkflow.from_api_json(MINIMAL_API)
        info = load_object_info(MINIMAL_OBJECT_INFO)

        report = validate_against_object_info(wf, info)

        assert report["valid"] is True
        assert report["errors"] == []

    def test_validate_against_object_info_unknown_class(self):
        """Missing custom node class_type fails before queue."""
        wf = ComfyWorkflow()
        wf.add_node("TotallyFakeNode", {"x": 1}, node_id="1")

        report = validate_against_object_info(wf, MINIMAL_OBJECT_INFO)

        assert report["valid"] is False
        assert any("TotallyFakeNode" in e for e in report["errors"])

    def test_validate_against_object_info_missing_required(self):
        """Missing required widget fails schema check."""
        wf = ComfyWorkflow()
        wf.add_node("SaveImage", {"images": ["0", 0]}, node_id="1")  # no prefix

        report = validate_against_object_info(wf, MINIMAL_OBJECT_INFO)

        assert report["valid"] is False
        assert any("filename_prefix" in e for e in report["errors"])

    def test_validate_against_object_info_bad_output_index(self):
        """Output slot past declared outputs fails."""
        wf = ComfyWorkflow()
        ckpt = wf.add_node(
            "CheckpointLoaderSimple",
            {"ckpt_name": "model.safetensors"},
            node_id="1",
        )
        wf.add_node(
            "CLIPTextEncode",
            {"text": "hi", "clip": ckpt.get_output_ref(99)},
            node_id="2",
        )

        report = validate_against_object_info(wf, MINIMAL_OBJECT_INFO)

        assert report["valid"] is False
        assert any("out of range" in e for e in report["errors"])

    def test_validate_against_object_info_unknown_input_warns(self):
        """Extra input keys warn by default; can escalate to errors."""
        wf = ComfyWorkflow()
        wf.add_node(
            "SaveImage",
            {
                "images": ["0", 0],
                "filename_prefix": "x",
                "totally_extra": 1,
            },
            node_id="1",
        )

        soft = validate_against_object_info(wf, MINIMAL_OBJECT_INFO)
        assert soft["valid"] is True
        assert any("totally_extra" in w for w in soft["warnings"])

        hard = validate_against_object_info(
            wf, MINIMAL_OBJECT_INFO, unknown_inputs_as_errors=True
        )
        assert hard["valid"] is False
        assert any("totally_extra" in e for e in hard["errors"])

    def test_load_object_info_rejects_bad_shape(self):
        with pytest.raises(ValueError, match="object_info"):
            load_object_info({"KSampler": {"not_input": True}})

    def test_validate_workflow_complete_valid(self):
        """Complete graph with loader + SaveImage passes."""
        wf = ComfyWorkflow()
        ckpt = wf.add_node("CheckpointLoaderSimple", {})
        decode = wf.add_node(
            "VAEDecode",
            {"vae": ckpt.get_output_ref(2), "samples": ["0", 0]},
        )
        wf.add_node("SaveImage", {"images": decode.get_output_ref()})

        result = validate_workflow_complete(wf)

        assert result["valid"] is True
        assert result["errors"] == []
        assert result["node_count"] == 3
        assert any("object_info" in w for w in result["warnings"])

    def test_validate_workflow_complete_broken_link(self):
        """Broken socket link marks valid=False."""
        wf = ComfyWorkflow()
        wf.add_node("KSampler", {"model": ["missing", 0]})

        result = validate_workflow_complete(wf)

        assert result["valid"] is False
        assert any("missing" in e for e in result["errors"])
        assert any("output node" in w.lower() for w in result["warnings"])


class TestComplexWorkflows:
    """Tests for more complex workflow scenarios."""

    def test_linear_pipeline(self):
        """Test creating a linear processing pipeline."""
        wf = ComfyWorkflow("linear_pipeline")

        # Loader -> Encoder -> Sampler -> Decoder -> Save
        loader = wf.add_node("CheckpointLoaderSimple", {}, node_id="1")
        encoder = wf.add_node(
            "CLIPTextEncode",
            {"clip": loader.get_output_ref(1)},
            node_id="2",
        )
        latent = wf.add_node("EmptyLatentImage", {}, node_id="3")
        sampler = wf.add_node(
            "KSampler",
            {
                "model": loader.get_output_ref(),
                "positive": encoder.get_output_ref(),
                "latent_image": latent.get_output_ref(),
            },
            node_id="4",
        )
        decoder = wf.add_node(
            "VAEDecode",
            {"samples": sampler.get_output_ref(), "vae": loader.get_output_ref(2)},
            node_id="5",
        )
        wf.add_node(
            "SaveImage",
            {"images": decoder.get_output_ref()},
            node_id="6",
        )

        assert len(wf) == 6

        # Validate all references
        errors = validate_node_references(wf)
        assert len(errors) == 0

    def test_multi_output_node(self):
        """Test nodes with multiple outputs."""
        wf = ComfyWorkflow()

        # Checkpoint produces model, clip, and vae outputs
        checkpoint = wf.add_node(
            "CheckpointLoaderSimple",
            {"ckpt_name": "model.safetensors"},
            node_id="ckpt",
        )

        # Use different outputs
        model_sampler = wf.add_node(
            "KSampler",
            {"model": checkpoint.get_output_ref(0)},  # Model output
            node_id="sampler",
        )
        text_encoder = wf.add_node(
            "CLIPTextEncode",
            {"clip": checkpoint.get_output_ref(1)},  # CLIP output
            node_id="encoder",
        )
        vae_decoder = wf.add_node(
            "VAEDecode",
            {
                "samples": model_sampler.get_output_ref(),
                "vae": checkpoint.get_output_ref(2),  # VAE output
            },
            node_id="decoder",
        )

        assert model_sampler.inputs["model"] == ["ckpt", 0]
        assert text_encoder.inputs["clip"] == ["ckpt", 1]
        assert vae_decoder.inputs["vae"] == ["ckpt", 2]


class TestLoadAndMutate:
    """v0.2: from_api_json + seed / input mutators."""

    def test_from_api_json_path(self):
        wf = ComfyWorkflow.from_api_json(MINIMAL_API)

        assert wf.name == "minimal_api_graph"
        assert len(wf) == 6
        assert wf.get_node("4").class_type == "KSampler"
        assert wf.get_node("4").inputs["seed"] == 42
        assert wf.get_node("1").title == "Load Checkpoint"
        assert validate_node_references(wf) == []

    def test_from_api_dict_prompt_wrapper(self):
        raw = json.loads(MINIMAL_API.read_text(encoding="utf-8"))
        wrapped = {"prompt": raw, "client_id": "test"}
        wf = ComfyWorkflow.from_api_dict(wrapped, name="wrapped")

        assert wf.name == "wrapped"
        assert len(wf) == 6

    def test_from_api_json_rejects_ui_shaped(self):
        ui_shaped = {"nodes": [], "links": [], "version": 0.4}
        with pytest.raises(ValueError, match="API-format"):
            ComfyWorkflow.from_api_dict(ui_shaped)

    def test_set_input_by_id_and_class_type(self):
        wf = ComfyWorkflow.from_api_json(MINIMAL_API)

        wf.set_input("2", "text", "a dog")
        assert wf.get_node("2").inputs["text"] == "a dog"

        n = wf.set_input_by_class_type("KSampler", "steps", 30)
        assert n == 1
        assert wf.get_node("4").inputs["steps"] == 30

        n = wf.set_input_by_title("Positive", "text", "a bird")
        assert n == 1
        assert wf.get_node("2").inputs["text"] == "a bird"

    def test_set_input_missing_node(self):
        wf = ComfyWorkflow.from_api_json(MINIMAL_API)
        with pytest.raises(KeyError):
            wf.set_input("999", "seed", 1)

    def test_randomize_seeds_deterministic(self):
        wf = ComfyWorkflow.from_api_json(MINIMAL_API)
        updated = wf.randomize_seeds(rng=random.Random(0))

        assert updated == ["4"]
        assert wf.get_node("4").inputs["seed"] != 42
        assert isinstance(wf.get_node("4").inputs["seed"], int)

    def test_bump_seeds(self):
        wf = ComfyWorkflow.from_api_json(MINIMAL_API)
        updated = wf.bump_seeds(5)

        assert updated == ["4"]
        assert wf.get_node("4").inputs["seed"] == 47

    def test_roundtrip_save_load(self, tmp_path):
        wf = ComfyWorkflow.from_api_json(MINIMAL_API)
        wf.bump_seeds(1)
        out = tmp_path / "roundtrip.json"
        wf.save_api_json(str(out))

        loaded = ComfyWorkflow.from_api_json(out)
        assert loaded.get_node("4").inputs["seed"] == 43
        assert loaded.to_api_format()["4"]["inputs"]["seed"] == 43

    def test_add_node_after_load_avoids_id_collision(self):
        wf = ComfyWorkflow.from_api_json(MINIMAL_API)
        new = wf.add_node("PreviewImage", {"images": ["5", 0]})
        assert new.node_id == "7"
        assert "7" in wf.nodes
