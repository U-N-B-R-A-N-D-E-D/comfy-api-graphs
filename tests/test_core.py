"""
Tests for core workflow building functionality.
"""

import pytest
from comfy_api_graphs import ComfyWorkflow, WorkflowNode
from comfy_api_graphs import (
    validate_node_references,
    estimate_vram_usage,
    validate_workflow_complete,
)


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

    def test_to_graph_format(self):
        """Test conversion to graph format."""
        wf = ComfyWorkflow()

        wf.add_node("CheckpointLoaderSimple", {}, node_id="1")
        wf.add_node("KSampler", {}, node_id="2")

        graph_format = wf.to_graph_format()

        assert "nodes" in graph_format
        assert "links" in graph_format
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

    def test_estimate_vram_usage(self):
        """Test VRAM estimation."""
        wf = ComfyWorkflow()

        wf.add_node(
            "CheckpointLoaderSimple",
            {"ckpt_name": "model.safetensors"},
        )
        wf.add_node("VAELoader", {"vae_name": "vae.safetensors"})

        estimate = estimate_vram_usage(wf)

        assert "estimated_vram_gb" in estimate
        assert estimate["estimated_vram_gb"] > 0
        assert "model_loaders" in estimate
        assert estimate["model_loaders"] == 1  # Just the checkpoint
        assert "recommendation" in estimate

    def test_estimate_vram_flux(self):
        """Test VRAM estimation recognizes FLUX models."""
        wf = ComfyWorkflow()

        wf.add_node(
            "CheckpointLoaderSimple",
            {"ckpt_name": "flux1-dev.safetensors"},
        )

        estimate = estimate_vram_usage(wf)

        # FLUX should have higher VRAM estimate
        assert estimate["estimated_vram_gb"] > 20

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
