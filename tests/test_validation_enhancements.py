"""
Tests for v0.4 validation enhancements.

Covers: link/scalar type-sanity warnings, orphan-node detection,
duplicate-seed-across-batch detection, and the new warn_orphans flag on
validate_against_object_info. Pure-Python, offline.
"""

from comfy_api_graphs import (
    ComfyWorkflow,
    find_duplicate_seeds_across_batch,
    find_orphan_nodes,
    validate_against_object_info,
)
from tests.test_core import MINIMAL_OBJECT_INFO


def _sampler_graph() -> ComfyWorkflow:
    wf = ComfyWorkflow("g")
    ckpt = wf.add_node("CheckpointLoaderSimple", {"ckpt_name": "m.safetensors"}, node_id="1")
    enc = wf.add_node("CLIPTextEncode", {"text": "x", "clip": ckpt.get_output_ref(1)}, node_id="2")
    latent = wf.add_node("EmptyLatentImage", {}, node_id="3")
    sampler = wf.add_node(
        "KSampler",
        {
            "model": ckpt.get_output_ref(0),
            "positive": enc.get_output_ref(0),
            "negative": enc.get_output_ref(0),
            "latent_image": latent.get_output_ref(0),
            "seed": 42,
            "steps": 20,
            "cfg": 7.0,
            "sampler_name": "euler",
            "scheduler": "normal",
            "denoise": 1.0,
        },
        node_id="4",
    )
    decode = wf.add_node(
        "VAEDecode",
        {"samples": sampler.get_output_ref(0), "vae": ckpt.get_output_ref(2)},
        node_id="5",
    )
    wf.add_node(
        "SaveImage", {"images": decode.get_output_ref(0), "filename_prefix": "out"}, node_id="6"
    )
    return wf


class TestTypeSanity:
    def test_link_where_scalar_expected_warns(self):
        # KSampler.seed is declared ["INT", {default:0}] -> scalar; pass a link.
        wf = ComfyWorkflow()
        ckpt = wf.add_node("CheckpointLoaderSimple", {"ckpt_name": "m.safetensors"}, node_id="1")
        wf.add_node(
            "KSampler",
            {
                "model": ckpt.get_output_ref(0),
                "seed": ckpt.get_output_ref(0),  # link where scalar expected
                "steps": 20,
                "cfg": 7.0,
                "sampler_name": "euler",
                "scheduler": "normal",
                "positive": ["0", 0],
                "negative": ["0", 0],
                "latent_image": ["0", 0],
                "denoise": 1.0,
            },
            node_id="2",
        )
        report = validate_against_object_info(wf, MINIMAL_OBJECT_INFO)
        assert any("seed" in w and "scalar" in w for w in report["warnings"])

    def test_scalar_where_link_expected_warns(self):
        # CLIPTextEncode.clip is declared ["CLIP"] -> link; pass a scalar.
        wf = ComfyWorkflow()
        wf.add_node("CLIPTextEncode", {"text": "hi", "clip": "not-a-link"}, node_id="1")
        report = validate_against_object_info(wf, MINIMAL_OBJECT_INFO)
        assert any("clip" in w and "link type" in w for w in report["warnings"])

    def test_correct_types_do_not_warn(self):
        wf = _sampler_graph()
        report = validate_against_object_info(wf, MINIMAL_OBJECT_INFO)
        # No type-sanity warnings for a correctly-wired graph.
        sanity = [w for w in report["warnings"] if "scalar" in w or "link type" in w]
        assert sanity == []


class TestOrphanDetection:
    def test_orphan_node_detected(self):
        # Add a VAEDecode whose output nothing consumes and which is not an output node.
        wf = _sampler_graph()
        wf.add_node("VAEDecode", {"samples": ["0", 0], "vae": ["1", 2]}, node_id="99")
        orphans = find_orphan_nodes(wf)
        assert "99" in orphans
        # The main pipeline nodes are NOT orphans (consumed or output).
        assert "4" not in orphans
        assert "6" not in orphans  # SaveImage is an output node

    def test_no_orphans_in_clean_pipeline(self):
        assert find_orphan_nodes(_sampler_graph()) == []

    def test_saveimage_not_orphan(self):
        wf = ComfyWorkflow()
        wf.add_node("SaveImage", {"images": ["0", 0], "filename_prefix": "x"}, node_id="1")
        assert find_orphan_nodes(wf) == []

    def test_custom_output_set(self):
        wf = ComfyWorkflow()
        wf.add_node("MyCustomSaver", {"images": ["0", 0]}, node_id="1")
        # Default set does not know MyCustomSaver -> orphan.
        assert find_orphan_nodes(wf) == ["1"]
        # Override -> not orphan.
        assert find_orphan_nodes(wf, output_class_types=["MyCustomSaver"]) == []

    def test_validate_against_object_info_warns_about_orphans(self):
        wf = _sampler_graph()
        wf.add_node("VAEDecode", {"samples": ["0", 0], "vae": ["1", 2]}, node_id="99")
        report = validate_against_object_info(wf, MINIMAL_OBJECT_INFO)
        assert any("99" in w and "discarded" in w for w in report["warnings"])

    def test_warn_orphans_can_be_disabled(self):
        wf = _sampler_graph()
        wf.add_node("VAEDecode", {"samples": ["0", 0], "vae": ["1", 2]}, node_id="99")
        report = validate_against_object_info(wf, MINIMAL_OBJECT_INFO, warn_orphans=False)
        assert not any("99" in w and "discarded" in w for w in report["warnings"])


class TestDuplicateSeedsAcrossBatch:
    def test_no_collisions(self):
        wf = _sampler_graph()
        copies = list(wf.batch_seeds(3, mode="bump"))
        assert find_duplicate_seeds_across_batch(copies) == {}

    def test_detects_collision(self):
        # Two copies with the same seed value -> collision.
        wf = _sampler_graph()
        a = ComfyWorkflow.from_api_dict(wf.to_api_format())
        b = ComfyWorkflow.from_api_dict(wf.to_api_format())
        # Force both to seed 42.
        a.nodes["4"].inputs["seed"] = 42
        b.nodes["4"].inputs["seed"] = 42
        collisions = find_duplicate_seeds_across_batch([a, b])
        assert 42 in collisions
        assert (0, "4") in collisions[42]
        assert (1, "4") in collisions[42]

    def test_ignores_non_int_seeds(self):
        wf = ComfyWorkflow()
        wf.add_node("KSampler", {"seed": "not-a-number"}, node_id="1")
        wf.add_node("KSampler", {"seed": "not-a-number"}, node_id="2")
        # Two workflows, both with a non-int seed -> no collision reported.
        a = ComfyWorkflow.from_api_dict(wf.to_api_format())
        b = ComfyWorkflow.from_api_dict(wf.to_api_format())
        assert find_duplicate_seeds_across_batch([a, b]) == {}

    def test_custom_input_keys(self):
        wf = ComfyWorkflow()
        wf.add_node("KSampler", {"noise_seed": 7, "seed": 99}, node_id="1")
        a = ComfyWorkflow.from_api_dict(wf.to_api_format())
        b = ComfyWorkflow.from_api_dict(wf.to_api_format())
        # Collide on noise_seed only.
        collisions = find_duplicate_seeds_across_batch([a, b], input_keys=["noise_seed"])
        assert 7 in collisions
        assert 99 not in collisions

    def test_same_seed_in_same_workflow_does_not_count(self):
        # A single workflow with two identical seeds is NOT a cross-batch collision.
        wf = ComfyWorkflow()
        wf.add_node("KSampler", {"seed": 5}, node_id="1")
        wf.add_node("KSampler", {"seed": 5}, node_id="2")
        # Only one workflow in the "batch" -> no cross-batch collision.
        assert find_duplicate_seeds_across_batch([wf]) == {}
