"""
Tests for v0.4 graph-introspection / mutation helpers on ``ComfyWorkflow``.

Covers: iter_links, remove_node, topological_order, to_mermaid, diff,
batch_seeds. These are pure-Python, offline, no ComfyUI required.
"""

import pytest

from comfy_api_graphs import ComfyWorkflow, validate_node_references


def _linear_workflow() -> ComfyWorkflow:
    """Loader -> Encoder -> Sampler -> Decoder -> Save (ids 1..5)."""
    wf = ComfyWorkflow("linear")
    loader = wf.add_node("CheckpointLoaderSimple", {}, node_id="1")
    enc = wf.add_node(
        "CLIPTextEncode",
        {"text": "a cat", "clip": loader.get_output_ref(1)},
        node_id="2",
    )
    latent = wf.add_node("EmptyLatentImage", {}, node_id="3")
    sampler = wf.add_node(
        "KSampler",
        {
            "model": loader.get_output_ref(0),
            "positive": enc.get_output_ref(0),
            "negative": enc.get_output_ref(0),
            "latent_image": latent.get_output_ref(0),
            "seed": 42,
        },
        node_id="4",
    )
    decode = wf.add_node(
        "VAEDecode",
        {"samples": sampler.get_output_ref(0), "vae": loader.get_output_ref(2)},
        node_id="5",
    )
    wf.add_node("SaveImage", {"images": decode.get_output_ref(0)}, node_id="6")
    return wf


@pytest.fixture
def linear() -> ComfyWorkflow:
    return _linear_workflow()


class TestIterLinks:
    def test_yields_all_links(self, linear: ComfyWorkflow):
        links = list(linear.iter_links())
        # 1->2(clip), 1->4(model), 2->4(pos), 2->4(neg), 3->4(latent),
        # 4->5(samples), 1->5(vae), 5->6(images) = 8 links
        assert len(links) == 8
        for src_id, _slot, dst_id, _inp in links:
            assert src_id in linear.nodes
            assert dst_id in linear.nodes

    def test_link_shape(self, linear: ComfyWorkflow):
        links = list(linear.iter_links())
        # Sampler consumes latent from node 3 slot 0
        assert ("3", 0, "4", "latent_image") in links
        # SaveImage consumes images from decoder slot 0
        assert ("5", 0, "6", "images") in links

    def test_skips_placeholder_conditioning(self):
        wf = ComfyWorkflow()
        wf.add_node("KSampler", {"positive": ["0", 0], "negative": ["0", 0]})
        assert list(wf.iter_links()) == []

    def test_walks_nested_list_refs(self):
        wf = ComfyWorkflow()
        a = wf.add_node("A", {}, node_id="1")
        wf.add_node("B", {"x": [a.get_output_ref(0), a.get_output_ref(1)]}, node_id="2")
        links = list(wf.iter_links())
        assert ("1", 0, "2", "x") in links
        assert ("1", 1, "2", "x") in links


class TestRemoveNode:
    def test_removes_node_and_drops_dangling_links(self, linear: ComfyWorkflow):
        linear.remove_node("5")  # VAEDecode consumed by SaveImage
        assert "5" not in linear.nodes
        # SaveImage's `images` input referenced node 5 -> dropped
        assert "images" not in linear.nodes["6"].inputs

    def test_missing_raises(self, linear: ComfyWorkflow):
        with pytest.raises(KeyError):
            linear.remove_node("999")

    def test_cleans_nested_list_refs(self):
        wf = ComfyWorkflow()
        a = wf.add_node("A", {}, node_id="1")
        wf.add_node("B", {"x": [a.get_output_ref(0), a.get_output_ref(1)]}, node_id="2")
        wf.remove_node("1")
        assert wf.nodes["2"].inputs["x"] == []

    def test_chains(self, linear: ComfyWorkflow):
        # Returns self for chaining
        result = linear.remove_node("6")
        assert result is linear


class TestTopologicalOrder:
    def test_linear_order(self, linear: ComfyWorkflow):
        order = linear.topological_order()
        assert set(order) == set(linear.nodes)
        pos = {nid: i for i, nid in enumerate(order)}
        # Loader before everything that consumes it
        assert pos["1"] < pos["2"]
        assert pos["1"] < pos["4"]
        assert pos["1"] < pos["5"]
        assert pos["2"] < pos["4"]
        assert pos["3"] < pos["4"]
        assert pos["4"] < pos["5"]
        assert pos["5"] < pos["6"]

    def test_cycle_raises(self):
        wf = ComfyWorkflow()
        wf.add_node("A", {"x": ["2", 0]}, node_id="1")
        wf.add_node("B", {"x": ["1", 0]}, node_id="2")
        with pytest.raises(ValueError, match="cycle"):
            wf.topological_order()

    def test_empty_graph(self):
        assert ComfyWorkflow().topological_order() == []

    def test_isolated_nodes_preserved(self):
        wf = ComfyWorkflow()
        wf.add_node("Solo", {}, node_id="1")
        wf.add_node("Alone", {}, node_id="2")
        order = wf.topological_order()
        assert set(order) == {"1", "2"}


class TestToMermaid:
    def test_renders_flowchart_header(self, linear: ComfyWorkflow):
        md = linear.to_mermaid()
        assert md.startswith("flowchart TD")
        assert '"CheckpointLoaderSimple"' in md
        # An edge from loader to encoder exists
        assert "1 --> 2" in md

    def test_compact_mode_shows_ids(self, linear: ComfyWorkflow):
        md = linear.to_mermaid(show_class=False)
        assert "CheckpointLoaderSimple" not in md
        assert '1["1"]' in md

    def test_renders_empty_graph(self):
        md = ComfyWorkflow().to_mermaid()
        assert md == "flowchart TD"

    def test_renders_isolated_node(self):
        wf = ComfyWorkflow()
        wf.add_node("Solo", {}, node_id="9")
        md = wf.to_mermaid()
        assert '9["Solo"]' in md


class TestDiff:
    def test_identical_workflows_no_diff(self, linear: ComfyWorkflow):
        other = ComfyWorkflow.from_api_dict(linear.to_api_format())
        d = linear.diff(other)
        assert d == {
            "added": [],
            "removed": [],
            "changed_class": [],
            "added_inputs": {},
            "removed_inputs": {},
            "changed_inputs": {},
        }

    def test_added_removed_nodes(self, linear: ComfyWorkflow):
        other = ComfyWorkflow.from_api_dict(linear.to_api_format())
        other.add_node("ExtraNode", {}, node_id="99")
        other.remove_node("6")
        d = linear.diff(other)
        assert d["added"] == ["99"]
        assert d["removed"] == ["6"]

    def test_changed_input_value(self, linear: ComfyWorkflow):
        other = ComfyWorkflow.from_api_dict(linear.to_api_format())
        other.set_input("4", "seed", 999)
        other.set_input("2", "text", "a dog")
        d = linear.diff(other)
        assert d["changed_inputs"] == {"2": ["text"], "4": ["seed"]}

    def test_added_removed_inputs(self, linear: ComfyWorkflow):
        other = ComfyWorkflow.from_api_dict(linear.to_api_format())
        other.set_input("4", "new_widget", 1)
        other.nodes["4"].inputs.pop("seed", None)
        d = linear.diff(other)
        assert d["added_inputs"] == {"4": ["new_widget"]}
        assert d["removed_inputs"] == {"4": ["seed"]}

    def test_changed_class_type(self, linear: ComfyWorkflow):
        other = ComfyWorkflow.from_api_dict(linear.to_api_format())
        other.nodes["4"].class_type = "KSamplerAdvanced"
        d = linear.diff(other)
        assert d["changed_class"] == ["4"]


class TestBatchSeeds:
    def test_bump_mode_distinct_seeds(self, linear: ComfyWorkflow):
        copies = list(linear.batch_seeds(3, mode="bump"))
        assert len(copies) == 3
        seeds = [c.nodes["4"].inputs["seed"] for c in copies]
        assert seeds == [42, 43, 44]  # loaded 42 + index
        # Original unchanged
        assert linear.nodes["4"].inputs["seed"] == 42

    def test_bump_mode_with_start(self, linear: ComfyWorkflow):
        copies = list(linear.batch_seeds(3, mode="bump", start=1000))
        seeds = [c.nodes["4"].inputs["seed"] for c in copies]
        assert seeds == [1000, 1001, 1002]

    def test_randomize_mode_distinct_seeds(self, linear: ComfyWorkflow):
        # Deterministic via seeding each copy's RNG indirectly: randomize uses
        # the global rng; just assert distinctness and int type.
        copies = list(linear.batch_seeds(5, mode="randomize"))
        seeds = [c.nodes["4"].inputs["seed"] for c in copies]
        assert len(seeds) == 5
        assert all(isinstance(s, int) for s in seeds)
        # Extremely unlikely to collide for 5 draws in 2^32 space.
        assert len(set(seeds)) == 5

    def test_zero_yields_nothing(self, linear: ComfyWorkflow):
        assert list(linear.batch_seeds(0)) == []

    def test_negative_raises(self, linear: ComfyWorkflow):
        with pytest.raises(ValueError, match=">= 0"):
            list(linear.batch_seeds(-1))

    def test_unknown_mode_raises(self, linear: ComfyWorkflow):
        with pytest.raises(ValueError, match="mode"):
            list(linear.batch_seeds(1, mode="nope"))

    def test_each_copy_validates_clean(self, linear: ComfyWorkflow):
        for c in linear.batch_seeds(4, mode="bump"):
            assert validate_node_references(c) == []

    def test_class_types_filter(self, linear: ComfyWorkflow):
        # Only KSampler has a seed; restricting to a non-matching class leaves
        # seeds untouched (still 42).
        copies = list(linear.batch_seeds(2, mode="bump", class_types=["NonExistent"]))
        seeds = [c.nodes["4"].inputs["seed"] for c in copies]
        assert seeds == [42, 42]
