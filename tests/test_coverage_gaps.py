"""
Tests closing coverage gaps on existing public APIs.

Covers branches that were undertested in v0.3.0:
- fetch_object_info (mocked urllib)
- load_object_info rejection paths (non-dict, empty, bad shape)
- validate_node_references nested-list refs
- bump_seeds non-int / bool handling
- deprecated save_graph_json (DeprecationWarning)
- from_api_json in-memory dict path + name defaulting
- estimate_vram_usage recommendation tiers
- validate_workflow_complete duplicate-id detection
"""

import json
from pathlib import Path
from unittest import mock

import pytest

from comfy_api_graphs import (
    ComfyWorkflow,
    estimate_vram_usage,
    fetch_object_info,
    load_object_info,
    validate_against_object_info,
    validate_node_references,
    validate_workflow_complete,
)
from tests.test_core import MINIMAL_OBJECT_INFO

FIXTURES = Path(__file__).parent / "fixtures"
MINIMAL_API = FIXTURES / "minimal_api_graph.json"


class TestFetchObjectInfo:
    def test_fetch_success_mocked(self):
        payload = json.dumps(MINIMAL_OBJECT_INFO).encode("utf-8")
        fake_resp = mock.MagicMock()
        fake_resp.read.return_value = payload
        fake_resp.__enter__.return_value = fake_resp
        fake_resp.__exit__.return_value = False
        with mock.patch("urllib.request.urlopen", return_value=fake_resp) as m:
            info = fetch_object_info("http://example.invalid/object_info", timeout=5)
        assert "KSampler" in info
        m.assert_called_once_with("http://example.invalid/object_info", timeout=5)

    def test_fetch_raises_on_bad_shape(self):
        # Response is valid JSON but not object_info shape.
        payload = json.dumps({"KSampler": {"not_input": True}}).encode("utf-8")
        fake_resp = mock.MagicMock()
        fake_resp.read.return_value = payload
        fake_resp.__enter__.return_value = fake_resp
        fake_resp.__exit__.return_value = False
        with (
            mock.patch("urllib.request.urlopen", return_value=fake_resp),
            pytest.raises(ValueError, match="object_info"),
        ):
            fetch_object_info("http://example.invalid/object_info")

    def test_fetch_propagates_urlerror(self):
        import urllib.error

        def boom(*a, **kw):
            raise urllib.error.URLError("no connection")

        with (
            mock.patch("urllib.request.urlopen", side_effect=boom),
            pytest.raises(urllib.error.URLError),
        ):
            fetch_object_info("http://example.invalid/object_info")


class TestLoadObjectInfoEdgeCases:
    def test_rejects_non_dict(self):
        with pytest.raises(TypeError):
            load_object_info(["not", "a", "dict"])  # type: ignore[arg-type]

    def test_empty_dict_returns_empty(self):
        assert load_object_info({}) == {}

    def test_rejects_bad_shape(self):
        with pytest.raises(ValueError, match="object_info"):
            load_object_info({"X": {"not_input": True}})

    def test_accepts_path(self, tmp_path):
        p = tmp_path / "oi.json"
        p.write_text(json.dumps(MINIMAL_OBJECT_INFO), encoding="utf-8")
        assert load_object_info(p) == MINIMAL_OBJECT_INFO


class TestValidateNodeReferencesNested:
    def test_nested_list_with_bad_ref(self):
        wf = ComfyWorkflow()
        wf.add_node("A", {}, node_id="1")
        wf.add_node("B", {"x": [["missing", 0], ["1", 0]]}, node_id="2")
        errors = validate_node_references(wf)
        assert any("missing" in e for e in errors)
        # The valid ref to "1" should not produce an error.
        assert all("non-existent node: 1" not in e for e in errors)

    def test_placeholder_skipped_in_nested(self):
        wf = ComfyWorkflow()
        wf.add_node("KSampler", {"positive": [["0", 0], ["0", 0]]}, node_id="1")
        assert validate_node_references(wf) == []


class TestBumpSeedsNonInt:
    def test_string_seed_replaced_with_n(self):
        wf = ComfyWorkflow()
        wf.add_node("KSampler", {"seed": "not-a-number"}, node_id="1")
        updated = wf.bump_seeds(7)
        assert updated == ["1"]
        assert wf.nodes["1"].inputs["seed"] == 7

    def test_bool_seed_replaced_with_n(self):
        # bool is a subclass of int; the helper must not treat True as 1+7.
        wf = ComfyWorkflow()
        wf.add_node("KSampler", {"seed": True}, node_id="1")
        wf.bump_seeds(3)
        assert wf.nodes["1"].inputs["seed"] == 3

    def test_float_seed_replaced_with_n(self):
        wf = ComfyWorkflow()
        wf.add_node("KSampler", {"seed": 1.5}, node_id="1")
        wf.bump_seeds(10)
        assert wf.nodes["1"].inputs["seed"] == 10


class TestDeprecatedSaveGraphJson:
    def test_save_graph_json_warns(self, tmp_path):
        wf = ComfyWorkflow()
        wf.add_node("CheckpointLoaderSimple", {}, node_id="1")
        out = tmp_path / "graph.json"
        with pytest.warns(DeprecationWarning, match="to_graph_format"):
            wf.save_graph_json(str(out))
        assert out.is_file()
        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["links"] == []
        assert len(data["nodes"]) == 1


class TestFromApiJsonDictPath:
    def test_from_api_json_with_dict_uses_default_name(self):
        raw = json.loads(MINIMAL_API.read_text(encoding="utf-8"))
        wf = ComfyWorkflow.from_api_json(raw)
        assert wf.name == "workflow"
        assert len(wf) == 6

    def test_from_api_json_with_dict_and_explicit_name(self):
        raw = json.loads(MINIMAL_API.read_text(encoding="utf-8"))
        wf = ComfyWorkflow.from_api_json(raw, name="custom")
        assert wf.name == "custom"

    def test_from_api_dict_rejects_non_dict_node(self):
        # First node valid so the payload-shape sample passes; second node is
        # not a dict -> the per-node check fires.
        raw = {
            "1": {"class_type": "X", "inputs": {}},
            "2": "not a dict",
        }
        with pytest.raises(ValueError, match="must be a dict"):
            ComfyWorkflow.from_api_dict(raw)  # type: ignore[arg-type]

    def test_from_api_dict_rejects_non_dict_payload(self):
        # Entire payload is not a dict -> TypeError from _unwrap_api_payload.
        with pytest.raises(TypeError, match="dict"):
            ComfyWorkflow.from_api_dict("not a dict")  # type: ignore[arg-type]

    def test_from_api_dict_rejects_missing_class_type(self):
        with pytest.raises(ValueError, match="class_type"):
            ComfyWorkflow.from_api_dict({"1": {"inputs": {}}})

    def test_from_api_dict_rejects_non_dict_inputs(self):
        with pytest.raises(ValueError, match="inputs must be a dict"):
            ComfyWorkflow.from_api_dict({"1": {"class_type": "X", "inputs": "bad"}})

    def test_from_api_dict_non_string_title_ignored(self):
        raw = {"1": {"class_type": "X", "inputs": {}, "_meta": {"title": 123}}}
        wf = ComfyWorkflow.from_api_dict(raw)
        assert wf.nodes["1"].title is None


class TestEstimateVramRecommendationTiers:
    def _wf_with_vram(self, total: float):
        """Build a workflow whose estimate_vram_usage total lands near ``total``."""
        wf = ComfyWorkflow()
        # Each VAELoader adds 0.5 GB + 2.0 overhead. To hit arbitrary totals we
        # use CheckpointLoaderSimple (12 GB default) + N VAELoaders.
        wf.add_node("CheckpointLoaderSimple", {"ckpt_name": "x.safetensors"}, node_id="1")
        n_vae = max(0, int((total - 2.0 - 12.0) / 0.5))
        for i in range(n_vae):
            wf.add_node("VAELoader", {"vae_name": "v.safetensors"}, node_id=str(i + 2))
        return wf

    def test_light_tier(self):
        # Total < 8 -> light recommendation.
        wf = ComfyWorkflow()
        wf.add_node("VAELoader", {"vae_name": "v.safetensors"})  # 0.5 + 2.0 = 2.5
        est = estimate_vram_usage(wf)
        assert est["estimated_vram_gb"] < 8
        assert "light" in est["recommendation"]

    def test_medium_tier(self):
        # 8 < total <= 12 -> 12GB-class hint.
        wf = ComfyWorkflow()
        for _ in range(13):
            wf.add_node("VAELoader", {"vae_name": "v.safetensors"})  # 13*0.5+2 = 8.5
        est = estimate_vram_usage(wf)
        assert 8 < est["estimated_vram_gb"] <= 12
        assert "12GB" in est["recommendation"]

    def test_heavy_tier(self):
        # 12 < total <= 20 -> 16GB-class hint.
        wf = self._wf_with_vram(15.0)
        est = estimate_vram_usage(wf)
        assert 12 < est["estimated_vram_gb"] <= 20
        assert "16GB" in est["recommendation"]

    def test_very_heavy_tier(self):
        # total > 20 -> 24GB-class hint.
        wf = ComfyWorkflow()
        wf.add_node("CheckpointLoaderSimple", {"ckpt_name": "flux1-dev.safetensors"})  # 22 + 2 = 24
        est = estimate_vram_usage(wf)
        assert est["estimated_vram_gb"] > 20
        assert "24GB" in est["recommendation"]


class TestValidateWorkflowCompleteDuplicateId:
    def test_duplicate_id_detected(self):
        # ComfyWorkflow.nodes is a dict keyed by id, so true duplicates cannot
        # coexist; this test documents that the duplicate-id check is a
        # defensive no-op for the current in-memory model (it cannot fire
        # because dict keys are unique). We verify the check does not error
        # on a normal graph.
        wf = ComfyWorkflow()
        ckpt = wf.add_node("CheckpointLoaderSimple", {})
        decode = wf.add_node("VAEDecode", {"vae": ckpt.get_output_ref(2), "samples": ["0", 0]})
        wf.add_node("SaveImage", {"images": decode.get_output_ref()})
        result = validate_workflow_complete(wf)
        assert result["valid"] is True
        assert all("Duplicate" not in e for e in result["errors"])


class TestValidateAgainstObjectInfoEmpty:
    def test_empty_object_info_fails(self):
        wf = ComfyWorkflow()
        wf.add_node("KSampler", {"model": ["0", 0]}, node_id="1")
        report = validate_against_object_info(wf, {})
        assert report["valid"] is False
        assert any("empty" in e for e in report["errors"])

    def test_input_schema_names_malformed(self):
        # A schema whose "input" is not a dict -> names collapse to empty sets.
        bad_info = {"X": {"input": "not-a-dict"}}
        wf = ComfyWorkflow()
        wf.add_node("X", {"a": 1}, node_id="1")
        report = validate_against_object_info(wf, bad_info)
        # No required inputs (empty set) -> no missing-required errors; unknown
        # input "a" warns.
        assert "a" in str(report["warnings"])
