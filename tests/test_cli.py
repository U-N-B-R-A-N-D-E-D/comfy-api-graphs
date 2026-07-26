"""
Tests for ``python -m comfy_api_graphs`` CLI (stdlib argparse, no GPU).

Covers the three subcommands (validate, seeds, show) end-to-end via the
in-process ``main()`` entry point, plus a smoke test of the module run.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from comfy_api_graphs.__main__ import build_parser, main

FIXTURES = Path(__file__).parent / "fixtures"
MINIMAL_API = FIXTURES / "minimal_api_graph.json"

# Reuse the minimal object_info slice from the core test suite.
from tests.test_core import MINIMAL_OBJECT_INFO  # noqa: E402


@pytest.fixture
def object_info_file(tmp_path: Path) -> Path:
    p = tmp_path / "object_info.json"
    p.write_text(json.dumps(MINIMAL_OBJECT_INFO), encoding="utf-8")
    return p


class TestValidate:
    def test_validate_clean_graph_no_object_info(self, capsys):
        rc = main(["validate", str(MINIMAL_API)])
        out = capsys.readouterr().out
        assert rc == 0
        assert "Link check: OK" in out

    def test_validate_with_object_info_passes(self, capsys, object_info_file):
        rc = main(["validate", str(MINIMAL_API), "--object-info", str(object_info_file)])
        out = capsys.readouterr().out
        assert rc == 0
        assert "valid: True" in out

    def test_validate_missing_file(self, capsys, tmp_path):
        rc = main(["validate", str(tmp_path / "nope.json")])
        err = capsys.readouterr().err
        assert rc == 2
        assert "not found" in err.lower()

    def test_validate_broken_links_returns_1(self, capsys, tmp_path):
        bad = {"1": {"class_type": "KSampler", "inputs": {"model": ["missing", 0]}}}
        p = tmp_path / "bad.json"
        p.write_text(json.dumps(bad), encoding="utf-8")
        rc = main(["validate", str(p)])
        out = capsys.readouterr().out
        assert rc == 1
        assert "missing" in out

    def test_validate_unknown_class_type_fails_with_object_info(
        self, capsys, tmp_path, object_info_file
    ):
        bad = {"1": {"class_type": "TotallyFakeNode", "inputs": {"x": 1}}}
        p = tmp_path / "bad.json"
        p.write_text(json.dumps(bad), encoding="utf-8")
        rc = main(["validate", str(p), "--object-info", str(object_info_file)])
        out = capsys.readouterr().out
        assert rc == 1
        assert "TotallyFakeNode" in out


class TestSeeds:
    def test_bump_mode_writes_files(self, capsys, tmp_path):
        out_dir = tmp_path / "out"
        rc = main(
            [
                "seeds",
                str(MINIMAL_API),
                "--bump",
                "100",
                "--count",
                "3",
                "--out-dir",
                str(out_dir),
            ]
        )
        out = capsys.readouterr().out
        assert rc == 0
        files = sorted(out_dir.glob("*.json"))
        assert len(files) == 3
        seeds = [json.loads(f.read_text(encoding="utf-8"))["4"]["inputs"]["seed"] for f in files]
        assert seeds == [100, 101, 102]
        assert "seeds=" in out

    def test_randomize_mode(self, capsys, tmp_path):
        out_dir = tmp_path / "out"
        rc = main(
            [
                "seeds",
                str(MINIMAL_API),
                "--randomize",
                "--count",
                "2",
                "--out-dir",
                str(out_dir),
            ]
        )
        assert rc == 0
        seeds = [
            json.loads(f.read_text(encoding="utf-8"))["4"]["inputs"]["seed"]
            for f in sorted(out_dir.glob("*.json"))
        ]
        assert len(seeds) == 2
        assert seeds[0] != seeds[1]
        assert all(isinstance(s, int) for s in seeds)

    def test_dry_run_no_files(self, capsys, tmp_path):
        rc = main(["seeds", str(MINIMAL_API), "--bump", "5", "--count", "2"])
        out = capsys.readouterr().out
        assert rc == 0
        assert "seeds=" in out
        # No out-dir -> nothing written anywhere we can see.

    def test_randomize_and_bump_mutual_exclusion(self, capsys, tmp_path):
        rc = main(["seeds", str(MINIMAL_API), "--randomize", "--bump", "5", "--count", "1"])
        err = capsys.readouterr().err
        assert rc == 2
        assert "randomize" in err.lower() or "bump" in err.lower()

    def test_no_seed_inputs(self, capsys, tmp_path):
        # A graph with no seed widgets.
        wf = {
            "1": {"class_type": "SaveImage", "inputs": {"images": ["0", 0], "filename_prefix": "x"}}
        }
        p = tmp_path / "noseed.json"
        p.write_text(json.dumps(wf), encoding="utf-8")
        rc = main(["seeds", str(p), "--bump", "1", "--count", "1"])
        err = capsys.readouterr().err
        assert rc == 1
        assert "nothing to mutate" in err.lower()

    def test_fail_on_collision_clean_batch_passes(self, capsys, tmp_path):
        # A clean bump batch has distinct seeds -> no collision -> rc 0 even with
        # --fail-on-collision. (The collision helper itself is tested in
        # test_validation_enhancements.py.)
        rc = main(
            ["seeds", str(MINIMAL_API), "--bump", "42", "--count", "3", "--fail-on-collision"]
        )
        out = capsys.readouterr().out
        assert rc == 0
        assert "seed collisions" not in out.lower()


class TestShow:
    def test_show_topological(self, capsys):
        rc = main(["show", str(MINIMAL_API)])
        out = capsys.readouterr().out
        assert rc == 0
        assert "CheckpointLoaderSimple" in out
        assert "KSampler" in out

    def test_show_mermaid(self, capsys):
        rc = main(["show", str(MINIMAL_API), "--mermaid"])
        out = capsys.readouterr().out
        assert rc == 0
        assert out.startswith("flowchart TD")
        assert "CheckpointLoaderSimple" in out
        assert " --> " in out

    def test_show_mermaid_ids_only(self, capsys):
        rc = main(["show", str(MINIMAL_API), "--mermaid", "--ids"])
        out = capsys.readouterr().out
        assert rc == 0
        assert "CheckpointLoaderSimple" not in out
        assert '1["1"]' in out


class TestParser:
    def test_no_subcommand_errors(self):
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_help_works(self, capsys):
        parser = build_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["--help"])
        assert exc_info.value.code == 0


class TestModuleRun:
    def test_python_m_invocation_help(self):
        # Smoke test: `python -m comfy_api_graphs --help` exits 0.
        result = subprocess.run(
            [sys.executable, "-m", "comfy_api_graphs", "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0
        assert "validate" in result.stdout
        assert "seeds" in result.stdout
        assert "show" in result.stdout
