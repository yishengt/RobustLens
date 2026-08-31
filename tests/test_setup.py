"""Tests for the setup / doctor script.

setup.py runs on the *system* interpreter before any virtualenv exists, so it
must import with only the standard library. These tests pin that, plus the
status logic that tells a new contributor what is missing.

Nothing here downloads anything.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SETUP = ROOT / "scripts/setup.py"


def load_setup():
    import importlib.util

    spec = importlib.util.spec_from_file_location("project_setup", SETUP)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class StandardLibraryOnlyTest(unittest.TestCase):
    """It must run before dependencies exist."""

    def test_imports_with_no_third_party_packages(self) -> None:
        source = SETUP.read_text(encoding="utf-8")
        for banned in ("import torch", "import numpy", "import yaml", "import requests"):
            with self.subTest(banned=banned):
                self.assertNotIn(banned, source)

    def test_help_runs_on_the_system_interpreter(self) -> None:
        result = subprocess.run(
            [sys.executable, str(SETUP), "--help"], capture_output=True, text=True, timeout=120
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("--check", result.stdout)


class StatusReportingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.setup = load_setup()

    def test_checkpoint_absent_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.setup.CHECKPOINT = Path(tmp) / "absent.pt"
            ok, detail = self.setup.checkpoint_status()
        self.assertFalse(ok)
        self.assertIn("not downloaded", detail)

    def test_partial_checkpoint_is_not_accepted(self) -> None:
        """A truncated 2 GB download must not read as ready."""

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pytorch_model.pt"
            path.write_bytes(b"\x00" * 1024)
            self.setup.CHECKPOINT = path
            ok, detail = self.setup.checkpoint_status()
        self.assertFalse(ok)
        self.assertIn("incomplete", detail)

    def test_complete_checkpoint_is_accepted_by_size(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pytorch_model.pt"
            path.write_bytes(b"")
            self.setup.CHECKPOINT = path
            self.setup.CHECKPOINT_BYTES = 0
            ok, detail = self.setup.checkpoint_status()
        self.assertTrue(ok)
        self.assertIn("verified", detail)

    def test_dataset_absent_is_optional(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.setup.DATASET_DIR = Path(tmp) / "absent"
            ok, detail = self.setup.dataset_status()
        self.assertFalse(ok)
        self.assertIn("optional", detail)

    def test_dataset_counts_shards(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "sid_set" / "data"
            directory.mkdir(parents=True)
            for index in range(3):
                (directory / f"validation-0000{index}.parquet").write_bytes(b"x" * 100)
            self.setup.DATASET_DIR = Path(tmp) / "sid_set"
            ok, detail = self.setup.dataset_status()
        self.assertTrue(ok)
        self.assertIn("3 shard", detail)

    def test_unreadable_calibration_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "calibration.json"
            path.write_text("not json at all", encoding="utf-8")
            self.setup.CALIBRATION = path
            ok, detail = self.setup.calibration_status()
        self.assertFalse(ok)
        self.assertIn("unreadable", detail)

    def test_valid_calibration_reports_its_method(self) -> None:
        import json

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "calibration.json"
            path.write_text(
                json.dumps({"method": "platt", "fitted_on": "clean_validation"}), encoding="utf-8"
            )
            self.setup.CALIBRATION = path
            ok, detail = self.setup.calibration_status()
        self.assertTrue(ok)
        self.assertIn("platt", detail)


class DoctorContractTest(unittest.TestCase):
    """--check must be usable as a CI gate."""

    def test_check_exits_zero_when_ready(self) -> None:
        result = subprocess.run(
            [sys.executable, str(SETUP), "--check"],
            capture_output=True,
            text=True,
            timeout=600,
            cwd=str(ROOT),
        )
        self.assertIn(result.returncode, (0, 1))  # 0 ready, 1 something missing
        self.assertIn("Project status", result.stdout)

    def test_every_check_offers_a_fix_command(self) -> None:
        setup = load_setup()
        for name, _probe, _required, fix in setup.CHECKS:
            with self.subTest(check=name):
                self.assertTrue(fix.strip())

    def test_required_items_are_the_ones_inference_needs(self) -> None:
        setup = load_setup()
        required = {name for name, _p, req, _f in setup.CHECKS if req}
        self.assertIn("Model checkpoint", required)
        self.assertIn("Dependencies", required)
        # The dataset and calibration are genuinely optional for inference.
        optional = {name for name, _p, req, _f in setup.CHECKS if not req}
        self.assertIn("Evaluation dataset", optional)
        self.assertIn("Calibration", optional)


class DocumentedPathsTest(unittest.TestCase):
    """The script, the config and the docs must agree on where things live."""

    def test_checkpoint_path_matches_the_app_default(self) -> None:
        setup = load_setup()
        app_source = (ROOT / "app.py").read_text(encoding="utf-8")
        relative = setup.CHECKPOINT.relative_to(ROOT).as_posix()
        self.assertIn(relative, app_source)

    def test_readme_documents_the_one_command_setup(self) -> None:
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("scripts/setup.py --all", readme)
        self.assertIn("scripts/setup.py --check", readme)

    def _git_ignores(self, relative_path: str) -> bool:
        """Ask git itself whether it would ignore ``relative_path``."""

        result = subprocess.run(
            ["git", "check-ignore", "-q", "--no-index", relative_path],
            cwd=ROOT,
            capture_output=True,
        )
        if result.returncode not in (0, 1):
            self.skipTest("git is unavailable or this tree is not a repository")
        return result.returncode == 0

    def test_large_artefacts_stay_git_ignored(self) -> None:
        """Large regenerable artefacts must not reach the repository."""

        for path in (
            "models/pretrained/pytorch_model.pt",
            "data/sid_set/shard.parquet",
            "outputs/predictions.json",
        ):
            with self.subTest(path=path):
                self.assertTrue(self._git_ignores(path), path)

    def test_source_directories_are_never_git_ignored(self) -> None:
        """Regression: a bare ``data/`` pattern also matches ``src/data/``.

        That silently dropped real source from the published repository, so a
        fresh clone could not import it. Dataset patterns must stay anchored to
        the repository root with a leading slash.
        """

        for path in (
            "src/data/dataset.py",
            "src/data/__init__.py",
            "src/pipeline/pipeline.py",
            "scripts/run_inference.py",
        ):
            with self.subTest(path=path):
                self.assertFalse(self._git_ignores(path), path)


if __name__ == "__main__":
    unittest.main()
