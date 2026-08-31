"""Focused tests for leakage-safe local-edit fine-tuning and adapter exports."""

from __future__ import annotations

import copy
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from src.finetune.dataset import SUBGROUP_LABELS, discover_split, verify_split_groups
from src.finetune.losses import binary_metrics, metrics_by_subgroup
from src.finetune.model import FineTuneModel, load_saved_adapter_into_model
from src.finetune.train_lora import _make_datasets, _training_transform
from tests.helpers import make_image, requires


def _write(path: Path, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    make_image(width=32, height=24, seed=seed).save(path)


class SubgroupDatasetTest(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        for split_offset, split in enumerate(("train", "validation", "test")):
            for group in range(2):
                _write(
                    self.root / "local" / split / "authentic" / f"g{split_offset}{group}" / "source.jpg",
                    100 * split_offset + group,
                )
                _write(
                    self.root / "local" / split / "ai_edited" / f"g{split_offset}{group}" / "edit.jpg",
                    100 * split_offset + group + 10,
                )
        _write(self.root / "local" / "train" / "moderate" / "m0" / "edit.jpg", 501)
        _write(self.root / "local" / "train" / "shifted" / "t0" / "edit.jpg", 502)
        for index in range(12):
            _write(self.root / "synthetic" / f"generator_{index % 2}" / f"s{index}.png", 700 + index)

    def _config(self):
        return {
            "_project_root": str(self.root),
            "training": {
                "seed": 17,
                "official_transformations": [],
                "consistency": {"enabled": False, "weight": 0.0},
            },
            "data": {
                "train_dir": "local/train",
                "validation_dir": "local/validation",
                "test_dir": "local/test",
                "masks_dir": None,
                "extensions": [".jpg", ".png"],
                "include_conflicting_labels": False,
                "quarantine_path": None,
                "synthetic_mixture_dir": "synthetic",
                "synthetic_mixture_fraction": 0.25,
                "subgroups": {
                    "authentic": {"directory": "authentic", "label": 0},
                    "minor_edit": {"directory": "ai_edited", "label": 1},
                    "moderate_edit": {"directory": "moderate", "label": 1},
                    "transformed": {"directory": "shifted", "label": 1},
                },
            },
        }

    def test_legacy_ai_edited_directory_is_minor_edit(self) -> None:
        summary = discover_split(self.root / "local" / "validation", "validation")
        edited = [record for record in summary.records if record.label == 1]
        self.assertTrue(edited)
        self.assertEqual({record.subgroup for record in edited}, {"minor_edit"})

    def test_all_configured_subgroups_are_preserved(self) -> None:
        datasets, summary = _make_datasets(self._config())
        counts = summary["train"]["subgroup_counts"]
        for subgroup in SUBGROUP_LABELS:
            with self.subTest(subgroup=subgroup):
                self.assertGreater(counts[subgroup], 0)
        self.assertEqual(
            {record.subgroup for record in datasets["train"].records},
            set(SUBGROUP_LABELS),
        )

    def test_synthetic_mixture_is_deterministic_and_train_only(self) -> None:
        first, first_summary = _make_datasets(self._config())
        second, second_summary = _make_datasets(self._config())
        first_paths = [
            str(record.image_path)
            for record in first["train"].records
            if record.subgroup == "synthetic"
        ]
        second_paths = [
            str(record.image_path)
            for record in second["train"].records
            if record.subgroup == "synthetic"
        ]
        self.assertEqual(first_paths, second_paths)
        self.assertEqual(first_summary["synthetic_mixture"], second_summary["synthetic_mixture"])
        self.assertTrue(first_paths)
        for split in ("validation", "test"):
            self.assertNotIn("synthetic", {record.subgroup for record in first[split].records})

    def test_group_limit_keeps_whole_source_groups(self) -> None:
        config = self._config()
        config["data"]["max_groups_per_split"] = {
            "train": 2,
            "validation": 1,
            "test": 1,
        }
        datasets, summary = _make_datasets(config)
        # The limit caps LOCAL-EDIT source groups. Replay-mixture images are
        # governed by synthetic_mixture_fraction instead, and are added after
        # the limit has settled the local-edit set -- applying the limit to them
        # too would discard almost the whole mixture, because every replay image
        # is its own group.
        local_groups = {
            r.group_id for r in datasets["train"].records if r.subgroup != "synthetic"
        }
        self.assertLessEqual(len(local_groups), 2)
        mixture = summary["synthetic_mixture"]
        if mixture["enabled"]:
            self.assertGreater(mixture["selected"], 0)
            self.assertLessEqual(mixture["realized_fraction"], mixture["requested_fraction"] + 1e-9)
        for split in ("validation", "test"):
            groups = {record.group_id for record in datasets[split].records}
            self.assertEqual(len(groups), 1)
            labels = {record.label for record in datasets[split].records}
            self.assertEqual(labels, {0, 1})

    def test_same_source_group_across_splits_is_rejected(self) -> None:
        _write(self.root / "local" / "validation" / "authentic" / "g00" / "copy.jpg", 99)
        summaries = [
            discover_split(self.root / "local" / split, split)
            for split in ("train", "validation", "test")
        ]
        with self.assertRaisesRegex(ValueError, "Dataset leakage"):
            verify_split_groups(summaries)


class FineTuneMetricTest(unittest.TestCase):
    def test_auc_ranks_higher_positive_scores_as_perfect(self) -> None:
        metrics = binary_metrics(
            np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9]), 0.5
        )
        self.assertEqual(metrics["auroc"], 1.0)

    def test_subgroup_metrics_report_absent_groups_honestly(self) -> None:
        result = metrics_by_subgroup(
            np.array([0, 1]),
            np.array([0.1, 0.9]),
            ["authentic", "minor_edit"],
            expected_subgroups=tuple(SUBGROUP_LABELS),
        )
        self.assertEqual(result["moderate_edit"], {"count": 0, "metrics": None})
        self.assertEqual(result["minor_edit"]["metrics"]["recall"], 1.0)

    def test_official_transformations_do_not_require_albumentations(self) -> None:
        transform = _training_transform(
            {"training": {"official_transformations": ["jpeg_90", "blur_0.5"]}}
        )
        output = transform(make_image())
        self.assertIsInstance(output, Image.Image)
        self.assertEqual(output.mode, "RGB")


class TinyAdapterBranch(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lora_A = nn.Parameter(torch.zeros(2, 3))
        self.lora_B = nn.Parameter(torch.zeros(3, 2))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + value @ self.lora_A.t() @ self.lora_B.t()


class TinyFineTuneDetector(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.siglip = TinyAdapterBranch()
        self.dinov2 = TinyAdapterBranch()
        self.classifier = nn.Linear(6, 1)

    def forward_with_features(self, left: torch.Tensor, right: torch.Tensor):
        left_features = self.siglip(left.mean(dim=(2, 3)))
        right_features = self.dinov2(right.mean(dim=(2, 3)))
        logits = self.classifier(torch.cat([left_features, right_features], dim=1)).reshape(-1)
        return logits, left_features, right_features

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return self.forward_with_features(left, right)[0]


@requires("safetensors")
class AdapterRoundTripTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        torch.manual_seed(4)
        self.template = TinyFineTuneDetector()
        self.initial = copy.deepcopy(self.template.state_dict())
        self.left = torch.rand(3, 3, 4, 4)
        self.right = torch.rand(3, 3, 5, 5)

    def _fresh(self) -> FineTuneModel:
        detector = TinyFineTuneDetector()
        detector.load_state_dict(self.initial)
        return FineTuneModel(detector, torch.device("cpu"), mode="head_only")

    def test_save_reload_preserves_predictions_exactly(self) -> None:
        trained = self._fresh()
        with torch.no_grad():
            trained.model.siglip.lora_A.add_(0.2)
            trained.model.siglip.lora_B.add_(0.1)
            trained.model.classifier.weight.add_(0.3)
            trained.model.classifier.bias.sub_(0.4)
        trained.eval()
        expected = trained(self.left, self.right).detach()
        trained.save_adapter(self.tmp / "adapter", metadata={"smoke": True})

        restored = self._fresh()
        restored.load_saved_adapter(self.tmp / "adapter")
        restored.eval()
        actual = restored(self.left, self.right).detach()
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    def test_standalone_loader_preserves_predictions(self) -> None:
        trained = self._fresh()
        with torch.no_grad():
            trained.model.dinov2.lora_A.add_(0.25)
            trained.model.dinov2.lora_B.sub_(0.15)
            trained.model.classifier.bias.add_(0.7)
        trained.eval()
        expected = trained(self.left, self.right).detach()
        trained.save_adapter(self.tmp / "standalone")

        restored = self._fresh()
        load_saved_adapter_into_model(restored.model, self.tmp / "standalone")
        restored.eval()
        actual = restored(self.left, self.right).detach()
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


if __name__ == "__main__":
    unittest.main()


class GroupIdClassDirectoryTest(unittest.TestCase):
    """Regression: a class directory must never become an image's group id.

    ``_group_id`` treats a nested parent as "one folder per original", so any
    directory naming a CLASS rather than an original has to be stripped first.
    Stripping only the two names in LABELS collapsed every image under the
    supported ``synthetic/`` subgroup into a single group, which then appeared
    in all three splits and tripped the leakage guard on a perfectly clean
    dataset.
    """

    def test_every_subgroup_directory_yields_per_image_groups(self) -> None:
        from src.finetune.dataset import DEFAULT_SUBGROUP_DIRECTORIES, _group_id

        root = Path("/dataset/train")
        for directory in DEFAULT_SUBGROUP_DIRECTORIES.values():
            with self.subTest(directory=directory):
                first = _group_id(root / directory / "image_one.jpg", root)
                second = _group_id(root / directory / "image_two.jpg", root)
                self.assertNotEqual(
                    first,
                    second,
                    f"{directory}/ collapsed distinct images into one group {first!r}",
                )
                self.assertNotEqual(first, directory)

    def test_a_per_original_folder_still_groups_its_versions(self) -> None:
        from src.finetune.dataset import _group_id

        root = Path("/dataset/train")
        a = _group_id(root / "authentic" / "scene_42" / "source.jpg", root)
        b = _group_id(root / "authentic" / "scene_42" / "edited.jpg", root)
        self.assertEqual(a, b)
