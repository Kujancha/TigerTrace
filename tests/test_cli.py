"""The documented source-checkout entry point and automatic paths."""

import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from PIL import Image
from tiger_reid.cli import main


class CliTests(unittest.TestCase):
    def raw_fixture(self, root):
        for split in ["train", "test"]:
            folder = root / "data" / split
            folder.mkdir(parents=True)
            names = []
            for i in range(2):
                name = f"{split}_{i}.png"
                names.append(name)
                Image.new("RGB", (12, 12), (i * 50, 30, 90)).save(folder / name)
            (folder / f"reid_keypoints_{split}.json").write_text(
                json.dumps({name: [0] * 45 for name in names})
            )
            text = "".join(
                (f"1,{name}\n" if split == "train" else f"{name}\n") for name in names
            )
            (folder / f"reid_list_{split}.csv").write_text(text)

    def test_prepare_resolves_annotations_and_copies_both(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            self.raw_fixture(root)
            with contextlib.redirect_stdout(io.StringIO()):
                main(["prepare"], project_root=root)
            self.assertTrue(
                (
                    root / "artifacts/prepared/train/labeled/tiger_001/train_0.png"
                ).exists()
            )
            self.assertTrue(
                (root / "artifacts/prepared/test/images/test_0.png").exists()
            )
            self.assertTrue((root / "data/train/train_0.png").exists())
            with self.assertRaises(SystemExit):
                main(["prepare"], project_root=root)

    def test_audit_only_is_read_only_for_images(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            self.raw_fixture(root)
            with contextlib.redirect_stdout(io.StringIO()):
                main(["prepare", "--audit-only"], project_root=root)
            self.assertFalse((root / "artifacts/prepared/train/labeled").exists())
            self.assertEqual(
                json.loads((root / "artifacts/prepared/train/audit.json").read_text())[
                    "status"
                ],
                "audit_passed",
            )

    def test_missing_annotations_have_actionable_error(self):
        with tempfile.TemporaryDirectory() as t:
            with self.assertRaisesRegex(SystemExit, "Extract them beside"):
                main(["prepare"], project_root=Path(t))

    def test_pilot_defaults_and_work_directory(self):
        from tiger_reid import train

        with (
            tempfile.TemporaryDirectory() as t,
            patch.object(train, "train") as callback,
        ):
            root = Path(t)
            main(["--work-dir", "work", "pilot"], project_root=root)
            args = callback.call_args.args[0]
            self.assertEqual(args.epochs, 2)
            self.assertEqual(args.steps, 5)
            self.assertEqual(args.seeds, [42])
            self.assertEqual(Path(args.output), root / "work/pilot")
            self.assertEqual(
                Path(args.manifest), root / "work/experiment/manifest.json"
            )

    def test_train_overrides_still_work(self):
        from tiger_reid import train

        with (
            tempfile.TemporaryDirectory() as t,
            patch.object(train, "train") as callback,
        ):
            main(
                ["train", "--epochs", "10", "--seeds", "10", "11", "12"],
                project_root=Path(t),
            )
            self.assertEqual(callback.call_args.args[0].epochs, 10)
            self.assertEqual(callback.call_args.args[0].seeds, [10, 11, 12])

    def test_evaluation_and_compare_defaults(self):
        from tiger_reid import evaluate, compare

        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            with patch.object(evaluate, "evaluate") as callback:
                main(["evaluate", "--include-pretrained"], project_root=root)
                self.assertTrue(callback.call_args.args[0].include_pretrained)
                self.assertEqual(
                    Path(callback.call_args.args[0].runs), root / "artifacts/runs"
                )
            with patch.object(compare, "compare") as callback:
                main(["compare", "--examples"], project_root=root)
                self.assertEqual(
                    Path(callback.call_args.args[0].manifest),
                    root / "artifacts/experiment/manifest.json",
                )

    def test_root_help_runs_from_another_working_directory(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as t:
            p = subprocess.run(
                [sys.executable, str(root / "run.py"), "--help"],
                cwd=t,
                text=True,
                capture_output=True,
            )
            self.assertEqual(p.returncode, 0, p.stderr)
            self.assertIn("prepare", p.stdout)


class EndToEndCliTests(unittest.TestCase):
    def test_documented_workflow_with_synthetic_images(self):
        import torch
        from tiger_reid import train, evaluate
        from tests.test_evaluate import TinyViT

        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            folder = root / "data/train"
            folder.mkdir(parents=True)
            labels = []
            points = {}
            for identity in range(8):
                for j in range(4):
                    name = f"{identity}_{j}.png"
                    Image.new("RGB", (12, 12), (identity * 25, j * 35, 70)).save(
                        folder / name
                    )
                    labels.append(f"{identity},{name}\n")
                    points[name] = [0] * 45
            (folder / "reid_list_train.csv").write_text("".join(labels))
            (folder / "reid_keypoints_train.json").write_text(json.dumps(points))
            public = root / "data/test"
            public.mkdir()
            Image.new("RGB", (12, 12), "white").save(public / "public.png")
            (public / "reid_list_test.csv").write_text("public.png\n")
            (public / "reid_keypoints_test.json").write_text(
                '{"public.png":' + json.dumps([0] * 45) + "}"
            )
            with (
                contextlib.redirect_stdout(io.StringIO()),
                patch.object(train, "TigerViT", TinyViT),
                patch.object(evaluate, "TigerViT", TinyViT),
            ):
                main(["prepare"], project_root=root)
                main(
                    ["split", "--train-ids", "4", "--val-ids", "2", "--test-ids", "2"],
                    project_root=root,
                )
                main(
                    ["baseline", "--device", "cpu", "--no-progress"], project_root=root
                )
                main(
                    [
                        "train",
                        "--device",
                        "cpu",
                        "--seeds",
                        "42",
                        "--epochs",
                        "1",
                        "--steps",
                        "1",
                        "--no-progress",
                    ],
                    project_root=root,
                )
                main(
                    [
                        "evaluate",
                        "--device",
                        "cpu",
                        "--seeds",
                        "42",
                        "--include-pretrained",
                        "--no-progress",
                    ],
                    project_root=root,
                )
                main(["compare", "--examples"], project_root=root)
            self.assertTrue((root / "artifacts/comparison/summary.csv").exists())
            self.assertTrue((root / "artifacts/comparison/retrieval.png").exists())
            self.assertEqual(
                json.loads((root / "artifacts/experiment/summary.json").read_text())[
                    "identity_counts"
                ],
                {"train": 4, "val": 2, "test": 2},
            )


if __name__ == "__main__":
    unittest.main()
