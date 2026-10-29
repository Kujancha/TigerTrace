import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F
from tiger_reid.data import BalancedBatches, load_manifest
from tiger_reid.metrics import retrieval
from tiger_reid.losses import MetricObjective, NAMES
from tiger_reid import train as app


class TinyViT(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.embedding_head = nn.Linear(3, 128)

    def forward(self, x):
        return F.normalize(self.embedding_head(x.mean((2, 3))), dim=1)

    def features(self, x):
        return F.normalize(x.mean((2, 3)), dim=1)


class TrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def fixture(self, b):
        root = b / "prepared"
        root.mkdir()
        records = []
        ids = {"train": ["0", "1", "2", "3"], "val": ["4", "5"], "test": ["6", "7"]}
        for split, identities in ids.items():
            for identity in identities:
                for j in range(4):
                    filename = f"{identity}_{j}.png"
                    p = root / filename
                    image = Image.new("RGB", (12, 12), (int(identity) * 25, j * 35, 70))
                    image.save(p)
                    records.append(
                        {
                            "filename": filename,
                            "identity": identity,
                            "path": filename,
                            "split": split,
                            "role": "train"
                            if split == "train"
                            else ("gallery" if j < 2 else "query"),
                            "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                            "decoded_sha256": hashlib.sha256(
                                str(image.size).encode() + image.tobytes()
                            ).hexdigest(),
                        }
                    )
        m = {
            "schema_version": 1,
            "prepared_root": str(root),
            "identity_splits": ids,
            "records": records,
            "capture_session_separation_verified": False,
            "review_status": "pending_manual_review",
        }
        path = b / "manifest.json"
        path.write_text(json.dumps(m))
        return path, root

    def args(self, path, out, *extra):
        return app.parser().parse_args(
            [
                "train",
                "--manifest",
                str(path),
                "--output",
                str(out),
                "--device",
                "cpu",
                "--seeds",
                "42",
                "--epochs",
                "2",
                "--steps",
                "2",
                "--no-progress",
                *extra,
            ]
        )

    def test_balanced_batches(self):
        labels = [i for i in range(4) for _ in range(4)]
        a = BalancedBatches(labels, 4, 2, 3, 42)
        b = BalancedBatches(labels, 4, 2, 3, 42)
        self.assertEqual(list(a), list(b))
        for batch in a:
            self.assertEqual(len(set(batch)), 8)
            values = [labels[i] for i in batch]
            self.assertTrue(all(values.count(i) == 2 for i in set(values)))
        with self.assertRaises(ValueError):
            BalancedBatches([0, 0, 1], 2, 2, 3, 42)

    def test_hand_calculated_metrics(self):
        rows = [
            {"filename": str(i), "identity": identity, "role": role}
            for i, (identity, role) in enumerate(
                [
                    ("a", "gallery"),
                    ("a", "gallery"),
                    ("b", "gallery"),
                    ("a", "query"),
                    ("b", "query"),
                ]
            )
        ]
        z = np.array([[1, 0], [0.8, 0.6], [0, 1], [1, 0], [1, 0]], dtype=float)
        r = retrieval(z, rows)["summary"]
        self.assertEqual(r["rank1"], 0.5)
        self.assertAlmostEqual(r["map"], 2 / 3)

    def test_loss_gradients_and_proxy_learning(self):
        for name in NAMES:
            torch.manual_seed(42)
            z = torch.randn(8, 128, requires_grad=True)
            y = torch.arange(4).repeat_interleave(2)
            objective = MetricObjective(name, 4)
            params = list(objective.parameters())
            before = [p.detach().clone() for p in params]
            opt = torch.optim.SGD([z] + params, lr=0.01)
            loss = objective(F.normalize(z, dim=1), y)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertTrue(torch.isfinite(z.grad).all())
            opt.step()
            if name == "proxy_anchor":
                self.assertTrue(
                    any(not torch.equal(a, b) for a, b in zip(before, params))
                )

    def test_manifest_rejects_overlap_and_old_schema(self):
        with tempfile.TemporaryDirectory() as t:
            b = Path(t)
            path, root = self.fixture(b)
            m = json.loads(path.read_text())
            m["identity_splits"]["val"].append("0")
            path.write_text(json.dumps(m))
            with self.assertRaises(ValueError):
                load_manifest(path)
            path.write_text("{}")
            with self.assertRaises(ValueError):
                load_manifest(path)

    def test_three_loss_pipeline_never_reads_test_images(self):
        with tempfile.TemporaryDirectory() as t:
            b = Path(t)
            path, root = self.fixture(b)
            for row in json.loads(path.read_text())["records"]:
                if row["split"] == "test":
                    (root / row["path"]).unlink()
            with patch.object(app, "TigerViT", TinyViT):
                app.train(self.args(path, b / "runs"))
            fingerprints = []
            for name in NAMES:
                run = b / "runs" / f"{name}_seed42"
                self.assertEqual(
                    json.loads((run / "status.json").read_text())["status"], "complete"
                )
                self.assertEqual(len(json.loads((run / "history.json").read_text())), 2)
                state = torch.load(run / "best.pt", weights_only=False)
                self.assertEqual(state["config"]["loss"], name)
                self.assertTrue((run / "last.pt").exists())
                self.assertTrue((run / "history.csv").exists())
                fingerprints.append(state["config"]["initial_head_sha256"])
            self.assertEqual(len(set(fingerprints)), 1)
            with patch.object(app, "TigerViT", TinyViT):
                app.train(self.args(path, b / "runs", "--resume"))

    def test_resume_matches_uninterrupted_training(self):
        with tempfile.TemporaryDirectory() as t:
            b = Path(t)
            path, root = self.fixture(b)
            args = self.args(path, b / "interrupted", "--losses", "proxy_anchor")
            original = app.save_torch

            def interrupt_after_epoch(path, value):
                original(path, value)
                if path.name == "last.pt" and value["epoch"] == 1:
                    raise KeyboardInterrupt()

            with (
                patch.object(app, "TigerViT", TinyViT),
                patch.object(app, "save_torch", interrupt_after_epoch),
            ):
                with self.assertRaises(KeyboardInterrupt):
                    app.train(args)
            args.resume = True
            with patch.object(app, "TigerViT", TinyViT):
                app.train(args)
            full = self.args(path, b / "full", "--losses", "proxy_anchor")
            with patch.object(app, "TigerViT", TinyViT):
                app.train(full)
            resumed = torch.load(
                b / "interrupted/proxy_anchor_seed42/last.pt", weights_only=False
            )
            uninterrupted = torch.load(
                b / "full/proxy_anchor_seed42/last.pt", weights_only=False
            )
            for key, value in resumed["model"].items():
                self.assertTrue(torch.equal(value, uninterrupted["model"][key]))
            for key, value in resumed["objective"].items():
                self.assertTrue(torch.equal(value, uninterrupted["objective"][key]))

    def test_epoch_progress_is_rendered(self):
        import contextlib, io

        with tempfile.TemporaryDirectory() as t:
            b = Path(t)
            path, root = self.fixture(b)
            args = self.args(path, b / "runs", "--losses", "triplet_hard")
            args.no_progress = False
            output = io.StringIO()
            with (
                patch.object(app, "TigerViT", TinyViT),
                contextlib.redirect_stderr(output),
            ):
                app.train(args)
            self.assertIn("epoch 1/2", output.getvalue())
            self.assertIn("epoch 2/2", output.getvalue())
            self.assertIn("validation", output.getvalue())

    def test_pretrained_baseline_uses_features(self):
        with tempfile.TemporaryDirectory() as t:
            b = Path(t)
            path, root = self.fixture(b)
            args = app.parser().parse_args(
                [
                    "baseline",
                    "--manifest",
                    str(path),
                    "--output",
                    str(b / "baseline"),
                    "--device",
                    "cpu",
                    "--no-progress",
                ]
            )
            with patch.object(app, "TigerViT", TinyViT):
                app.baseline(args)
            z = np.load(b / "baseline/validation_embeddings.npz")["embeddings"]
            self.assertEqual(
                z.shape[1], 3
            )  # Tiny fixture features, not its 128D random head.
            self.assertTrue((b / "baseline/validation.json").exists())


if __name__ == "__main__":
    unittest.main()
