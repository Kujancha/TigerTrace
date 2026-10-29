import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import hashlib
import numpy as np
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F
from tiger_reid import train as train_atrw
from tiger_reid import evaluate as evaluate_atrw
from tiger_reid import compare as compare_results


class TinyViT(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.embedding_head = nn.Linear(3, 128)

    def features(self, x):
        return F.normalize(x.mean((2, 3)), dim=1)

    def forward(self, x):
        return F.normalize(self.embedding_head(x.mean((2, 3))), dim=1)


class EvaluationTests(unittest.TestCase):
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
                    name = f"{identity}_{j}.png"
                    p = root / name
                    im = Image.new("RGB", (12, 12), (int(identity) * 25, j * 35, 70))
                    im.save(p)
                    records.append(
                        {
                            "filename": name,
                            "identity": identity,
                            "path": name,
                            "split": split,
                            "role": "train"
                            if split == "train"
                            else ("gallery" if j < 2 else "query"),
                            "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                            "decoded_sha256": hashlib.sha256(
                                str(im.size).encode() + im.tobytes()
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
            "limitations": ["Synthetic test fixture only."],
        }
        path = b / "manifest.json"
        path.write_text(json.dumps(m))
        return path, root

    def train(self, b, path):
        args = train_atrw.parser().parse_args(
            [
                "train",
                "--manifest",
                str(path),
                "--output",
                str(b / "runs"),
                "--device",
                "cpu",
                "--seeds",
                "42",
                "--epochs",
                "1",
                "--steps",
                "1",
                "--no-progress",
            ]
        )
        with (
            patch.object(train_atrw, "TigerViT", TinyViT),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            train_atrw.train(args)

    def args(self, b, path, *extra):
        return evaluate_atrw.parser().parse_args(
            [
                "--manifest",
                str(path),
                "--runs",
                str(b / "runs"),
                "--output",
                str(b / "eval"),
                "--device",
                "cpu",
                "--seeds",
                "42",
                "--no-progress",
                *extra,
            ]
        )

    def test_full_train_evaluate_compare_and_resume(self):
        with tempfile.TemporaryDirectory() as t:
            b = Path(t)
            path, root = self.fixture(b)
            self.train(b, path)
            with (
                patch.object(evaluate_atrw, "TigerViT", TinyViT),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                evaluate_atrw.evaluate(self.args(b, path, "--include-pretrained"))
                evaluate_atrw.evaluate(
                    self.args(b, path, "--include-pretrained", "--resume")
                )
            for method in evaluate_atrw.METHODS:
                r = json.loads(
                    (b / "eval" / f"{method}_seed42/evaluation.json").read_text()
                )
                self.assertEqual(r["summary"]["gallery_images"], 4)
                self.assertEqual(r["summary"]["queries"], 4)
                self.assertIn("best_reference_path", r["queries"][0])
            args = compare_results.parser().parse_args(
                [
                    "--evaluation",
                    str(b / "eval"),
                    "--output",
                    str(b / "comparison"),
                    "--manifest",
                    str(path),
                ]
            )
            compare_results.compare(args)
            for name in [
                "runs.csv",
                "summary.csv",
                "RESULTS.md",
                "retrieval.png",
                "cmc.png",
                "learning_and_cost.png",
                "loss_curves.png",
                "similarity_distributions.png",
                "per_identity.png",
            ]:
                self.assertGreater((b / "comparison" / name).stat().st_size, 0)

    def test_changed_test_image_is_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            b = Path(t)
            path, root = self.fixture(b)
            self.train(b, path)
            (root / "6_0.png").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "Test image changed"):
                evaluate_atrw.evaluate(self.args(b, path))

    def test_incomplete_run_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            b = Path(t)
            path, root = self.fixture(b)
            self.train(b, path)
            (b / "runs/proxy_anchor_seed42/status.json").write_text(
                '{"status":"failed"}'
            )
            with self.assertRaisesRegex(ValueError, "incomplete"):
                evaluate_atrw.evaluate(self.args(b, path))

    def test_similarity_pair_counts(self):
        z = np.array([[1, 0], [0, 1], [1, 0], [0, 1]], dtype=float)
        rows = [
            {"filename": str(j), "identity": i, "path": str(j), "role": role}
            for j, (i, role) in enumerate(
                [("0", "gallery"), ("1", "gallery"), ("0", "query"), ("1", "query")]
            )
        ]
        r = evaluate_atrw.extra_retrieval(z, rows, evaluate_atrw.retrieval(z, rows))
        self.assertEqual(r["similarities"]["same_identity"], [1.0, 1.0])
        self.assertEqual(r["similarities"]["different_identity"], [0.0, 0.0])
        self.assertEqual(r["summary"]["rank1"], 1.0)
        self.assertEqual(r["queries"][0]["best_reference_path"], "0")


if __name__ == "__main__":
    unittest.main()
