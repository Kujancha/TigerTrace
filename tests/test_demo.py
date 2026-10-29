"""Demo tests use synthetic images/features; no weights or tiger dataset needed."""

import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
from tiger_reid.demo import Gallery, Encoder, build_app


class FakeEncoder:
    fingerprint = "synthetic-model"
    manifest_hash = ""
    config = {"loss": "synthetic", "seed": 42}

    def __call__(self, image):
        result = np.zeros(128, dtype=np.float32)
        result[:3] = np.asarray(image).mean((0, 1))
        return result


class DemoTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.encoder = FakeEncoder()
        self.gallery = Gallery(self.root / "gallery.sqlite3", self.encoder)
        self.red = Image.new("RGB", (12, 12), (255, 0, 0))
        self.green = Image.new("RGB", (12, 12), (0, 255, 0))
        self.blue = Image.new("RGB", (12, 12), (0, 0, 255))

    def fixture(self):
        from tests.test_evaluate import EvaluationTests

        manifest, prepared = EvaluationTests().fixture(self.root)
        self.encoder.manifest_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
        return manifest, prepared

    def test_persistence_duplicate_and_conflicting_label(self):
        self.gallery.enroll(self.red, "tiger_A", "New tiger", True)
        reopened = Gallery(self.gallery.path, self.encoder)
        self.assertEqual(reopened.inventory(), [["tiger_A", 1]])
        reopened.enroll(self.red, "tiger_A", "Existing tiger", True)
        self.assertEqual(reopened.inventory(), [["tiger_A", 1]])
        with self.assertRaisesRegex(ValueError, "already belongs"):
            reopened.enroll(self.red, "tiger_B", "New tiger", True)

    def test_distinct_top_three_threshold_and_self_match(self):
        for name, im in [
            ("A", self.red),
            ("B", self.green),
            ("C", self.blue),
            ("D", Image.new("RGB", (12, 12), (255, 255, 0))),
        ]:
            self.gallery.enroll(im, name, "New tiger", True)
        self.gallery.enroll(
            Image.new("RGB", (12, 12), (250, 0, 0)), "A", "Existing tiger", True
        )
        status, rows, images = self.gallery.identify(self.red, 0.8)
        self.assertIn("Candidate match: A", status)
        self.assertIn("self-match", status)
        self.assertEqual(len(rows), 3)
        self.assertEqual(len({r[1] for r in rows}), 3)
        self.assertEqual(len(images), 3)
        self.assertAlmostEqual(rows[0][2], 1)
        self.gallery.delete("C", True)
        status, _, _ = self.gallery.identify(self.blue, 0.8)
        self.assertIn("No confident match", status)

    def test_confirmation_empty_invalid_and_delete(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            self.gallery.identify(self.red, 0.6)
        with self.assertRaises(ValueError):
            self.gallery.enroll(self.red, "A", "New tiger", False)
        with self.assertRaises(ValueError):
            self.gallery.enroll(self.red, "../escape", "New tiger", True)
        self.gallery.enroll(self.red, "A", "New tiger", True)
        with self.assertRaises(ValueError):
            self.gallery.delete("A", False)
        with self.assertRaises(ValueError):
            self.gallery.identify(self.red, float("nan"))
        self.gallery.delete("A", True)
        self.assertEqual(self.gallery.inventory(), [])

    def test_checkpoint_mismatch_rejected(self):
        different = FakeEncoder()
        different.fingerprint = "another-model"
        with self.assertRaisesRegex(ValueError, "another checkpoint"):
            Gallery(self.gallery.path, different)

    def test_import_only_gallery_exclusion_and_no_queries(self):
        manifest, prepared = self.fixture()
        self.gallery.import_manifest(manifest, prepared, "test-gallery", "7")
        self.assertEqual(self.gallery.inventory(), [["6", 2]])
        with self.assertRaisesRegex(ValueError, "already registered"):
            self.gallery.import_manifest(manifest, prepared, "test-gallery", "6")
        self.gallery.import_manifest(manifest, prepared, "train")
        self.assertEqual(
            dict(self.gallery.inventory()), {"0": 4, "1": 4, "2": 4, "3": 4, "6": 2}
        )

    def test_import_atomic_and_manifest_hash(self):
        manifest, prepared = self.fixture()
        (prepared / "7_1.png").write_bytes(b"bad image")
        with self.assertRaisesRegex(ValueError, "changed"):
            self.gallery.import_manifest(manifest, prepared, "test-gallery")
        self.assertEqual(self.gallery.inventory(), [])
        self.encoder.manifest_hash = "wrong"
        with self.assertRaisesRegex(ValueError, "Manifest differs"):
            self.gallery.import_manifest(manifest, prepared, "train")

    def test_real_checkpoint_format_and_preprocessing(self):
        import torch
        from tests.test_evaluate import TinyViT
        from tiger_reid import model

        network = TinyViT()
        path = self.root / "best.pt"
        torch.save(
            {
                "model": network.state_dict(),
                "config": {"settings": {"blocks": 4}},
                "manifest_sha256": "abc",
            },
            path,
        )
        with patch.object(model, "TigerViT", TinyViT):
            encoder = Encoder(path, "cpu")
        vector = encoder(self.red)
        self.assertEqual(vector.shape, (128,))
        self.assertAlmostEqual(float(np.linalg.norm(vector)), 1, places=5)
        from tiger_reid.evaluate import TestImages

        self.red.save(self.root / "image.png")
        dataset = TestImages(
            {"records": [{"split": "test", "path": "image.png"}]}, self.root
        )
        self.assertTrue(torch.equal(encoder.transform(self.red), dataset[0]))

    @unittest.skipUnless(
        importlib.util.find_spec("gradio"), "optional Gradio not installed"
    )
    def test_gradio_build_and_callbacks(self):
        app = build_app(self.gallery, self.root / "manifest.json", self.root)
        functions = [item.fn for item in app.fns.values() if item.fn]
        save = next(
            fn
            for fn in functions
            if getattr(fn, "__wrapped__", None) and fn.__wrapped__.__name__ == "do_save"
        )
        result = save(self.red, "A", "New tiger", True)
        self.assertEqual(result[1], [["A", 1]])
        self.assertFalse(result[-1])
        predict = next(
            fn
            for fn in functions
            if getattr(fn, "__wrapped__", None)
            and fn.__wrapped__.__name__ == "identify"
        )
        self.assertIn("Candidate match: A", predict(self.red, 0.6)[0])
        app.close()
