import csv
import json
from pathlib import Path
import tempfile
import unittest
from PIL import Image
from tiger_reid.prepare import prepare, prepare_unlabeled


class PreparationTests(unittest.TestCase):
    def fixture(self, base):
        root = base / "data/train"
        root.mkdir(parents=True)
        rows = [["1", "a.jpg"], ["1", "b.jpg"], ["2", "c.jpg"], ["2", "d.jpg"]]
        for j, (_, name) in enumerate(rows):
            Image.new("RGB", (12, 12), (j * 50, 30, 20)).save(root / name)
        Image.new("RGB", (12, 12), "white").save(root / "extra.jpg")
        labels = base / "labels.csv"
        with labels.open("w", newline="") as f:
            csv.writer(f).writerows(rows)
        points = base / "keypoints.json"
        points.write_text(json.dumps({name: [0] * 45 for _, name in rows}))
        return root, labels, points

    def test_copy_and_provenance(self):
        with tempfile.TemporaryDirectory() as t:
            b = Path(t)
            root, labels, k = self.fixture(b)
            a = prepare(root, labels, b / "prepared", k, True)
            self.assertEqual(a["annotation_rows"], 4)
            self.assertEqual(a["annotation_identities"], 2)
            self.assertEqual(a["unmatched_image_paths"], ["extra.jpg"])
            self.assertEqual(len(list((b / "prepared/labeled").rglob("*.jpg"))), 4)
            self.assertEqual(len(list(root.glob("*.jpg"))), 5)
            self.assertEqual(
                json.loads((b / "prepared/identity_mapping.json").read_text()),
                {"tiger_001": "1", "tiger_002": "2"},
            )
            with self.assertRaises(ValueError):
                prepare(root, labels, b / "prepared", k, True)

    def test_missing_image_blocks_copy(self):
        with tempfile.TemporaryDirectory() as t:
            b = Path(t)
            root, labels, k = self.fixture(b)
            (root / "a.jpg").unlink()
            with self.assertRaises(ValueError):
                prepare(root, labels, b / "prepared", k, True)
            self.assertFalse((b / "prepared/labeled").exists())

    def test_conflicting_duplicate_blocks_copy(self):
        import shutil

        with tempfile.TemporaryDirectory() as t:
            b = Path(t)
            root, labels, k = self.fixture(b)
            shutil.copy2(root / "a.jpg", root / "c.jpg")
            with self.assertRaises(ValueError):
                prepare(root, labels, b / "prepared", k, True)
            self.assertIn(
                "identical_image_conflicting_identity",
                (b / "prepared/audit.json").read_text(),
            )

    def test_ambiguous_filename_blocks_copy(self):
        import shutil

        with tempfile.TemporaryDirectory() as t:
            b = Path(t)
            root, labels, k = self.fixture(b)
            (root / "nested").mkdir()
            shutil.copy2(root / "a.jpg", root / "nested/a.jpg")
            with self.assertRaises(ValueError):
                prepare(root, labels, b / "prepared", k, True)
            self.assertIn("ambiguous_filename", (b / "prepared/audit.json").read_text())

    def test_same_id_duplicate_exclusion(self):
        import shutil

        with tempfile.TemporaryDirectory() as t:
            b = Path(t)
            root, labels, k = self.fixture(b)
            shutil.copy2(root / "a.jpg", root / "e.jpg")
            with labels.open("a", newline="") as f:
                csv.writer(f).writerow(["1", "e.jpg"])
            a = prepare(root, labels, b / "prepared", copy=True)
            self.assertEqual(a["retained_images"], 4)
            self.assertEqual(len(a["same_identity_duplicates_excluded"]), 1)
            self.assertTrue((root / "e.jpg").exists())

    def test_unlabeled_copy_without_fabricated_ids(self):
        with tempfile.TemporaryDirectory() as t:
            b = Path(t)
            root, labels, k = self.fixture(b)
            a = prepare_unlabeled(root, b / "test_output", copy=True)
            self.assertFalse(a["identity_labels_available"])
            self.assertEqual(a["readable_images"], 5)
            self.assertTrue((b / "test_output/images/a.jpg").exists())
            self.assertNotIn(
                "identity",
                (b / "test_output/inventory.csv").read_text().splitlines()[0],
            )
            self.assertEqual(len(list(root.glob("*.jpg"))), 5)
            with self.assertRaises(ValueError):
                prepare_unlabeled(root, b / "test_output", copy=True)

    def test_unlabeled_empty_blocks_copy(self):
        with tempfile.TemporaryDirectory() as t:
            b = Path(t)
            root = b / "test"
            root.mkdir()
            with self.assertRaises(ValueError):
                prepare_unlabeled(root, b / "out", copy=True)
            self.assertFalse((b / "out/images").exists())

    def test_unlabeled_keypoints_and_nested_paths(self):
        with tempfile.TemporaryDirectory() as t:
            b = Path(t)
            root, labels, k = self.fixture(b)
            (root / "nested").mkdir()
            (root / "a.jpg").rename(root / "nested/a.jpg")
            a = prepare_unlabeled(root, b / "out", keypoints=k, copy=True)
            self.assertEqual(a["keypoint_files_missing"], ["extra.jpg"])
            self.assertTrue((b / "out/images/nested/a.jpg").exists())
            self.assertTrue((b / "out/annotations/reid_keypoints_test.json").exists())

    def test_test_filename_list_filters_and_checks_completeness(self):
        with tempfile.TemporaryDirectory() as t:
            b = Path(t)
            root, labels, k = self.fixture(b)
            listing = b / "test.csv"
            listing.write_text("a.jpg\nb.jpg\n")
            a = prepare_unlabeled(root, b / "out", copy=True, test_list=listing)
            self.assertEqual(a["readable_images"], 2)
            self.assertEqual(len(a["unlisted_image_paths"]), 3)
            self.assertFalse((b / "out/images/extra.jpg").exists())
            self.assertTrue((b / "out/annotations/reid_list_test.csv").exists())
            listing.write_text("missing.jpg\n")
            with self.assertRaises(ValueError):
                prepare_unlabeled(root, b / "bad", copy=True, test_list=listing)
            self.assertFalse((b / "bad/images").exists())


if __name__ == "__main__":
    unittest.main()
