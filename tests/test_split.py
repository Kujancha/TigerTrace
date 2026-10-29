import csv
import json
from pathlib import Path
import tempfile
import unittest
from PIL import Image
from tiger_reid.split import create, file_hash
import hashlib


class SplitTests(unittest.TestCase):
    def fixture(self, b):
        root = b / "prepared/train"
        (root / "annotations").mkdir(parents=True)
        labels = []
        inventory = []
        for identity in range(8):
            for j in range(5):
                name = f"{identity:03d}_{j}.png"
                p = root / "labeled" / f"tiger_{identity:03d}" / name
                p.parent.mkdir(exist_ok=True, parents=True)
                im = Image.new("RGB", (16, 16), (identity * 25, j * 35, 50))
                im.putpixel((j, j), (255, 255, 255))
                im.save(p)
                decoded = hashlib.sha256(
                    str(im.size).encode() + im.tobytes()
                ).hexdigest()
                labels.append([str(identity), name])
                inventory.append(
                    {
                        "filename": name,
                        "identity": str(identity),
                        "included": "True",
                        "prepared_relative_path": p.relative_to(root).as_posix(),
                        "sha256": file_hash(p),
                        "decoded_sha256": decoded,
                    }
                )
        ann = root / "annotations/reid_list_train.csv"
        with ann.open("w", newline="") as f:
            csv.writer(f).writerows(labels)
        with (root / "inventory.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(inventory[0]))
            w.writeheader()
            w.writerows(inventory)
        audit = {
            "status": "copied",
            "annotation_sha256": file_hash(ann),
            "annotation_rows": 40,
            "retained_images": 40,
            "identity_counts": {str(i): 5 for i in range(8)},
        }
        (root / "audit.json").write_text(json.dumps(audit))
        return root

    def test_disjoint_complete_and_reproducible(self):
        with tempfile.TemporaryDirectory() as t:
            b = Path(t)
            root = self.fixture(b)
            m, s = create(root, b / "one", train_count=4, val_count=2, test_count=2)
            m2, s2 = create(root, b / "two", train_count=4, val_count=2, test_count=2)
            self.assertEqual(m, m2)
            self.assertEqual(s, s2)
            self.assertEqual(s["identity_counts"], {"train": 4, "val": 2, "test": 2})
            self.assertEqual(len(m["records"]), 40)
            self.assertEqual(s["role_counts"]["val/gallery"], 4)
            self.assertEqual(s["role_counts"]["test/query"], 6)
            groups = [set(v) for v in m["identity_splits"].values()]
            self.assertFalse(
                groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]
            )
            for identity in m["identity_splits"]["test"]:
                images = [r for r in m["records"] if r["identity"] == identity]
                self.assertEqual(sum(r["role"] == "gallery" for r in images), 2)
                self.assertEqual(sum(r["role"] == "query" for r in images), 3)
            with self.assertRaises(ValueError):
                create(root, b / "one", train_count=4, val_count=2, test_count=2)

    def test_changed_image_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            b = Path(t)
            root = self.fixture(b)
            p = next((root / "labeled").rglob("*.png"))
            p.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "hash differs"):
                create(root, b / "exp", train_count=4, val_count=2, test_count=2)
            self.assertFalse((b / "exp").exists())

    def test_identity_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            b = Path(t)
            root = self.fixture(b)
            path = root / "inventory.csv"
            with path.open(newline="") as f:
                rows = list(csv.DictReader(f))
            rows[0]["identity"] = "999"
            with path.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
            with self.assertRaisesRegex(ValueError, "official CSV"):
                create(root, b / "exp", train_count=4, val_count=2, test_count=2)

    def test_requires_query_and_correct_counts(self):
        with tempfile.TemporaryDirectory() as t:
            b = Path(t)
            root = self.fixture(b)
            with self.assertRaisesRegex(ValueError, "dataset has"):
                create(root, b / "exp", train_count=4, val_count=2, test_count=3)
            with self.assertRaisesRegex(ValueError, "needs 6"):
                create(
                    root,
                    b / "exp",
                    train_count=4,
                    val_count=2,
                    test_count=2,
                    gallery_count=5,
                )

    def test_near_duplicate_review_output(self):
        with tempfile.TemporaryDirectory() as t:
            b = Path(t)
            root = self.fixture(b)
            _, s = create(
                root,
                b / "exp",
                train_count=4,
                val_count=2,
                test_count=2,
                near_threshold=64,
            )
            self.assertEqual(s["near_duplicate_candidates"], 24)
            self.assertTrue((b / "exp/near_duplicate_candidates.csv").is_file())
            self.assertEqual(len(list((root / "labeled").rglob("*.png"))), 40)


if __name__ == "__main__":
    unittest.main()
