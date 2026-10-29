"""Create immutable, annotation-identity-disjoint ATRW experiment manifests.
Consumes the copied output of prepare_atrw.py, not the old project split.
"""

import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from PIL import Image, ImageOps


def file_hash(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def dhash(image):
    image = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    pixels = image.load()
    result = 0
    for y in range(8):
        for x in range(8):
            result = (result << 1) | int(pixels[x, y] > pixels[x + 1, y])
    return result


def create(
    prepared,
    output,
    seed=42,
    train_count=60,
    val_count=14,
    test_count=33,
    gallery_count=2,
    near_threshold=4,
):
    root = Path(prepared).resolve()
    out = Path(output).resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError(
            "Output directory is not empty. Use a new --output; saved splits are never overwritten."
        )
    if out == root or root in out.parents or out in root.parents:
        raise ValueError("Split output must be separate from the prepared image tree.")
    audit_path = root / "audit.json"
    inventory_path = root / "inventory.csv"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("status") != "copied" or not (root / "labeled").is_dir():
        raise ValueError(
            "First run the labeled preparation command with --copy. This is not the unlabeled public test inventory."
        )
    ann = root / "annotations/reid_list_train.csv"
    if not ann.is_file() or file_hash(ann) != audit.get("annotation_sha256"):
        raise ValueError(
            "Original annotation copy is missing or differs from its recorded hash."
        )
    official = {}
    with ann.open(newline="", encoding="utf-8-sig") as f:
        for number, row in enumerate(csv.reader(f), 1):
            if not row or not any(v.strip() for v in row):
                continue
            if number == 1 and row[0].strip().lower() in {"id", "identity", "tiger_id"}:
                continue
            if len(row) != 2 or row[1].strip() in official:
                raise ValueError(
                    "Invalid or duplicate filename in original annotations."
                )
            official[row[1].strip()] = row[0].strip()
    with inventory_path.open(newline="", encoding="utf-8-sig") as f:
        inventory = list(csv.DictReader(f))
    if len(inventory) != audit.get("annotation_rows") or len(inventory) != len(
        official
    ):
        raise ValueError(
            "Inventory row count does not match the official annotation count."
        )
    if len({r["filename"] for r in inventory}) != len(inventory):
        raise ValueError("Inventory filenames are not unique.")
    for r in inventory:
        if official.get(r["filename"]) != r["identity"]:
            raise ValueError(
                "Inventory identity does not match the official CSV: " + r["filename"]
            )
        if r["included"].lower() not in {"true", "false"}:
            raise ValueError("Invalid included flag in inventory.")
    rows = []
    seen_pixels = {}
    hashes = {}
    for r in inventory:
        if r["included"].lower() == "false":
            continue
        p = (root / r["prepared_relative_path"]).resolve()
        if not p.is_file() or (root / "labeled").resolve() not in p.parents:
            raise ValueError("Missing image or path outside labeled/: " + r["filename"])
        if file_hash(p) != r["sha256"]:
            raise ValueError(
                "Image byte hash differs from preparation: " + r["filename"]
            )
        with Image.open(p) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            im.load()
            decoded = hashlib.sha256(str(im.size).encode() + im.tobytes()).hexdigest()
            if decoded != r["decoded_sha256"]:
                raise ValueError("Decoded image hash differs: " + r["filename"])
            if decoded in seen_pixels:
                raise ValueError(
                    "Exact duplicate remains in retained inventory: "
                    + r["filename"]
                    + " / "
                    + seen_pixels[decoded]
                )
            seen_pixels[decoded] = r["filename"]
            hashes[r["filename"]] = dhash(im)
        rows.append(
            {
                "filename": r["filename"],
                "identity": r["identity"],
                "path": p.relative_to(root).as_posix(),
                "sha256": r["sha256"],
                "decoded_sha256": decoded,
            }
        )
    counts = Counter(r["identity"] for r in rows)
    if len(rows) != audit.get("retained_images") or dict(counts) != audit.get(
        "identity_counts"
    ):
        raise ValueError("Retained image counts differ from preparation audit.")
    if min(train_count, val_count, test_count) < 2 or gallery_count < 1:
        raise ValueError(
            "Each identity partition needs >=2 identities; gallery count must be >=1."
        )
    if train_count + val_count + test_count != len(counts):
        raise ValueError(
            f"Partition counts sum to {train_count + val_count + test_count}, but dataset has {len(counts)} identities. Set counts explicitly; do not drop identities silently."
        )
    if not 0 <= near_threshold <= 64:
        raise ValueError("near threshold must be in 0..64.")
    rng = random.Random(seed)
    ids = sorted(counts, key=lambda i: (int(i), i))
    rng.shuffle(ids)
    split_ids = {
        "train": sorted(ids[:train_count], key=int),
        "val": sorted(ids[train_count : train_count + val_count], key=int),
        "test": sorted(ids[train_count + val_count :], key=int),
    }
    membership = {
        identity: split
        for split, identities in split_ids.items()
        for identity in identities
    }
    by_id = defaultdict(list)
    for row in sorted(rows, key=lambda r: r["filename"]):
        by_id[row["identity"]].append(row)
    for identity in sorted(by_id, key=int):
        items = by_id[identity]
        split = membership[identity]
        needed = 2 if split == "train" else gallery_count + 1
        if len(items) < needed:
            raise ValueError(
                f"Identity {identity} has {len(items)} retained images, needs {needed}."
            )
        rng.shuffle(items)
        for j, row in enumerate(items):
            row["split"] = split
            row["role"] = (
                "train"
                if split == "train"
                else ("gallery" if j < gallery_count else "query")
            )
    rows.sort(key=lambda r: (r["split"], int(r["identity"]), r["filename"]))
    candidates = []
    for identity, items in by_id.items():
        galleries = [r for r in items if r["role"] == "gallery"]
        queries = [r for r in items if r["role"] == "query"]
        for g in galleries:
            for q in queries:
                distance = (hashes[g["filename"]] ^ hashes[q["filename"]]).bit_count()
                if distance <= near_threshold:
                    candidates.append(
                        {
                            "split": g["split"],
                            "identity": identity,
                            "gallery_path": g["path"],
                            "query_path": q["path"],
                            "dhash_distance": distance,
                        }
                    )
    candidates.sort(
        key=lambda r: (
            r["dhash_distance"],
            r["split"],
            int(r["identity"]),
            r["gallery_path"],
            r["query_path"],
        )
    )
    manifest = {
        "schema_version": 1,
        "dataset": "ATRW official labeled training pool",
        "protocol": "custom annotation-identity-disjoint split; image-level gallery/query",
        "prepared_root": str(root),
        "seed": seed,
        "gallery_images_per_evaluation_identity": gallery_count,
        "identity_splits": split_ids,
        "inventory_sha256": file_hash(inventory_path),
        "annotation_sha256": file_hash(ann),
        "records": rows,
        "capture_session_separation_verified": False,
        "review_status": "pending_manual_review",
        "limitations": [
            "This is a custom evaluation split, not the official unlabeled competition test.",
            "Identity disjointness follows official annotation IDs; no biological equivalence beyond those labels is inferred.",
            "Capture session metadata was not provided. Gallery/query session separation is unverified.",
            "dHash flags candidate near-duplicates only; it can miss burst frames and flag unrelated poses. Review manually.",
        ],
    }
    summary = {
        "identity_counts": {s: len(v) for s, v in split_ids.items()},
        "image_counts": {s: sum(r["split"] == s for r in rows) for s in split_ids},
        "role_counts": dict(Counter(r["split"] + "/" + r["role"] for r in rows)),
        "total_images": len(rows),
        "near_duplicate_candidates": len(candidates),
        "dhash_threshold": near_threshold,
        "exact_duplicate_check": "passed",
        "identity_overlap_check": "passed",
        "capture_session_check": "not available",
        "next_step": "Review near_duplicate_candidates.csv and gallery_query_review.csv before training. Do not tune this split using model accuracy.",
    }
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "manifest.json", manifest)
    write_json(out / "summary.json", summary)
    fields = [
        "filename",
        "identity",
        "path",
        "sha256",
        "decoded_sha256",
        "split",
        "role",
    ]
    write_csv(out / "assignments.csv", rows, fields)
    write_csv(
        out / "gallery_query_review.csv",
        [r for r in rows if r["split"] != "train"],
        fields,
    )
    write_csv(
        out / "near_duplicate_candidates.csv",
        candidates,
        ["split", "identity", "gallery_path", "query_path", "dhash_distance"],
    )
    # Hash the saved manifest so later training can record the exact protocol.
    (out / "manifest.sha256").write_text(
        file_hash(out / "manifest.json") + "  manifest.json\n", encoding="utf-8"
    )
    return manifest, summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--prepared",
        default="prepared/train",
        help="Directory containing audit.json, inventory.csv, labeled/ and annotations/.",
    )
    p.add_argument("--output", default="experiment")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-ids", type=int, default=60)
    p.add_argument("--val-ids", type=int, default=14)
    p.add_argument("--test-ids", type=int, default=33)
    p.add_argument("--gallery-images", type=int, default=2)
    p.add_argument("--near-threshold", type=int, default=4)
    a = p.parse_args()
    try:
        _, summary = create(
            a.prepared,
            a.output,
            a.seed,
            a.train_ids,
            a.val_ids,
            a.test_ids,
            a.gallery_images,
            a.near_threshold,
        )
    except (ValueError, OSError, KeyError) as e:
        p.exit(1, f"ERROR: {e}\n")
    print("Identity counts:", summary["identity_counts"])
    print("Image counts:", summary["image_counts"])
    print("Roles:", summary["role_counts"])
    print(
        "Near-duplicate pairs flagged for review:", summary["near_duplicate_candidates"]
    )
    print("Saved:", Path(a.output) / "manifest.json")
    print(
        "Images were not copied or moved. Review gallery/query pairs before training; session independence is not verified."
    )


if __name__ == "__main__":
    main()
