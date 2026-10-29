"""Prepare official ATRW identity CSV without renaming source images.
Audit by default; add --copy to materialize audited, labeled image folders.
"""

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

try:
    from PIL import Image, ImageOps
except ImportError:
    raise SystemExit(
        "Pillow is required. Install it with: python -m pip install Pillow"
    )

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def read_labels(path):
    rows = []
    seen = set()
    with Path(path).open(newline="", encoding="utf-8-sig") as f:
        for line, row in enumerate(csv.reader(f), 1):
            if not row or not any(x.strip() for x in row):
                continue
            if len(row) != 2:
                raise ValueError(
                    f"CSV line {line}: expected exactly identity,filename."
                )
            identity, name = [x.strip() for x in row]
            if line == 1 and identity.lower() in {"id", "identity", "tiger_id"}:
                continue
            if not re.fullmatch(r"\d+", identity):
                raise ValueError(
                    f"CSV line {line}: nonnumeric or unknown identity {identity!r}."
                )
            if not name or Path(name).name != name or "/" in name or "\\" in name:
                raise ValueError(
                    f"CSV line {line}: expected a bare image filename, got {name!r}."
                )
            if name in seen:
                raise ValueError(
                    f"CSV repeats filename {name!r}; inspect annotations before proceeding."
                )
            seen.add(name)
            rows.append(
                {
                    "identity": identity,
                    "filename": name,
                    "folder": "tiger_" + identity.zfill(3),
                }
            )
    if not rows:
        raise ValueError("CSV contains no identity rows.")
    mapping = defaultdict(set)
    for r in rows:
        mapping[r["folder"]].add(r["identity"])
    if any(len(v) > 1 for v in mapping.values()):
        raise ValueError(
            "Distinct original identity strings map to the same folder; annotations need review."
        )
    return rows


def prepare(images, labels, output, keypoints=None, copy=False):
    root, out = Path(images).resolve(), Path(output).resolve()
    labels = Path(labels).resolve()
    if not root.is_dir():
        raise ValueError(f"Image folder does not exist: {root}")
    if out == root or root in out.parents or out in root.parents:
        raise ValueError(
            "Output must be separate from the source image tree, not inside it or its parent."
        )
    if copy and ((out / "labeled").exists() or (out / ".staging").exists()):
        raise ValueError(
            "Output labeled/ or .staging/ already exists. Use a new --output folder; no files were overwritten."
        )
    rows = read_labels(labels)
    points = None
    if keypoints:
        points = json.loads(Path(keypoints).read_text(encoding="utf-8"))
        if not isinstance(points, dict):
            raise ValueError("Keypoint JSON must map filenames to coordinate arrays.")
    index = defaultdict(list)
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in EXTENSIONS:
            if root not in path.resolve().parents:
                raise ValueError(f"Image resolves outside source root: {path}")
            index[path.name].append(path)
    errors, duplicates, prepared = [], [], []
    hashes = {}
    all_zero, missing_keypoints = [], []
    for row in rows:
        name = row["filename"]
        matches = index.get(name, [])
        if len(matches) != 1:
            errors.append(
                {
                    "filename": name,
                    "issue": "missing" if not matches else "ambiguous_filename",
                    "paths": [p.relative_to(root).as_posix() for p in matches],
                }
            )
            continue
        path = matches[0]
        try:
            with Image.open(path) as im:
                im = ImageOps.exif_transpose(im).convert("RGB")
                im.load()
                pixel_hash = hashlib.sha256(
                    str(im.size).encode() + im.tobytes()
                ).hexdigest()
                width, height = im.size
        except Exception as e:
            errors.append({"filename": name, "issue": "unreadable", "detail": str(e)})
            continue
        if points is not None:
            value = points.get(name)
            if value is None:
                missing_keypoints.append(name)
            elif (
                not isinstance(value, list)
                or len(value) != 45
                or not all(
                    isinstance(v, (int, float)) and math.isfinite(v) for v in value
                )
                or any(v not in (0, 1, 2) for v in value[2::3])
            ):
                errors.append({"filename": name, "issue": "invalid_15_keypoint_array"})
            elif not any(value[2::3]):
                all_zero.append(name)
        current = {
            **row,
            "source_relative_path": path.relative_to(root).as_posix(),
            "prepared_relative_path": f"labeled/{row['folder']}/{name}",
            "width": width,
            "height": height,
            "sha256": sha_file(path),
            "decoded_sha256": pixel_hash,
            "included": True,
            "duplicate_of": "",
        }
        if pixel_hash in hashes:
            first = hashes[pixel_hash]
            if first["identity"] != row["identity"]:
                errors.append(
                    {
                        "filename": name,
                        "issue": "identical_image_conflicting_identity",
                        "other": first["filename"],
                        "identities": [first["identity"], row["identity"]],
                    }
                )
            else:
                current["included"] = False
                current["duplicate_of"] = first["filename"]
                duplicates.append(
                    {
                        "excluded": name,
                        "kept": first["filename"],
                        "identity": row["identity"],
                    }
                )
        else:
            hashes[pixel_hash] = current
        prepared.append(current)
    counts = Counter(r["identity"] for r in prepared if r["included"])
    for identity in sorted({r["identity"] for r in rows}, key=int):
        if counts[identity] < 2:
            errors.append(
                {
                    "identity": identity,
                    "issue": "fewer_than_two_distinct_images",
                    "count": counts[identity],
                }
            )
    names = {r["filename"] for r in rows}
    unmatched = sorted(
        p.relative_to(root).as_posix()
        for name, paths in index.items()
        if name not in names
        for p in paths
    )
    audit = {
        "source_root": str(root),
        "annotation_csv": str(labels),
        "annotation_sha256": sha_file(labels),
        "keypoints_sha256": sha_file(keypoints) if keypoints else None,
        "annotation_rows": len(rows),
        "annotation_identities": len({r["identity"] for r in rows}),
        "source_image_files": sum(map(len, index.values())),
        "retained_images": sum(counts.values()),
        "identity_counts": dict(sorted(counts.items(), key=lambda x: int(x[0]))),
        "unmatched_image_paths": unmatched,
        "same_identity_duplicates_excluded": duplicates,
        "keypoint_files_missing": missing_keypoints,
        "keypoint_files_without_visible_points": all_zero,
        "keypoint_extra_filenames": sorted(set(points or {}) - names),
        "errors": errors,
        "status": "failed" if errors else "audit_passed",
        "limitations": [
            "Exact decoded-image duplicates are checked; near-duplicate burst frames are not detected.",
            "Keypoints provide no capture-session labels here; session independence is not established.",
            "No train/validation/test split is created at this formatting step.",
            "Images absent from the identity CSV are not assigned an identity or copied.",
        ],
    }
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "audit.json", audit)
    if prepared:
        with (out / "inventory.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(prepared[0]))
            w.writeheader()
            w.writerows(prepared)
    if errors:
        raise ValueError(
            f"Audit failed with {len(errors)} issues; read {out / 'audit.json'}. No images copied."
        )
    if copy:
        stage = out / ".staging"
        stage.mkdir()
        for row in prepared:
            if not row["included"]:
                continue
            src = root / row["source_relative_path"]
            dest = stage / row["folder"] / row["filename"]
            dest.parent.mkdir(exist_ok=True)
            shutil.copy2(src, dest)
            if sha_file(dest) != row["sha256"]:
                raise ValueError(
                    f"Image changed during copying: {src}. Incomplete .staging left for inspection."
                )
        stage.rename(out / "labeled")
        provenance = out / "annotations"
        provenance.mkdir(exist_ok=True)
        shutil.copy2(labels, provenance / "reid_list_train.csv")
        if keypoints:
            shutil.copy2(keypoints, provenance / "reid_keypoints_train.json")
        write_json(
            out / "identity_mapping.json", {r["folder"]: r["identity"] for r in rows}
        )
        audit["status"] = "copied"
        write_json(out / "audit.json", audit)
    return audit


def prepare_unlabeled(images, output, keypoints=None, copy=False, test_list=None):
    """Inventory public test images without claiming identity ground truth."""
    root, out = Path(images).resolve(), Path(output).resolve()
    if not root.is_dir():
        raise ValueError(f"Image folder does not exist: {root}")
    if out == root or root in out.parents or out in root.parents:
        raise ValueError("Output must be separate from the source image tree.")
    if copy and ((out / "images").exists() or (out / ".staging").exists()):
        raise ValueError(
            "Output images/ or .staging/ exists. Use a new --output folder."
        )
    points = (
        json.loads(Path(keypoints).read_text(encoding="utf-8")) if keypoints else None
    )
    if points is not None and not isinstance(points, dict):
        raise ValueError("Keypoint JSON must map filenames to coordinate arrays.")
    paths = sorted(
        p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONS
    )
    requested = None
    if test_list:
        requested = set()
        with Path(test_list).open(newline="", encoding="utf-8-sig") as f:
            for number, row in enumerate(csv.reader(f), 1):
                if not row or not any(x.strip() for x in row):
                    continue
                if len(row) != 1:
                    raise ValueError(
                        f"Test list line {number}: expected one filename, not identity labels."
                    )
                name = row[0].strip()
                if not name or "/" in name or "\\" in name:
                    raise ValueError(
                        f"Test list line {number}: expected a bare filename."
                    )
                if name in requested:
                    raise ValueError(f"Test list repeats filename: {name}")
                requested.add(name)
        if not requested:
            raise ValueError("Test list is empty.")
    all_paths = paths
    unlisted = [
        p.relative_to(root).as_posix()
        for p in paths
        if requested is not None and p.name not in requested
    ]
    if requested is not None:
        paths = [p for p in paths if p.name in requested]
    errors, rows, duplicates, missing_points, invisible = [], [], [], [], []
    if requested is not None:
        for name in sorted(requested - {p.name for p in paths}):
            errors.append({"filename": name, "issue": "listed_test_image_missing"})
    seen, names = {}, Counter(p.name for p in paths)
    if not paths:
        errors.append({"issue": "no_image_files"})
    for path in paths:
        if root not in path.resolve().parents:
            raise ValueError(f"Image resolves outside source root: {path}")
        rel = path.relative_to(root).as_posix()
        try:
            with Image.open(path) as im:
                im = ImageOps.exif_transpose(im).convert("RGB")
                im.load()
                width, height = im.size
                pixel_hash = hashlib.sha256(
                    str(im.size).encode() + im.tobytes()
                ).hexdigest()
        except Exception as e:
            errors.append({"path": rel, "issue": "unreadable", "detail": str(e)})
            continue
        if pixel_hash in seen:
            duplicates.append({"path": rel, "identical_to": seen[pixel_hash]})
        else:
            seen[pixel_hash] = rel
        if requested is not None and names[path.name] > 1:
            errors.append({"filename": path.name, "issue": "ambiguous_listed_filename"})
        if points is not None:
            if names[path.name] > 1:
                errors.append(
                    {"path": rel, "issue": "ambiguous_basename_for_keypoints"}
                )
            value = points.get(path.name)
            if value is None:
                missing_points.append(rel)
            elif (
                not isinstance(value, list)
                or len(value) != 45
                or not all(
                    isinstance(v, (int, float)) and math.isfinite(v) for v in value
                )
                or any(v not in (0, 1, 2) for v in value[2::3])
            ):
                errors.append({"path": rel, "issue": "invalid_15_keypoint_array"})
            elif not any(value[2::3]):
                invisible.append(rel)
        rows.append(
            {
                "filename": path.name,
                "source_relative_path": rel,
                "prepared_relative_path": "images/" + rel,
                "width": width,
                "height": height,
                "sha256": sha_file(path),
                "decoded_sha256": pixel_hash,
            }
        )
    audit = {
        "mode": "unlabeled_test",
        "source_root": str(root),
        "source_image_files": len(all_paths),
        "test_list_sha256": sha_file(test_list) if test_list else None,
        "listed_image_count": len(requested) if requested is not None else None,
        "unlisted_image_paths": unlisted,
        "readable_images": len(rows),
        "identity_labels_available": False,
        "exact_duplicate_images": duplicates,
        "keypoint_files_missing": missing_points,
        "keypoint_files_without_visible_points": invisible,
        "keypoint_extra_filenames": sorted(set(points or {}) - set(names)),
        "keypoints_sha256": sha_file(keypoints) if keypoints else None,
        "errors": errors,
        "status": "failed" if errors else "audit_passed",
        "limitations": [
            "No test identity labels were supplied or inferred.",
            "No Rank-1, mAP, or unknown-rejection evaluation is supported by this inventory alone.",
            "Exact duplicates are reported but retained; no near-duplicate detection.",
        ],
    }
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "audit.json", audit)
    if rows:
        with (out / "inventory.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    if errors:
        raise ValueError(
            f"Test audit failed with {len(errors)} issues; read {out / 'audit.json'}. No images copied."
        )
    if copy:
        stage = out / ".staging"
        stage.mkdir()
        for row in rows:
            dest = stage / row["source_relative_path"]
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(root / row["source_relative_path"], dest)
            if sha_file(dest) != row["sha256"]:
                raise ValueError(
                    f"Image changed while copying: {row['filename']}. Inspect .staging."
                )
        stage.rename(out / "images")
        if keypoints:
            (out / "annotations").mkdir(exist_ok=True)
            shutil.copy2(keypoints, out / "annotations/reid_keypoints_test.json")
        if test_list:
            (out / "annotations").mkdir(exist_ok=True)
            shutil.copy2(test_list, out / "annotations/reid_list_test.csv")
        audit["status"] = "copied"
        write_json(out / "audit.json", audit)
    return audit


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--images", default="data/train")
    p.add_argument(
        "--unlabeled",
        action="store_true",
        help="Inventory test images without identity labels; --images must point to the test tree.",
    )
    p.add_argument("--labels", default="reid_list_train.csv")
    p.add_argument("--keypoints")
    p.add_argument(
        "--test-list",
        help="Official one-column test filename CSV; use with --unlabeled.",
    )
    p.add_argument("--output", default="prepared")
    p.add_argument(
        "--copy",
        action="store_true",
        help="Copy verified labeled images; default is audit only.",
    )
    a = p.parse_args()
    try:
        if a.unlabeled:
            audit = prepare_unlabeled(
                a.images, a.output, a.keypoints, a.copy, a.test_list
            )
        else:
            if a.test_list:
                raise ValueError("--test-list requires --unlabeled.")
            audit = prepare(a.images, a.labels, a.output, a.keypoints, a.copy)
    except (ValueError, OSError) as e:
        p.exit(1, f"ERROR: {e}\n")
    if a.unlabeled:
        print(
            "Test images:",
            audit["source_image_files"],
            "Readable:",
            audit["readable_images"],
        )
        print("Exact duplicate images reported:", len(audit["exact_duplicate_images"]))
        print("Status:", audit["status"])
        print(
            "Identity labels: not supplied or inferred. Test metrics cannot be computed yet."
        )
        print("Details:", Path(a.output) / "audit.json")
        return
    print(
        f"Annotation rows: {audit['annotation_rows']}; identities: {audit['annotation_identities']}"
    )
    print(
        f"Source image files: {audit['source_image_files']}; distinct labeled images retained: {audit['retained_images']}"
    )
    print(f"Unmatched images (left untouched): {len(audit['unmatched_image_paths'])}")
    print(
        f"Same-identity duplicate images excluded: {len(audit['same_identity_duplicates_excluded'])}"
    )
    print("Status:", audit["status"])
    print("Details:", Path(a.output) / "audit.json")
    if not a.copy:
        print(
            "Audit only: no images copied. Review the report, then repeat with --copy."
        )


if __name__ == "__main__":
    main()
