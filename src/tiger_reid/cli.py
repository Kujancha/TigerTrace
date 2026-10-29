"""One entry point for data preparation, experiments and comparisons."""

import argparse
import json
from pathlib import Path

COMMANDS = {
    "prepare": "Audit and copy raw train/test data using annotations beside the images.",
    "split": "Create fixed train/validation/test identity and gallery/query assignments.",
    "doctor": "Check Python, dependencies and CUDA availability.",
    "baseline": "Evaluate unfine-tuned pretrained features on validation images.",
    "pilot": "Run a small three-loss GPU smoke test (one seed, two epochs).",
    "train": "Run the full three-loss, three-seed training grid.",
    "evaluate": "Evaluate validation-selected checkpoints on the held-out test partition.",
    "compare": "Generate result tables and comparison figures.",
    "test": "Run the synthetic software test suite (no tiger data or pretrained download).",
}


def parser():
    p = argparse.ArgumentParser(
        prog="python run.py",
        description="ATRW tiger re-identification: a reproducible ViT loss comparison.",
    )
    p.add_argument(
        "--data-dir",
        default="data",
        help="Raw input directory (default: data under project root). Put global options before the command.",
    )
    p.add_argument(
        "--work-dir",
        default="artifacts",
        help="All generated outputs (default: artifacts under project root).",
    )
    sub = p.add_subparsers(dest="command", required=True)
    for command, description in COMMANDS.items():
        sub.add_parser(command, help=description, add_help=False)
    return p


def resolve(base, value):
    path = Path(value)
    return (path if path.is_absolute() else base / path).resolve()


def prepare_command(raw, work, extras):
    from .prepare import prepare, prepare_unlabeled

    p = argparse.ArgumentParser(
        prog="python run.py prepare", description=COMMANDS["prepare"]
    )
    p.add_argument(
        "--audit-only", action="store_true", help="Check inputs without copying images."
    )
    group = p.add_mutually_exclusive_group()
    group.add_argument("--train-only", action="store_true")
    group.add_argument("--test-only", action="store_true")
    a = p.parse_args(extras)
    train_root = raw / "train"
    test_root = raw / "test"
    required = []
    if not a.test_only:
        required += [
            train_root / "reid_list_train.csv",
            train_root / "reid_keypoints_train.json",
        ]
    if not a.train_only:
        required += [
            test_root / "reid_list_test.csv",
            test_root / "reid_keypoints_test.json",
        ]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise ValueError(
            "Missing annotations. Extract them beside the corresponding images:\n  "
            + "\n  ".join(missing)
        )
    # Preflight both outputs before copying either dataset.
    for name in (
        ["test"] if a.test_only else ["train"] if a.train_only else ["train", "test"]
    ):
        dest = work / "prepared" / name
        if (
            (dest / "labeled").exists()
            or (dest / "images").exists()
            or (dest / ".staging").exists()
        ):
            raise ValueError(
                f"{dest} already contains prepared images. Continue with split/train, or choose a new --work-dir. Existing copies are not overwritten."
            )
    if not a.test_only:
        audit = prepare(
            train_root,
            train_root / "reid_list_train.csv",
            work / "prepared/train",
            keypoints=train_root / "reid_keypoints_train.json",
            copy=not a.audit_only,
        )
        print(
            f"Train: {audit['retained_images']} distinct labeled images, {audit['annotation_identities']} identity IDs; {audit['status']}."
        )
        print(
            f"Unmatched source images left untouched: {len(audit['unmatched_image_paths'])}"
        )
    if not a.train_only:
        audit = prepare_unlabeled(
            test_root,
            work / "prepared/test",
            keypoints=test_root / "reid_keypoints_test.json",
            copy=not a.audit_only,
            test_list=test_root / "reid_list_test.csv",
        )
        print(
            f"Public test: {audit['readable_images']} readable listed images; {audit['status']}; no identity labels inferred."
        )
    print("Audit reports:", work / "prepared")


def main(argv=None, project_root=None):
    root = Path(project_root or Path.cwd()).resolve()
    args, extras = parser().parse_known_args(argv)
    raw = resolve(root, args.data_dir)
    work = resolve(root, args.work_dir)
    manifest = work / "experiment/manifest.json"
    prepared = work / "prepared/train"
    try:
        if args.command == "test":
            import unittest

            p = argparse.ArgumentParser(
                prog="python run.py test", description=COMMANDS["test"]
            )
            p.add_argument("--pattern", default="test_*.py")
            a = p.parse_args(extras)
            suite = unittest.defaultTestLoader.discover(
                str(root / "tests"), pattern=a.pattern, top_level_dir=str(root)
            )
            result = unittest.TextTestRunner(verbosity=2).run(suite)
            if not result.wasSuccessful():
                raise SystemExit(1)
            return
        if args.command == "prepare":
            return prepare_command(raw, work, extras)
        if args.command == "split":
            from .split import create

            p = argparse.ArgumentParser(
                prog="python run.py split", description=COMMANDS["split"]
            )
            p.add_argument("--seed", type=int, default=42)
            p.add_argument("--train-ids", type=int, default=60)
            p.add_argument("--val-ids", type=int, default=14)
            p.add_argument("--test-ids", type=int, default=33)
            p.add_argument("--gallery-images", type=int, default=2)
            p.add_argument("--near-threshold", type=int, default=4)
            a = p.parse_args(extras)
            _, summary = create(
                prepared,
                work / "experiment",
                a.seed,
                a.train_ids,
                a.val_ids,
                a.test_ids,
                a.gallery_images,
                a.near_threshold,
            )
            print(json.dumps(summary, indent=2))
            print("Manifest:", manifest)
            print(
                "Review gallery/query assignments and flagged pairs before interpreting model results."
            )
            return
        if args.command in {"doctor", "train", "pilot", "baseline"}:
            from . import train

            engine = train.parser()
            engine.prog = "python run.py"
            if args.command == "doctor":
                engine.parse_args(["doctor", *extras])
                import torch

                print(
                    json.dumps(
                        train.environment(
                            torch.device("cuda" if torch.cuda.is_available() else "cpu")
                        ),
                        indent=2,
                    )
                )
                print("CUDA available:", torch.cuda.is_available())
                return
            mode = "train" if args.command == "pilot" else args.command
            output = work / (
                "pilot"
                if args.command == "pilot"
                else "pretrained_baseline"
                if mode == "baseline"
                else "runs"
            )
            defaults = [
                "--manifest",
                str(manifest),
                "--prepared",
                str(prepared),
                "--output",
                str(output),
            ]
            if args.command == "pilot":
                defaults += ["--seeds", "42", "--epochs", "2", "--steps", "5"]
            a = engine.parse_args([mode, *defaults, *extras])
            if a.eval_batch < 1:
                raise ValueError("eval-batch must be positive.")
            if mode == "baseline":
                return train.baseline(a)
            if a.epochs < 1 or a.patience < 1 or a.lr <= 0:
                raise ValueError("epochs, patience, lr must be positive.")
            if len(set(a.seeds)) != len(a.seeds) or len(set(a.losses)) != len(a.losses):
                raise ValueError("Duplicate seed or loss.")
            if any(s < 0 or s > 2**32 - 1 for s in a.seeds):
                raise ValueError("Seeds must be in 0..2**32-1.")
            return train.train(a)
        if args.command == "evaluate":
            from . import evaluate

            p = evaluate.parser()
            p.prog = "python run.py evaluate"
            a = p.parse_args(
                [
                    "--manifest",
                    str(manifest),
                    "--prepared",
                    str(prepared),
                    "--runs",
                    str(work / "runs"),
                    "--output",
                    str(work / "evaluation"),
                    *extras,
                ]
            )
            return evaluate.evaluate(a)
        if args.command == "compare":
            from . import compare

            p = compare.parser()
            p.prog = "python run.py compare"
            p.add_argument(
                "--examples",
                action="store_true",
                help="Also build a failure-image contact sheet.",
            )
            a = p.parse_args(
                [
                    "--evaluation",
                    str(work / "evaluation"),
                    "--output",
                    str(work / "comparison"),
                    *extras,
                ]
            )
            if a.examples:
                a.manifest = a.manifest or str(manifest)
                a.prepared = a.prepared or str(prepared)
            return compare.compare(a)
    except ModuleNotFoundError as e:
        raise SystemExit(
            f"Missing dependency: {e.name}. Install a matching GPU-enabled torch/torchvision pair, then run: python -m pip install -r requirements.txt"
        ) from e
    except (ValueError, OSError, KeyError) as e:
        raise SystemExit("ERROR: " + str(e)) from e
