"""Final test retrieval for validation-selected fresh ATRW checkpoints. No training."""

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from PIL import Image, ImageOps
from tqdm.auto import tqdm
import timm
from .data import load_manifest, sha256, save_json, save_csv
from .model import TigerViT, BACKBONE
from .metrics import retrieval

METHODS = ("triplet_hard", "multi_similarity", "proxy_anchor")


class TestImages(Dataset):
    def __init__(self, manifest, root):
        self.rows = [r for r in manifest["records"] if r["split"] == "test"]
        self.root = root
        self.transform = T.Compose(
            [
                T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize([0.5] * 3, [0.5] * 3),
            ]
        )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        with Image.open(self.root / self.rows[i]["path"]) as image:
            return self.transform(ImageOps.exif_transpose(image).convert("RGB"))


def verify_test(rows, root, progress=True):
    for row in tqdm(
        rows, desc="Verify final-test images", unit="image", disable=not progress
    ):
        p = root / row["path"]
        if sha256(p) != row["sha256"]:
            raise ValueError("Test image changed: " + row["path"])
        with Image.open(p) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            im.load()
            decoded = hashlib.sha256(str(im.size).encode() + im.tobytes()).hexdigest()
        if decoded != row["decoded_sha256"]:
            raise ValueError("Test decoded image differs: " + row["path"])


def extra_retrieval(vectors, rows, result):
    vectors = np.asarray(vectors, dtype=np.float32)
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    gallery = [i for i, r in enumerate(rows) if r["role"] == "gallery"]
    queries = [i for i, r in enumerate(rows) if r["role"] == "query"]
    same, different = [], []
    for q, record in zip(queries, result["queries"]):
        scores = np.clip(vectors[gallery] @ vectors[q], -1, 1)
        image_order = np.argsort(-scores, kind="stable")
        # Display the best reference image of the predicted identity (identity ties are resolved in retrieval()).
        predicted_gallery = [
            j
            for j in image_order
            if rows[gallery[j]]["identity"] == record["predicted"]
        ]
        best = gallery[predicted_gallery[0]]
        record.update(
            query_path=rows[q]["path"], best_reference_path=rows[best]["path"]
        )
        for j, index in enumerate(gallery):
            (
                same if rows[index]["identity"] == rows[q]["identity"] else different
            ).append(float(scores[j]))
    result["similarities"] = {
        "same_identity": same,
        "different_identity": different,
        "note": "Descriptive pair scores; pairs are dependent and not calibrated probabilities.",
    }
    return result


def run_plan(runs, methods, seeds, digest):
    plan = []
    shared = None
    for method in methods:
        for seed in seeds:
            run = Path(runs) / f"{method}_seed{seed}"
            if not (run / "best.pt").is_file():
                raise ValueError("Missing best checkpoint: " + str(run))
            status = json.loads((run / "status.json").read_text())
            if status.get("status") != "complete":
                raise ValueError("Training run is incomplete: " + str(run))
            config = json.loads((run / "config.json").read_text())
            settings = config["settings"]
            if settings["manifest_sha256"] != digest:
                raise ValueError("Manifest differs from training: " + str(run))
            if config["loss"] != method or config["seed"] != seed:
                raise ValueError("Run folder/config mismatch.")
            if settings["backbone"] != BACKBONE or settings["embedding_dim"] != 128:
                raise ValueError("Unsupported architecture: " + str(run))
            if shared is None:
                shared = settings
            elif shared != settings:
                raise ValueError(
                    "Runs have differing settings/source/budgets; use a consistent run grid."
                )
            history = json.loads((run / "history.json").read_text())
            if not history:
                raise ValueError("Empty training history.")
            plan.append((run, method, seed, config, history))
    return plan, shared


@torch.inference_mode()
def extract(model, dataset, device, batch_size, description, progress=True, raw=False):
    model.eval()
    vectors = []
    for x in tqdm(
        DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0),
        desc=description,
        unit="batch",
        dynamic_ncols=True,
        disable=not progress,
    ):
        x = x.to(device)
        z = model.features(x) if raw else model(x)
        vectors.append(z.float().cpu().numpy())
    return np.concatenate(vectors)


def evaluate(args):
    m, root, digest = load_manifest(args.manifest, args.prepared)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable. Check your PyTorch GPU installation.")
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive.")
    if len(set(args.methods)) != len(args.methods) or len(set(args.seeds)) != len(
        args.seeds
    ):
        raise ValueError("Duplicate method or seed.")
    plan, settings = run_plan(args.runs, args.methods, args.seeds, digest)
    # Fail before evaluating any checkpoints if an incompatible output already exists.
    out = Path(args.output)
    specification = {
        "schema_version": 1,
        "manifest_sha256": digest,
        "methods": args.methods,
        "seeds": args.seeds,
        "include_pretrained": args.include_pretrained,
        "settings": settings,
    }
    if out.exists() and any(out.iterdir()):
        if not args.resume:
            raise ValueError(
                "Evaluation output exists. Use --resume to continue or choose a new --output."
            )
        if (
            not (out / "evaluation_config.json").exists()
            or json.loads((out / "evaluation_config.json").read_text()) != specification
        ):
            raise ValueError("Existing evaluation config differs; use a new output.")
    dataset = TestImages(m, root)
    verify_test(dataset.rows, root, not args.no_progress)
    timm.layers.set_fused_attn(settings["fast_attention"])
    torch.use_deterministic_algorithms(True, warn_only=True)
    out.mkdir(parents=True, exist_ok=True)
    save_json(out / "evaluation_config.json", specification)
    review = {
        "capture_session_separation_verified": m.get(
            "capture_session_separation_verified", False
        ),
        "review_status": m.get("review_status", "unspecified"),
        "limitations": m.get("limitations", []),
    }
    save_json(out / "protocol_notes.json", review)
    for run, method, seed, config, history in plan:
        dest = out / run.name
        fingerprint = sha256(run / "best.pt")
        if (dest / "evaluation.json").exists():
            old = json.loads((dest / "evaluation.json").read_text())
            if (
                not args.resume
                or old["checkpoint_sha256"] != fingerprint
                or old["manifest_sha256"] != digest
            ):
                raise ValueError(
                    "Existing evaluation/checkpoint mismatch: " + str(dest)
                )
            tqdm.write("Skipping completed " + run.name)
            continue
        # Trusted local checkpoints only. Never load unknown pickle checkpoint files.
        state = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
        if state["manifest_sha256"] != digest or state["config"] != config:
            raise ValueError("Checkpoint/config mismatch: " + str(run))
        model = TigerViT(pretrained=False, blocks=settings["blocks"]).to(device)
        model.load_state_dict(state["model"])
        del state["model"]
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        tick = time.perf_counter()
        vectors = extract(
            model,
            dataset,
            device,
            args.batch_size,
            "Test " + run.name,
            not args.no_progress,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        inference_seconds = time.perf_counter() - tick
        result = extra_retrieval(
            vectors, dataset.rows, retrieval(vectors, dataset.rows)
        )
        result.update(
            method=method,
            seed=seed,
            manifest_sha256=digest,
            checkpoint_sha256=fingerprint,
            checkpoint_epoch=state["epoch"],
            embedding_dim=128,
            inference_seconds=inference_seconds,
            validation_at_selected_epoch=state["validation"],
            protocol_notes=review,
            training={
                "epochs": len(history),
                "total_train_seconds": sum(r["train_seconds"] for r in history),
                "seconds_per_batch": float(
                    np.mean([r["seconds_per_batch"] for r in history])
                ),
                "peak_allocated_vram_gb": max(
                    r["peak_allocated_vram_gb"] for r in history
                ),
            },
            training_settings=settings,
        )
        dest.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            dest / "embeddings.npz",
            embeddings=vectors,
            filenames=np.array([r["filename"] for r in dataset.rows]),
            identities=np.array([r["identity"] for r in dataset.rows]),
            roles=np.array([r["role"] for r in dataset.rows]),
        )
        save_csv(
            dest / "per_query.csv",
            [{k: v for k, v in q.items() if k != "top5"} for q in result["queries"]],
        )
        save_json(dest / "history.json", history)
        save_json(dest / "evaluation.json", result)  # Completion marker written last.
        tqdm.write(run.name + ": " + str(result["summary"]))
        del model, state, vectors
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if args.include_pretrained:
        dest = out / "pretrained_features"
        if not (dest / "evaluation.json").exists():
            timm.layers.set_fused_attn(False)
            torch.manual_seed(42)
            model = TigerViT(pretrained=True, blocks=0).to(device)
            vectors = extract(
                model,
                dataset,
                device,
                args.batch_size,
                "Test pretrained features",
                not args.no_progress,
                raw=True,
            )
            result = extra_retrieval(
                vectors, dataset.rows, retrieval(vectors, dataset.rows)
            )
            result.update(
                method="pretrained_features",
                seed=None,
                manifest_sha256=digest,
                embedding_dim=768,
                backbone=BACKBONE,
                protocol_notes=review,
                note="Single deterministic pretrained feature extractor; no projection or fine-tuning. Not a matched-dimension ablation.",
            )
            dest.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                dest / "embeddings.npz",
                embeddings=vectors,
                filenames=np.array([r["filename"] for r in dataset.rows]),
                identities=np.array([r["identity"] for r in dataset.rows]),
                roles=np.array([r["role"] for r in dataset.rows]),
            )
            save_json(dest / "evaluation.json", result)
            tqdm.write("pretrained_features: " + str(result["summary"]))
            del model, vectors
    save_json(
        out / "status.json",
        {
            "status": "complete",
            "models": len(plan),
            "pretrained_baseline": args.include_pretrained,
        },
    )
    print("Evaluation complete:", out)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", default="experiment/manifest.json")
    p.add_argument("--prepared")
    p.add_argument("--runs", default="runs")
    p.add_argument("--output", default="evaluation")
    p.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--include-pretrained", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--no-progress", action="store_true")
    return p
