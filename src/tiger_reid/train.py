"""Train ATRW metric-learning experiments with epoch/batch progress bars."""

import os

# Must be set before CUDA initialization. Attention uses the unfused path by default.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse
import hashlib
import json
import math
import platform
import random
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from PIL import Image, ImageOps
from tqdm.auto import tqdm
import timm
from .data import (
    load_manifest,
    verify_train_val,
    BalancedBatches,
    save_json,
    save_csv,
    sha256,
)
from .model import TigerViT, BACKBONE
from .losses import MetricObjective, NAMES
from .metrics import retrieval


class Images(Dataset):
    def __init__(self, manifest, root, split, augment=False):
        if split not in {"train", "val"}:
            raise ValueError(
                "Training deliberately loads only training/validation images."
            )
        self.rows = [r for r in manifest["records"] if r["split"] == split]
        self.root = root
        ids = sorted({r["identity"] for r in self.rows})
        self.labels = [ids.index(r["identity"]) for r in self.rows]
        operations = [T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC)]
        if augment:
            operations += [
                T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1)
            ]
        self.transform = T.Compose(
            operations + [T.ToTensor(), T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
        )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        with Image.open(self.root / self.rows[index]["path"]) as im:
            x = self.transform(ImageOps.exif_transpose(im).convert("RGB"))
        return x, self.labels[index]


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def device_for(choice):
    if choice == "cuda" and not torch.cuda.is_available():
        raise ValueError(
            "CUDA unavailable. Run doctor and install a GPU-enabled PyTorch build. CPU is only for debugging."
        )
    return torch.device(choice)


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def save_torch(path, value):
    temp = path.with_name(path.name + ".tmp")
    torch.save(value, temp)
    temp.replace(path)


@torch.inference_mode()
def validate(
    model,
    dataset,
    device,
    batch_size,
    description,
    progress=True,
    raw=False,
    save_vectors=None,
):
    model.eval()
    vectors = []
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    for x, _ in tqdm(
        loader,
        desc=description,
        unit="batch",
        dynamic_ncols=True,
        disable=not progress,
        leave=False,
    ):
        z = model.features(x.to(device)) if raw else model(x.to(device))
        vectors.append(F.normalize(z.float(), dim=1).cpu().numpy())
    z = np.concatenate(vectors)
    if save_vectors:
        np.savez_compressed(
            save_vectors,
            embeddings=z,
            filenames=np.array([r["filename"] for r in dataset.rows]),
            identities=np.array([r["identity"] for r in dataset.rows]),
            roles=np.array([r["role"] for r in dataset.rows]),
        )
    return retrieval(z, dataset.rows)


def settings(args, manifest_hash, steps):
    keys = [
        "epochs",
        "patience",
        "p",
        "k",
        "blocks",
        "lr",
        "device",
        "eval_batch",
        "checkpointing",
        "no_amp",
        "fast_attention",
        "triplet_margin",
        "ms_alpha",
        "ms_beta",
        "ms_base",
        "ms_epsilon",
        "proxy_margin",
        "proxy_alpha",
    ]
    config = {key: getattr(args, key) for key in keys}
    config.update(
        steps_per_epoch=steps,
        manifest_sha256=manifest_hash,
        backbone=BACKBONE,
        embedding_dim=128,
        preprocessing="Bicubic Resize224 + train ColorJitter(.15,.15,.1); checkpoint mean/std=(.5,.5,.5); no flip",
        optimizer="Adam, no weight decay; same lr for backbone, head and trainable proxies",
        sampler="P identities x K distinct images; equal batch budget across losses",
    )
    config["source_sha256"] = {
        name: sha256(Path(__file__).parent / name)
        for name in ["train.py", "data.py", "model.py", "losses.py", "metrics.py"]
    }
    return config


def environment(device):
    import torchvision, pytorch_metric_learning

    return {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "torchvision": torchvision.__version__,
        "timm": timm.__version__,
        "pytorch_metric_learning": pytorch_metric_learning.__version__,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "determinism": "best effort; warn_only=True; no bitwise cross-hardware guarantee",
    }


def train(args):
    manifest, root, digest = load_manifest(args.manifest, args.prepared)
    progress = not args.no_progress
    device = device_for(args.device)
    timm.layers.set_fused_attn(args.fast_attention)
    verify_train_val(
        manifest,
        root,
        lambda rows: tqdm(
            rows,
            desc="Verify train/validation images",
            unit="image",
            disable=not progress,
        ),
    )
    dataset = Images(manifest, root, "train", augment=True)
    validation = Images(manifest, root, "val")
    steps = (
        args.steps
        if args.steps is not None
        else math.ceil(len(dataset) / (args.p * args.k))
    )
    shared = settings(args, digest, steps)
    plan = [(loss, seed) for loss in args.losses for seed in args.seeds]
    # Catch incompatible or existing runs before starting any expensive work.
    for loss_name, seed in plan:
        run = Path(args.output) / f"{loss_name}_seed{seed}"
        if run.exists():
            if not args.resume:
                raise ValueError(
                    f"{run} exists. Use --resume or a new --output folder."
                )
            stored = json.loads((run / "config.json").read_text())
            for key in [
                "torch",
                "torchvision",
                "timm",
                "pytorch_metric_learning",
                "device",
            ]:
                if stored["environment"][key] != environment(device)[key]:
                    raise ValueError(
                        f"Environment changed since {run}; use a new output directory."
                    )
            if stored["settings"] != shared:
                raise ValueError(
                    f"Config/source/manifest differs from {run}; cannot resume."
                )
            if not (run / "last.pt").is_file():
                raise ValueError(
                    f"{run} has no completed epoch checkpoint. Use a new folder or move this incomplete run aside."
                )
    for number, (loss_name, seed) in enumerate(plan, 1):
        run = Path(args.output) / f"{loss_name}_seed{seed}"
        status_path = run / "status.json"
        if (
            args.resume
            and status_path.exists()
            and json.loads(status_path.read_text()).get("status") == "complete"
        ):
            tqdm.write(f"[{number}/{len(plan)}] Skipping completed {run.name}")
            continue
        resume_file = run / "last.pt"
        resuming = args.resume and resume_file.exists()
        seed_all(seed)
        model = TigerViT(
            pretrained=not resuming,
            blocks=args.blocks,
            checkpointing=args.checkpointing,
        ).to(device)
        initial_head_hash = hashlib.sha256(
            b"".join(
                p.detach().cpu().numpy().tobytes()
                for p in model.embedding_head.parameters()
            )
        ).hexdigest()
        objective = MetricObjective(
            loss_name,
            len(set(dataset.labels)),
            args.triplet_margin,
            args.ms_alpha,
            args.ms_beta,
            args.ms_base,
            args.ms_epsilon,
            args.proxy_margin,
            args.proxy_alpha,
        ).to(device)
        params = [p for p in model.parameters() if p.requires_grad] + list(
            objective.parameters()
        )
        optimizer = torch.optim.Adam(params, lr=args.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs
        )
        amp = device.type == "cuda" and not args.no_amp
        scaler = torch.amp.GradScaler("cuda", enabled=amp)
        sampler = BalancedBatches(dataset.labels, args.p, args.k, steps, seed)
        loader = DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=0,
            pin_memory=device.type == "cuda",
        )
        history = []
        best = -1.0
        stale = 0
        first_epoch = 1
        best_epoch = 0
        config = {
            "loss": loss_name,
            "seed": seed,
            "settings": shared,
            "environment": environment(device),
            "initial_head_sha256": initial_head_hash,
            "trainable_parameters": sum(p.numel() for p in params),
            "capture_session_separation_verified": manifest.get(
                "capture_session_separation_verified", False
            ),
            "manifest_review_status": manifest.get("review_status", "unspecified"),
        }
        if resuming:
            # Only load your own trusted checkpoints. They include optimizer Python objects.
            state = torch.load(resume_file, map_location=device, weights_only=False)
            model.load_state_dict(state["model"])
            objective.load_state_dict(state["objective"])
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            scaler.load_state_dict(state["scaler"])
            history = state["history"]
            best = state["best_val_map"]
            stale = state["stale"]
            best_epoch = state["best_epoch"]
            first_epoch = state["epoch"] + 1
            config = state["config"]
        else:
            run.mkdir(parents=True)
            save_json(run / "config.json", config)
            save_json(run / "manifest.json", manifest)
            (run / "environment.txt").write_text(
                subprocess.check_output(
                    [sys.executable, "-m", "pip", "freeze"], text=True
                ),
                encoding="utf-8",
            )
            initial = validate(
                model,
                validation,
                device,
                args.eval_batch,
                f"{run.name}: initial validation",
                progress,
            )
            save_json(run / "initial_validation.json", initial)
        save_json(
            status_path, {"status": "running", "last_completed_epoch": first_epoch - 1}
        )
        tqdm.write(
            f"\n[{number}/{len(plan)}] {loss_name} | seed {seed} | {steps} batches/epoch | batch {args.p * args.k}"
        )
        try:
            for epoch in range(first_epoch, args.epochs + 1):
                if stale >= args.patience:
                    break
                # Epoch-local RNG makes an epoch-boundary resume reproducible on the same environment.
                seed_all(seed + epoch * 1000003)
                sampler.epoch = epoch
                model.train()
                objective.train()
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                sync(device)
                tick = time.perf_counter()
                total = 0.0
                current_lr = optimizer.param_groups[0]["lr"]
                bar = tqdm(
                    loader,
                    desc=f"{loss_name} s{seed} | epoch {epoch}/{args.epochs}",
                    unit="batch",
                    dynamic_ncols=True,
                    disable=not progress,
                )
                for batch, (x, y) in enumerate(bar, 1):
                    x, y = (
                        x.to(device, non_blocking=True),
                        y.to(device, non_blocking=True),
                    )
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(device_type=device.type, enabled=amp):
                        z = model(x)
                    with torch.autocast(device_type=device.type, enabled=False):
                        loss = objective(z.float(), y)
                    if not torch.isfinite(loss):
                        raise RuntimeError(
                            "Non-finite loss; inspect data and learning rate."
                        )
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(params, 5.0)
                    scaler.step(optimizer)
                    scaler.update()
                    total += float(loss.detach())
                    bar.set_postfix(loss=f"{total / batch:.4f}", lr=f"{current_lr:.1e}")
                sync(device)
                train_seconds = time.perf_counter() - tick
                val_tick = time.perf_counter()
                result = validate(
                    model,
                    validation,
                    device,
                    args.eval_batch,
                    f"{loss_name} s{seed}: validation",
                    progress,
                )
                sync(device)
                validation_seconds = time.perf_counter() - val_tick
                score = result["summary"]["map"]
                improved = score > best + 1e-8
                if improved:
                    best, stale, best_epoch = score, 0, epoch
                else:
                    stale += 1
                row = {
                    "epoch": epoch,
                    "train_loss": total / steps,
                    "lr": current_lr,
                    "train_seconds": train_seconds,
                    "seconds_per_batch": train_seconds / steps,
                    "validation_seconds": validation_seconds,
                    "peak_allocated_vram_gb": torch.cuda.max_memory_allocated(device)
                    / 1024**3
                    if device.type == "cuda"
                    else 0.0,
                    "peak_reserved_vram_gb": torch.cuda.max_memory_reserved(device)
                    / 1024**3
                    if device.type == "cuda"
                    else 0.0,
                    "val_rank1": result["summary"]["rank1"],
                    "val_rank5": result["summary"]["rank5"],
                    "val_map": score,
                }
                history.append(row)
                scheduler.step()
                if improved:
                    save_torch(
                        run / "best.pt",
                        {
                            "schema_version": 1,
                            "model": model.state_dict(),
                            "config": config,
                            "epoch": epoch,
                            "validation": result["summary"],
                            "manifest_sha256": digest,
                        },
                    )
                    save_json(run / "best_validation.json", result)
                state = {
                    "schema_version": 1,
                    "model": model.state_dict(),
                    "objective": objective.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                    "history": history,
                    "epoch": epoch,
                    "best_val_map": best,
                    "best_epoch": best_epoch,
                    "stale": stale,
                    "config": config,
                }
                save_torch(resume_file, state)
                save_json(run / "history.json", history)
                save_csv(run / "history.csv", history)
                save_json(
                    status_path,
                    {
                        "status": "running",
                        "last_completed_epoch": epoch,
                        "best_epoch": best_epoch,
                    },
                )
                tqdm.write(
                    f"  val Rank1={row['val_rank1']:.3f} mAP={score:.3f} | train {train_seconds:.1f}s | best epoch {best_epoch} | patience {stale}/{args.patience}"
                )
            # Refresh saved logs from checkpoint after a resume of an already finished epoch budget.
            save_json(run / "history.json", history)
            save_csv(run / "history.csv", history)
            summary = {
                "loss": loss_name,
                "seed": seed,
                "epochs": len(history),
                "best_epoch": best_epoch,
                "best_val_map": best,
                "total_train_seconds": sum(r["train_seconds"] for r in history),
                "total_validation_seconds": sum(
                    r["validation_seconds"] for r in history
                ),
                "manifest_sha256": digest,
            }
            save_json(run / "run_summary.json", summary)
            save_json(status_path, {"status": "complete", **summary})
        except BaseException as e:
            save_json(
                status_path,
                {
                    "status": "interrupted"
                    if isinstance(e, KeyboardInterrupt)
                    else "failed",
                    "error": str(e),
                    "resume_note": "Resume restarts after the last saved complete epoch.",
                },
            )
            raise
        finally:
            # Drop checkpoint dictionaries and the final batch tensors before the next run.
            state = None
            x = y = z = loss = None
            del model, objective, optimizer, scheduler, scaler, params
            if device.type == "cuda":
                torch.cuda.empty_cache()
    tqdm.write(
        "\nTraining grid complete. Final test evaluation is a separate step; test images were not loaded."
    )


def baseline(args):
    m, root, digest = load_manifest(args.manifest, args.prepared)
    device = device_for(args.device)
    out = Path(args.output)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Baseline output is not empty; use a new directory.")
    verify_train_val(
        m,
        root,
        lambda rows: tqdm(
            rows, desc="Verify train/validation images", disable=args.no_progress
        ),
    )
    timm.layers.set_fused_attn(False)
    seed_all(42)
    model = TigerViT(pretrained=True, blocks=0).to(device)
    dataset = Images(m, root, "val")
    out.mkdir(parents=True, exist_ok=True)
    r = validate(
        model,
        dataset,
        device,
        args.eval_batch,
        "Pretrained 768D feature baseline",
        not args.no_progress,
        raw=True,
        save_vectors=out / "validation_embeddings.npz",
    )
    r.update(
        manifest_sha256=digest,
        backbone=BACKBONE,
        embedding_dim=768,
        note="Normalized pretrained backbone features, no learned/random projection; dimension differs from trained 128D embeddings.",
    )
    save_json(out / "validation.json", r)
    save_json(out / "environment.json", environment(device))
    print(r["summary"])


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor")
    for name in ["train", "baseline"]:
        a = sub.add_parser(name)
        a.add_argument("--manifest", default="experiment/manifest.json")
        a.add_argument("--prepared", help="Override relocated prepared root")
        a.add_argument(
            "--output", default="runs" if name == "train" else "pretrained_baseline"
        )
        a.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
        a.add_argument("--eval-batch", type=int, default=8)
        a.add_argument("--no-progress", action="store_true")
        if name == "baseline":
            continue
        a.add_argument("--losses", choices=NAMES, nargs="+", default=list(NAMES))
        a.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
        a.add_argument("--epochs", type=int, default=30)
        a.add_argument("--patience", type=int, default=8)
        a.add_argument("--p", type=int, default=4)
        a.add_argument("--k", type=int, default=2)
        a.add_argument("--steps", type=int)
        a.add_argument("--blocks", type=int, default=4)
        a.add_argument("--lr", type=float, default=1e-4)
        a.add_argument("--checkpointing", action="store_true")
        a.add_argument("--no-amp", action="store_true")
        a.add_argument(
            "--fast-attention",
            action="store_true",
            help="Allow fused attention; may reduce reproducibility.",
        )
        a.add_argument("--resume", action="store_true")
        for flag, value in [
            ("triplet-margin", 0.3),
            ("ms-alpha", 2.0),
            ("ms-beta", 50.0),
            ("ms-base", 0.5),
            ("ms-epsilon", 0.1),
            ("proxy-margin", 0.1),
            ("proxy-alpha", 32.0),
        ]:
            a.add_argument("--" + flag, type=float, default=value)
    return p
