"""Create paper-ready static comparisons from completed ATRW test evaluations."""

import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .data import save_json, save_csv

COLORS = {
    "triplet_hard": "#357caa",
    "multi_similarity": "#d88643",
    "proxy_anchor": "#479677",
    "pretrained_features": "#777777",
}
LABELS = {
    "triplet_hard": "Batch-hard Triplet",
    "multi_similarity": "Multi-Similarity",
    "proxy_anchor": "Proxy Anchor",
    "pretrained_features": "Pretrained features (768D)",
}


def load_results(root):
    root = Path(root)
    if (
        not (root / "status.json").is_file()
        or json.loads((root / "status.json").read_text()).get("status") != "complete"
    ):
        raise ValueError("Evaluation is incomplete; finish evaluate_atrw.py first.")
    config = json.loads((root / "evaluation_config.json").read_text())
    results = []
    for method in config["methods"]:
        for seed in config["seeds"]:
            path = root / f"{method}_seed{seed}" / "evaluation.json"
            r = json.loads(path.read_text())
            if (
                r["method"] != method
                or r["seed"] != seed
                or r["manifest_sha256"] != config["manifest_sha256"]
            ):
                raise ValueError("Evaluation identity/config mismatch.")
            if r["training_settings"] != config["settings"]:
                raise ValueError("Training settings differ.")
            results.append(r)
    baseline = None
    if config["include_pretrained"]:
        baseline = json.loads(
            (root / "pretrained_features/evaluation.json").read_text()
        )
        if baseline["manifest_sha256"] != config["manifest_sha256"]:
            raise ValueError("Baseline uses a different split.")
    identity_sets = {
        tuple(sorted(r["per_identity"]))
        for r in results + ([baseline] if baseline else [])
    }
    if len(identity_sets) != 1:
        raise ValueError("Evaluations use different test identity sets.")
    return config, results, baseline


def average_std(values):
    return float(np.mean(values)), float(np.std(values, ddof=1)) if len(
        values
    ) > 1 else None


def figure_save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def make_error_sheet(results, manifest, prepared, output):
    from PIL import Image, ImageOps
    from .data import load_manifest

    m, root, digest = load_manifest(manifest, prepared)
    if any(r["manifest_sha256"] != digest for r in results):
        raise ValueError("Manifest differs from evaluation.")
    # Fixed first training seed per method, not a best-test-seed selection.
    examples = []
    for method in sorted({r["method"] for r in results}):
        r = min((r for r in results if r["method"] == method), key=lambda r: r["seed"])
        failures = sorted(
            (q for q in r["queries"] if q["rank"] > 1), key=lambda q: q["filename"]
        )[:2]
        examples.extend((method, r["seed"], q) for q in failures)
    if not examples:
        return
    fig, axes = plt.subplots(
        len(examples), 2, figsize=(8, 3 * len(examples)), squeeze=False
    )
    for axes_row, (method, seed, q) in zip(axes, examples):
        for ax, key, title in zip(
            axes_row,
            ["query_path", "best_reference_path"],
            [
                f"Query ID {q['identity']}",
                f"Predicted ID {q['predicted']}, correct rank {q['rank']}",
            ],
        ):
            p = (root / q[key]).resolve()
            if root not in p.parents:
                raise ValueError("Unsafe image path.")
            with Image.open(p) as im:
                ax.imshow(ImageOps.exif_transpose(im).convert("RGB"))
            ax.set_title(f"{LABELS[method]} / seed {seed}\n{title}", fontsize=9)
            ax.axis("off")
    figure_save(fig, Path(output) / "failure_examples.png")


def compare(args):
    config, results, baseline = load_results(args.evaluation)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    methods = config["methods"]
    summaries = []
    flat = []
    for r in results:
        flat.append(
            {
                "loss": r["method"],
                "seed": r["seed"],
                **r["summary"],
                "best_epoch": r["checkpoint_epoch"],
                **r["training"],
                "test_embedding_seconds": r["inference_seconds"],
            }
        )
    save_csv(out / "runs.csv", flat)
    for method in methods:
        values = [r for r in flat if r["loss"] == method]
        row = {"loss": method, "runs": len(values)}
        for key in [
            "rank1",
            "rank5",
            "map",
            "total_train_seconds",
            "seconds_per_batch",
            "peak_allocated_vram_gb",
        ]:
            row[key + "_mean"], row[key + "_std"] = average_std(
                [r[key] for r in values]
            )
        summaries.append(row)
    save_csv(out / "summary.csv", summaries)
    save_json(out / "summary.json", summaries)
    paired = []
    if "triplet_hard" in methods:
        ref = {r["seed"]: r for r in flat if r["loss"] == "triplet_hard"}
        for r in flat:
            if r["loss"] == "triplet_hard":
                continue
            paired.append(
                {
                    "loss": r["loss"],
                    "seed": r["seed"],
                    **{
                        k + "_delta_pp": 100 * (r[k] - ref[r["seed"]][k])
                        for k in ["rank1", "rank5", "map"]
                    },
                }
            )
        save_csv(out / "paired_vs_triplet.csv", paired)
    if baseline:
        save_json(out / "pretrained_baseline.json", baseline)
    failures = [
        {"loss": r["method"], "seed": r["seed"], **q}
        for r in results
        for q in r["queries"]
        if q["rank"] > 1
    ]
    save_json(out / "failed_queries.json", failures)
    plt.style.use("seaborn-v0_8-whitegrid")
    display = methods + (["pretrained_features"] if baseline else [])
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    for ax, key, title in zip(
        axes,
        ["rank1", "rank5", "map"],
        ["Identity Rank-1", "Identity Rank-5", "Image retrieval mAP"],
    ):
        means = [100 * row[key + "_mean"] for row in summaries]
        errors = [100 * (row[key + "_std"] or 0) for row in summaries]
        if baseline:
            means.append(100 * baseline["summary"][key])
            errors.append(0)
        ax.bar(
            range(len(display)),
            means,
            yerr=errors,
            capsize=4,
            color=[COLORS[m] for m in display],
        )
        ax.set_xticks(
            range(len(display)), [LABELS[m] for m in display], rotation=30, ha="right"
        )
        ax.set(title=title, ylabel="Macro average (%)", ylim=(0, 105))
    fig.suptitle(
        "Final test retrieval; error bars: sample SD across training seeds\nPretrained baseline is one 768D extractor, not a matched-dimension comparison",
        fontsize=11,
    )
    figure_save(fig, out / "retrieval.png")
    fig, ax = plt.subplots(figsize=(8, 5))
    n = len(results[0]["per_identity"])
    ranks = np.arange(1, n + 1)
    for method in display:
        group = (
            [r for r in results if r["method"] == method]
            if method != "pretrained_features"
            else [baseline]
        )
        curves = []
        for r in group:
            by_identity = {
                i: [q["rank"] for q in r["queries"] if q["identity"] == i]
                for i in r["per_identity"]
            }
            curves.append(
                [
                    np.mean([np.mean(np.array(v) <= k) for v in by_identity.values()])
                    for k in ranks
                ]
            )
        mean = np.mean(curves, axis=0) * 100
        ax.plot(
            ranks,
            mean,
            label=LABELS[method],
            color=COLORS[method],
            linestyle="--" if method == "pretrained_features" else "-",
        )
        if len(curves) > 1:
            std = np.std(curves, axis=0, ddof=1) * 100
            ax.fill_between(
                ranks,
                np.maximum(0, mean - std),
                np.minimum(100, mean + std),
                color=COLORS[method],
                alpha=0.13,
            )
    ax.set(
        xlabel="Rank (unique identities)",
        ylabel="Macro identification rate (%)",
        title="Cumulative matching curves",
        ylim=(0, 101),
        xlim=(1, max(2, n)),
    )
    ax.legend()
    figure_save(fig, out / "cmc.png")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for r in results:
        h = json.loads(
            (
                Path(args.evaluation)
                / f"{r['method']}_seed{r['seed']}"
                / "history.json"
            ).read_text()
        )
        axes[0].plot(
            [x["epoch"] for x in h],
            [100 * x["val_map"] for x in h],
            color=COLORS[r["method"]],
            alpha=0.65,
            label=f"{LABELS[r['method']]} / {r['seed']}",
        )
        axes[1].scatter(
            r["training"]["total_train_seconds"] / 60,
            100 * r["summary"]["map"],
            color=COLORS[r["method"]],
            alpha=0.8,
        )
    axes[0].set(
        xlabel="Epoch",
        ylabel="Validation macro mAP (%)",
        title="Validation trajectories",
    )
    axes[0].legend(fontsize=6)
    axes[1].set(
        xlabel="Recorded training time (minutes)",
        ylabel="Test macro mAP (%)",
        title="Quality versus observed cost (each point is a run)",
    )
    figure_save(fig, out / "learning_and_cost.png")
    fig, axes = plt.subplots(
        len(methods), 1, figsize=(8, 2.8 * len(methods)), squeeze=False
    )
    for ax, method in zip(axes[:, 0], methods):
        for r in [r for r in results if r["method"] == method]:
            h = json.loads(
                (
                    Path(args.evaluation) / f"{method}_seed{r['seed']}" / "history.json"
                ).read_text()
            )
            ax.plot(
                [x["epoch"] for x in h],
                [x["train_loss"] for x in h],
                label=f"seed {r['seed']}",
            )
        ax.set(
            title=LABELS[method] + " — native objective scale",
            xlabel="Epoch",
            ylabel="Training loss",
        )
        ax.legend(fontsize=8)
    figure_save(fig, out / "loss_curves.png")
    fig, axes = plt.subplots(
        len(display), 1, figsize=(8, 2.7 * len(display)), squeeze=False
    )
    for ax, method in zip(axes[:, 0], display):
        r = (
            min((r for r in results if r["method"] == method), key=lambda r: r["seed"])
            if method != "pretrained_features"
            else baseline
        )
        for key, label in [
            ("same_identity", "Same identity"),
            ("different_identity", "Different identity"),
        ]:
            ax.hist(
                r["similarities"][key],
                bins=np.linspace(-1, 1, 61),
                density=True,
                alpha=0.55,
                label=label,
            )
        ax.set(
            title=LABELS[method]
            + (f" / fixed seed {r['seed']}" if r["seed"] is not None else ""),
            xlabel="Cosine similarity",
            ylabel="Density",
        )
        ax.legend()
    figure_save(fig, out / "similarity_distributions.png")
    identities = sorted(results[0]["per_identity"], key=int)
    matrix = []
    for method in display:
        group = (
            [r for r in results if r["method"] == method]
            if method != "pretrained_features"
            else [baseline]
        )
        matrix.append(
            [
                np.mean([r["per_identity"][i]["rank1"] for r in group]) * 100
                for i in identities
            ]
        )
    fig, ax = plt.subplots(figsize=(max(9, len(identities) * 0.3), 3.5))
    im = ax.imshow(matrix, aspect="auto", vmin=0, vmax=100, cmap="YlGnBu")
    ax.set_xticks(range(len(identities)), identities, rotation=90)
    ax.set_yticks(range(len(display)), [LABELS[m] for m in display])
    ax.set(
        xlabel="Original annotation identity ID",
        title="Per-identity Rank-1 (mean over training seeds)",
    )
    fig.colorbar(im, ax=ax, label="Rank-1 (%)")
    figure_save(fig, out / "per_identity.png")
    lines = [
        "# Final test comparison",
        "",
        f"Training seeds: {config['seeds']}. Manifest: `{config['manifest_sha256']}`.",
        "",
        "| Configuration | Runs | Rank-1 (%) | Rank-5 (%) | mAP (%) |",
        "|---|---:|---:|---:|---:|",
    ]
    for s in summaries:

        def fmt(k):
            return f"{100 * s[k + '_mean']:.2f}" + (
                f" ± {100 * s[k + '_std']:.2f}"
                if s[k + "_std"] is not None
                else " (one run)"
            )

        lines.append(
            f"| {LABELS[s['loss']]} | {s['runs']} | {fmt('rank1')} | {fmt('rank5')} | {fmt('map')} |"
        )
    if baseline:
        b = baseline["summary"]
        lines.append(
            f"| Pretrained features (768D) | 1 extractor | {100 * b['rank1']:.2f} | {100 * b['rank5']:.2f} | {100 * b['map']:.2f} |"
        )
    lines += [
        "",
        "Rank-k ranks identities by maximum reference-image similarity. AP ranks gallery images; mAP averages queries within each identity, then averages identities.",
        "± denotes sample SD across training seeds, not a confidence interval over animals or datasets. Three seeds do not establish statistical significance.",
        "These are results under one fixed custom identity split, not official competition test scores or evidence of unknown-identity rejection.",
        "Native training loss magnitudes differ across objectives. Timing is observed, includes variable early-stopping budgets, and may be affected by system load.",
        "Pretrained features have 768 dimensions; trained embeddings have 128. This is a representation baseline, not a matched-dimension ablation.",
        "Gallery/query capture-session independence remains as documented in protocol_notes.json. dHash screening alone cannot verify independence.",
        "Do not change hyperparameters or gallery/query choices using these final test results. Report negative/equal outcomes honestly.",
    ]
    (out / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    if args.manifest:
        make_error_sheet(results, args.manifest, args.prepared, out)
    print("Created tables and six comparison figures in", out)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--evaluation", default="evaluation")
    p.add_argument("--output", default="comparison")
    p.add_argument(
        "--manifest",
        help="Optional: also generate failure-example contact sheet from local images.",
    )
    p.add_argument(
        "--prepared", help="Override relocated prepared image root for contact sheet."
    )
    return p
