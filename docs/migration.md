# Reuse completed experiments without retraining

This repository packages the **fresh annotation-driven** pipeline, not the first experiment that used independently renumbered reference folders. Keep a backup of your old project until the migrated outputs have been checked.

## Existing evaluation outputs: regenerate figures immediately

Copy your earlier fresh pipeline's `evaluation/` directory to `artifacts/evaluation/`. Preserve its full contents: `status.json`, `evaluation_config.json`, per-run `evaluation.json` and `history.json`, and the baseline directory if included. Then:

```powershell
python run.py compare
```

This reads saved metrics/embeddings metadata and produces `artifacts/comparison/`. It does not train or load model weights. Without `--examples`, raw images and the experiment manifest are not required for these plots. The documented earlier fresh evaluation JSON schema remains supported.

## Existing trained checkpoints: run inference with them

Copy these directories as complete units:

| Old location | New location |
|---|---|
| `prepared/train/` | `artifacts/prepared/train/` |
| `experiment/` | `artifacts/experiment/` |
| `runs/` | `artifacts/runs/` |
| `evaluation/` (if already evaluated) | `artifacts/evaluation/` |

If your earlier labeled preparation used `prepared/` directly, copy that **labeled preparation root** (audit, inventory, annotations, labeled) to `artifacts/prepared/train/`. Do not add an extra `train` folder level or use the unlabeled test preparation as the labeled root.

Preserve `experiment/manifest.json` **byte for byte**. It fingerprints the original assignments and must match the checkpoint. Its absolute prepared-root path may point to the old machine; the unified CLI automatically supplies the new `artifacts/prepared/train/` root as an override, so no manifest editing is necessary.

Each run needs at least `best.pt`, `config.json`, `status.json` marked complete, and `history.json` for final evaluation. Copy `last.pt` too if keeping the original training recovery archive. The network state-dict keys and model computation have been preserved; evaluation accepts the earlier complete fresh-pipeline checkpoint format.

If evaluation has not been done:

```powershell
python run.py evaluate --include-pretrained
python run.py compare --examples
```

If evaluation is already complete, simply compare it. There is no reason to repeatedly inspect test metrics or rerun model inference to reorganize folders. For optional failure sheets, copy the manifest and prepared labeled images too.

## Training resume is different from inference

The refactor changes module paths and source fingerprints. Do not resume an interrupted old training run with the new package. Finish it with the original scripts/environment, then migrate the complete outputs, or start a separately versioned experiment. Inference from completed state-dict checkpoints does not require optimizer resume.

A new training experiment should use a new work directory. Do not mix old/random-Triplet and new/batch-hard-Triplet runs or outputs from different manifests. The comparison script expects consistent settings and seeds.
