# Tiger Re-Identification with Vision Transformers

**A comparative study of metric-learning loss functions for individual tiger identification.**

This project evaluates **Batch-Hard Triplet Loss**, **Multi-Similarity Loss**, and **Proxy Anchor Loss** using a fine-tuned Vision Transformer (ViT-Base). Each model learns 128-dimensional embeddings for image retrieval: photographs of the same tiger should have similar embeddings, while different individuals should be distinguishable.

The experiments share a fixed identity-disjoint dataset split, evaluation protocol, and training configuration. Each loss is evaluated across three random seeds. The repository also includes a local Gradio application for identifying and registering tigers against a persistent reference gallery.

**Research workflow:** prepare and audit ATRW data → create a fixed split → train models → evaluate retrieval → compare results. The Gradio application is optional and uses an existing trained checkpoint.

```powershell
python run.py --help
```

## Contents

- [Dataset](#dataset)
- [Installation](#installation)
- [Data preparation and splits](#data-preparation-and-splits)
- [Training](#training)
- [Evaluation and results](#evaluation-and-results)
- [Gradio demo](#gradio-demo)
- [Repository layout](#repository-layout)
- [Testing and reproducibility](#testing-and-reproducibility)
- [References](#references)

## Dataset

Download **all four Re-identification archives** from the [ATRW dataset page on LILA BC](https://lila.science/datasets/atrw):

1. Re-ID training images.
2. Re-ID training annotations.
3. Re-ID test images.
4. Re-ID test annotations.

Extract the images and **put each CSV and JSON annotation file in the same folder as its corresponding images**. Do not leave the annotations in a separate archive folder. Do not rename images or assign new tiger IDs.

Your raw input must look like this:

```text
data/
  train/
    003597.jpg
    ...other training images...
    reid_list_train.csv
    reid_keypoints_train.json
  test/
    000999.jpg
    ...other public test images...
    reid_list_test.csv
    reid_keypoints_test.json
```

Images can also be in nested subfolders under `train/` or `test/`; their basenames must be unambiguous. The four annotation files themselves must be directly inside the corresponding `train/` or `test/` directory. Keep the original numeric filenames, including leading zeros.

**The public test CSV lists filenames, not identity ground truth.** We inventory it without inventing labels. Scored training/validation/test experiments use a fixed custom identity split of the officially labeled training pool. The inspected training CSV contains 1,887 image labels for 107 identity IDs; additional downloaded images without identity labels are reported and left untouched.

No dataset, annotation archive, pretrained weights, or user checkpoints are distributed with this repository. Follow the dataset publisher's [CC BY-NC-SA 4.0 terms](https://lila.science/datasets/atrw) and citation instructions.

## Installation

If your existing environment already trains successfully on your NVIDIA GPU, keep using it. For a new environment, Python 3.11 or 3.12 is recommended (the code supports Python >=3.10 with compatible PyTorch wheels).

From the repository root on Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

**Install a matching CUDA-enabled `torch` and `torchvision` pair first** using the [official PyTorch installation selector](https://pytorch.org/get-started/locally/) for your OS and driver. PyTorch >=2.3 is required. Do not install a CPU-only build for GPU training, and do not replace an existing working CUDA installation unnecessarily.

Then install the other dependencies:

```powershell
python -m pip install -r requirements.txt
python run.py doctor
```

Confirm `CUDA available: True`. If PowerShell activation is unavailable, use `.\.venv\Scripts\python.exe` in place of `python` for each command. Linux/macOS users can activate the environment with `source .venv/bin/activate`; the Python workflow is the same.

The named timm pretrained checkpoint downloads on first model use if not cached. It requires internet access. Training and final inference run locally.

## Data preparation and splits

### Prepare and audit the dataset

```powershell
python run.py prepare
```

This automatically finds all four annotation files beside their images, audits both datasets, and copies the verified images into `artifacts/prepared/`. **No `--copy` flag is needed in this unified interface.** Raw `data/` files remain unchanged.

For an audit without copying:

```powershell
python run.py prepare --audit-only
```

Read the generated `audit.json` files. Missing/corrupt images, ambiguous names, or conflicting identity labels block preparation. Exact duplicates within an identity are recorded and excluded from the labeled prepared copy. Unlabeled and unlisted files are not assigned identities. Test exact duplicates are reported and retained.

Once preparation is complete, skip this command on later runs. Existing prepared copies are not silently overwritten. Use `--train-only` or `--test-only` to prepare only one side if needed.

### Create the experiment split

```powershell
python run.py split
```

Default protocol:

| Partition | Annotation identity IDs | Use |
|---|---:|---|
| Train | 60 | Weight updates |
| Validation | 14 | Hyperparameter decisions and checkpoint selection |
| Held-out test | 33 | Final scored comparison |

The split seed is 42. Each validation/test identity has **two gallery images**, with the rest used as queries. These assignments are saved in `artifacts/experiment/manifest.json` and `assignments.csv`; **images are not physically moved into train/val/test folders**.

Review `gallery_query_review.csv` and `near_duplicate_candidates.csv` before training. Low dHash distance is a review cue, not proof of a duplicate. Capture-session separation is not established by these annotation files or similarity checks. The manifest records this limitation rather than claiming encounter-independent evaluation.

Do not rerun splitting to get more favorable results. The same assignments are used for every loss and training seed. Details are in [docs/protocol.md](docs/protocol.md).

## Training

### Pilot run

```powershell
python run.py pilot
```

The pilot runs all three losses, seed 42, two epochs and five batches per epoch. It is a memory/software smoke test, not a full experiment. Training and validation have progress bars; training bars show the epoch, batch progress, running loss, learning rate and time estimates.

Default batch size is 4 identities x 2 distinct images = 8, at 224x224 resolution. If you encounter an out-of-memory error, close other GPU applications and retry in a fresh work directory with `--checkpointing` or a smaller `--p 2 --k 2`. Use the same chosen settings for every method in the final comparison. See [docs/usage.md](docs/usage.md).

### Baseline and full experiments

```powershell
python run.py baseline
python run.py train
```

The baseline evaluates raw normalized **768D pretrained features** on validation images. It is a representation reference, not a matched-dimension ablation. Each training run also records its initialized 128D head's validation performance.

Full training defaults to **three methods x seeds 42, 43, 44 = nine sequential runs**, with up to 30 epochs and early stopping patience 8. Only the last four ViT blocks, final backbone normalization, and embedding head are trainable. Model input normalization follows the named checkpoint: mean/std=(0.5,0.5,0.5).

Each run saves `best.pt`, a resumable `last.pt`, config/source fingerprints, environment versions, per-query validation results, epoch CSV/JSON logs, training/validation durations, and GPU-memory measurements. Best checkpoints are selected by **validation macro mAP**, never test scores. The script does not read final-test image bytes during training.

To resume after an interruption:

```powershell
python run.py train --resume
```

Repeat any options from the original command. Completed runs are skipped; an interrupted run resumes after the last fully checkpointed epoch. Source, settings, manifest and environment must match. A partial epoch is rerun. Runs interrupted before their first checkpoint need a new output directory. Allow roughly 10–15 GB for nine runs' model/optimizer checkpoints; actual usage varies.

## Evaluation and results

After fixing training decisions and completing all runs:

```powershell
python run.py evaluate --include-pretrained
python run.py compare --examples
```

Evaluation loads the validation-selected model for each run and scores the fixed held-out gallery/query assignments. The optional pretrained-feature reference is evaluated separately. Outputs include embeddings, ranked matches, per-identity results and similarity distributions. To continue an interrupted evaluation, repeat its command with `--resume`.

The comparison is saved in **`artifacts/comparison/`**:

| Output | Contents |
|---|---|
| `summary.csv` | Means and sample standard deviations across training seeds |
| `runs.csv` | Individual run scores, timings and memory |
| `paired_vs_triplet.csv` | Per-seed metric differences from Triplet in percentage points |
| `RESULTS.md` | Results table and interpretation notes |
| `retrieval.png`, `cmc.png` | Rank-1/5, mAP and cumulative matching curves |
| `learning_and_cost.png`, `loss_curves.png` | Learning behavior and recorded cost |
| `similarity_distributions.png`, `per_identity.png` | Embedding separation and identity-level performance |
| `failed_queries.json`, `failure_examples.png` | Error-analysis records and optional image sheet |

Use `python run.py compare` without `--examples` if only regenerated tables/plots are needed. No model retraining or repeated inference is needed to regenerate figures from a complete evaluation directory.

## Gradio demo

The optional Gradio app provides an interactive way to query a local reference database, inspect matches, register new images, and run identification on a directory of photographs. It uses the same trained `TigerViT` architecture and preprocessing as the experiment pipeline; **launching the demo does not retrain the model or alter the research results**.

The interface follows the original Tiger Identification demo's minimalist dark design and is organized into four tabs: **Setup**, **Identify**, **Batch**, and **Info**.

### Launch

Install Gradio in the same environment as the project:

```powershell
python -m pip install "gradio>=6,<7"
```

From the repository root, launch the demo with an existing `best.pt` checkpoint:

```powershell
python demo.py --checkpoint artifacts/runs/proxy_anchor_seed42/best.pt
```

Gradio opens locally at [http://127.0.0.1:7860](http://127.0.0.1:7860). Use `--device cpu` to force CPU inference or `--port 7861` if the default port is occupied. The server is bound to `127.0.0.1` and does not create a public sharing link. Only load checkpoints you trust.

### Setup: load or build the reference database

The application stores registered identities, reference images, and normalized embeddings in a SQLite database at:

```text
artifacts/demo/gallery.sqlite3
```

The database persists between launches. **Load Existing Database** refreshes its inventory; it does not import or recompute embeddings. For a new database, choose a **Reference source** and click **Build Database from Images**:

| Reference source | Images imported |
|---|---|
| `test-gallery` (default) | Gallery-role images from the custom held-out split; query-role images are excluded |
| `train` | Labeled training images from the experiment split |

Imports use `artifacts/experiment/manifest.json` and images stored under `artifacts/prepared/train/`. The manifest must match the manifest associated with the checkpoint. This workflow **does not scan arbitrary `data/reference/` identity folders** as the original project did.

You can enter comma-separated exact identities in **IDs to withhold (optional)** before importing. This is useful for a known-versus-unknown demonstration. The import is additive; choosing a different source does not clear existing records. If an identity is already registered, it cannot be withheld retroactively without first deleting it or starting a fresh database.

The **Setup** tab also shows the current identity inventory. To remove a tiger, choose its ID, tick the deletion confirmation, and click **Delete Tiger**. This removes its gallery records, not the source dataset files.

### Identify: match and register a tiger

In the **Identify** tab, upload a cropped tiger photograph and click **Identify**. The application extracts a 128-dimensional embedding, compares it with registered references using cosine similarity, and takes the **highest score per identity**. It displays the three strongest distinct candidates in the result text, plus an expandable gallery of their best reference images.

When the leading score is at least **0.96**, the result is **MATCH FOUND** and the reference photographs of that identity appear below the result. Otherwise the app displays **NEW TIGER DETECTED**. This means only that **no registered identity passed the threshold**; it is not proof that the animal is new.

To register the uploaded photograph, enter a **Tiger name / ID**, review the results, tick the confirmation checkbox, and click **Add Tiger to Database**. An existing ID receives an additional reference image; an unused ID creates a new identity. The application skips exact duplicate images and rejects conflicting labels. Enrollment updates the SQLite gallery only; it does not update model weights.

### Batch: identify a directory of images

The **Batch** tab runs the same embedding and matching procedure on supported image files under a local directory, including subdirectories. The default query path is `data/test`, which you can change in the interface. Click **Run Batch** to see the predicted identity (or **NEW TIGER**) and cosine similarity for each image. Individual files that cannot be processed are reported as errors rather than stopping the whole batch.

### Threshold and evaluation caveats

The demo uses a **fixed cosine-similarity threshold of 0.96**, restored from the original interface. This threshold is *not calibrated for the new model or its loss functions*. The displayed percentages represent cosine similarities, **not confidence probabilities**. Different loss functions may produce different score distributions even when they use the same embedding dimension.

For a meaningful unknown-identity experiment, use a fresh database and import `test-gallery` while withholding one or more held-out IDs. Query a separate image of a registered tiger, then one from a withheld identity. If you register that identity, evaluate it with a **different** photograph. An exact registered-image query is flagged as a self-match and must not be treated as independent evidence of recognition performance.

The gallery is a mutable demonstration database, separate from the fixed train/validation/test experiment. Use the evaluation pipeline—not the demo's threshold decisions—for Rank-1, mAP, CMC, and comparative loss-function results. Calibrating an unknown-detection threshold would require validation identities divided into known and withheld groups, with false-accept and false-reject rates reported.

### Database and checkpoint options

Different checkpoints should use **different SQLite databases**. The database stores a checkpoint fingerprint and preprocessing signature to prevent mixing incompatible embeddings.

```powershell
python demo.py --checkpoint artifacts/runs/multi_similarity_seed42/best.pt --database artifacts/demo/multi_similarity.sqlite3
```

If experiment artifacts live elsewhere, provide their paths explicitly:

```powershell
python demo.py --checkpoint "D:\old-project\runs\proxy_anchor_seed42\best.pt" --manifest "D:\old-project\experiment\manifest.json" --prepared-root "D:\old-project\prepared\train" --database artifacts/demo/gallery.sqlite3
```

The `--prepared-root` path must be the root relative to which manifest image paths are resolved. Back up the database while the application is closed, and avoid running multiple demo processes against the same database. See [migration guidance](docs/migration.md) for moving existing experiment outputs.

To check the demo entry point and its software tests:

```powershell
python demo.py --help
python run.py test --pattern test_demo.py
```

Full model inference with trained weights requires the checkpoint and dataset on your machine.

## Repository layout

| Path | Responsibility |
|---|---|
| `run.py` | Research pipeline entry point |
| `demo.py` | Launcher for the local Gradio identification app |
| `src/tiger_reid/` | Data preparation, splits, models, loss functions, training, evaluation, plots, and Gradio implementation |
| `tests/` | Synthetic software tests; no tiger download required |
| `docs/` | Protocol, advanced usage, migration and reported results |
| `data/train/`, `data/test/` | Your downloaded images **and their annotations** |
| `artifacts/` | All generated data copies, splits, runs, embeddings and reports |

`data/` contents, `artifacts/`, checkpoints, feature arrays and environment files are ignored by Git. Keep the placeholder `.gitkeep` files, source, tests and documentation in your GitHub repository.

Custom dataset/output locations use global options **before** the command:

```powershell
python run.py --data-dir "D:\datasets\atrw" --work-dir "D:\experiments\atrw" prepare
python run.py --data-dir "D:\datasets\atrw" --work-dir "D:\experiments\atrw" split
```

Use that same `--work-dir` for the remaining commands. You can inspect individual command options with `python run.py train --help`, etc.

## Testing and reproducibility

```powershell
python run.py test
```

Tests cover raw-data audits, fixed splits, sampling, metrics, training/proxy gradients, progress bars, resume, final evaluation, plotting, and the unified CLI. They use synthetic fixtures and tiny models to keep them fast; they are not tiger performance results.

**Already completed the earlier fresh pipeline? You do not need to retrain just to organize the repository.** Read [docs/migration.md](docs/migration.md) before moving any files. Prior completed state-dict checkpoints and evaluation outputs can be reused for inference/plots. Strict training resume across the source refactor is intentionally not promised.

A snapshot of the completed study is documented separately in [docs/results.md](docs/results.md), including the stronger 768D pretrained reference. It is not a promise that another machine, split, or tuning configuration reproduces identical scores. New runs generate their own results under `artifacts/`.


## References

- [ATRW dataset, downloads, license and citation](https://lila.science/datasets/atrw)
- [Multi-Similarity Loss, CVPR 2019](https://openaccess.thecvf.com/content_CVPR_2019/html/Wang_Multi-Similarity_Loss_With_General_Pair_Weighting_for_Deep_Metric_Learning_CVPR_2019_paper.html)
- [Proxy Anchor Loss, CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/html/Kim_Proxy_Anchor_Loss_for_Deep_Metric_Learning_CVPR_2020_paper.html)
- [PyTorch Metric Learning documentation](https://kevinmusgrave.github.io/pytorch-metric-learning/)

This repository implements a controlled empirical comparison of established configurations; it does not propose a new loss or claim universally superior results.
