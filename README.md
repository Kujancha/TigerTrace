# Tiger Re-identification: ViT Loss Comparison

A reproducible comparison of **batch-hard Triplet**, **Multi-Similarity**, and **Proxy Anchor** for tiger re-identification. A pretrained ViT-Base is fine-tuned to produce compact 128D identity embeddings, with a fixed dataset protocol and three training seeds per method.

One research entry point, from raw annotations to comparison figures (optional Gradio demo below):

```powershell
python run.py --help
```

## 1. Download the dataset first

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

## 2. Set up Python and GPU dependencies

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

## 3. Prepare and audit the raw files

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

## 4. Create the fixed experiment split

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

## 5. Check the GPU with a pilot

```powershell
python run.py pilot
```

The pilot runs all three losses, seed 42, two epochs and five batches per epoch. It is a memory/software smoke test, not a full experiment. Training and validation have progress bars; training bars show the epoch, batch progress, running loss, learning rate and time estimates.

Default batch size is 4 identities x 2 distinct images = 8, at 224x224 resolution. If you encounter an out-of-memory error, close other GPU applications and retry in a fresh work directory with `--checkpointing` or a smaller `--p 2 --k 2`. Use the same chosen settings for every method in the final comparison. See [docs/usage.md](docs/usage.md).

## 6. Run the baseline and full training

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

## 7. Evaluate and create the comparison

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

## 8. Optional Gradio identification demo

Already trained? **Do not rerun training.** Use an existing `best.pt` from this project's runs. The demo adds a local, persistent gallery without changing any training or evaluation outputs. It expects cropped tiger photographs; it is not an animal detector.

Install the one optional UI dependency in the same Python environment (tested with Gradio 6.31.0):

```powershell
python -m pip install "gradio>=6,<7"
```

Launch from the project folder (replace the checkpoint path with your chosen run):

```powershell
python demo.py --checkpoint artifacts/runs/proxy_anchor_seed42/best.pt
```

Open **http://127.0.0.1:7860** if the browser does not open automatically. CUDA is used when available, with CPU fallback. Override with `--device cpu` or change a busy port with `--port 7861`. Paths are relative to the project folder, or you can pass absolute paths. The app binds only to the local computer and does not create a public sharing link. Only load your own trusted study checkpoints; no `.pkl` conversion is needed.

### Build your known-tiger database

1. Open **Manage gallery** and choose **train**.
2. Click **Import reference images**. The progress bar tracks embedding extraction. Identity labels and paths come from `artifacts/experiment/manifest.json`, with image files under `artifacts/prepared/train`. The manifest must match the checkpoint's original manifest byte for byte.
3. Open **Identify & register**, upload a different tiger photograph, and click **Find top 3 tigers**.
4. The app shows up to three **distinct identities**, each identity's best matching reference image, and its cosine similarity. With fewer than three registered identities, fewer results are shown.
5. If the best cosine is below the threshold, the app reports **No confident match — possibly an unregistered tiger**. This is not proof that the tiger is new.
6. To enroll an animal, enter a new ID, review the candidates, tick the confirmation box, and click **Save reference to gallery**. To add another photograph of a known tiger, select **Existing tiger** and enter that exact ID. Enrollment is allowed after review whether or not the threshold accepted the match.

The database is `artifacts/demo/gallery.sqlite3`. It stores IDs, reference images, normalized 128D embeddings, and a checkpoint fingerprint. It persists across restarts. No training is performed during enrollment. Exact decoded-image duplicates are skipped; conflicting labels are rejected. The gallery tab can remove an incorrectly registered identity and all its references, after confirmation. Back up the database while the app is closed. Do not run multiple app processes against the same database.

All generated demo data stays inside the already-ignored `artifacts/` directory. A different model needs a new database because embeddings from different checkpoints must not be mixed:

```powershell
python demo.py --checkpoint artifacts/runs/multi_similarity_seed42/best.pt --database artifacts/demo/multi_similarity.sqlite3
```

To use existing artifacts elsewhere:

```powershell
python demo.py --checkpoint "D:\old-project\runs\proxy_anchor_seed42\best.pt" --manifest "D:\old-project\experiment\manifest.json" --prepared-root "D:\old-project\prepared\train" --database artifacts/demo/gallery.sqlite3
```

Use the directory that actually contains the manifest's relative image paths for `--prepared-root`. See [migration instructions](docs/migration.md) if you are moving the previous study outputs.

### A stronger presentation: known and unknown tigers

Use a fresh database, e.g. `--database artifacts/demo/heldout.sqlite3`. In **Manage gallery**, choose **test-gallery**. This uses only the gallery-role images from the custom held-out split within the labeled training pool, not the official unlabeled test archive. Query-role images are never imported by this option.

Enter one or more exact manifest IDs in **IDs to withhold**, separated by commas. Choose IDs from `identity_splits.test` in your manifest. The importer refuses to withhold an identity already present in the database; remove it or use a fresh database first. Imports are additive, so switching source does not clear existing identities.

Demonstrate a separate query photograph from a registered tiger, then a query of a withheld tiger. Review and register the withheld animal, then query it using a **different photograph**. An exact gallery image is explicitly flagged as a self-match. False matches are possible and should be reported honestly.

### Similarity threshold and scope

The identity score is the **maximum cosine similarity over that identity's reference images**. The best identity becomes a candidate when its score is **greater than or equal to** the threshold. Scores are not probabilities or confidence percentages.

The initial threshold **0.6 is illustrative and uncalibrated**. For a defensible unknown-detection claim, choose the threshold using validation identities divided into registered and withheld groups. Measure false acceptance of withheld animals and false rejection of registered animals. Keep final-test data out of threshold selection. Changing the number of references or the model may require recalibration. The app provides manual threshold exploration, not an automated calibration experiment.

This mutable gallery demo is separate from the study's fixed retrieval protocol. Its threshold-based decisions are not new research results and do not change the comparison tables. The UI uses the same `TigerViT`, 128D head, EXIF handling, 224×224 bicubic resize, and normalization as the existing evaluator. No pretrained-backbone baseline is mixed into this gallery.

### Verify the addition

```powershell
python run.py test --pattern test_demo.py
python demo.py --help
```

Tests use synthetic features and images, including checkpoint-format loading, matching, persistence, duplicate rejection, import rollback, and held-out gallery selection. The Gradio callback test runs when the optional dependency is installed. Full GPU inference with your trained weights must be run on your machine.

## Repository structure

| Path | Responsibility |
|---|---|
| `run.py` | Research pipeline entry point |
| `demo.py` | Optional local Gradio gallery app |
| `src/tiger_reid/` | Preparation, splits, datasets, models, losses, training, evaluation and plots |
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

## Tests and existing results

```powershell
python run.py test
```

Tests cover raw-data audits, fixed splits, sampling, metrics, training/proxy gradients, progress bars, resume, final evaluation, plotting, and the unified CLI. They use synthetic fixtures and tiny models to keep them fast; they are not tiger performance results.

**Already completed the earlier fresh pipeline? You do not need to retrain just to organize the repository.** Read [docs/migration.md](docs/migration.md) before moving any files. Prior completed state-dict checkpoints and evaluation outputs can be reused for inference/plots. Strict training resume across the source refactor is intentionally not promised.

A snapshot of the completed study is documented separately in [docs/results.md](docs/results.md), including the stronger 768D pretrained reference. It is not a promise that another machine, split, or tuning configuration reproduces identical scores. New runs generate their own results under `artifacts/`.

## Put the project on GitHub

The ZIP contains one repository folder. Open a terminal inside it, then:

```powershell
git init
git add .
git status --short
```

Review the staged files before committing and adding your own remote. The checked ignore rules keep raw images/annotations, model checkpoints, generated outputs and virtual environments out of the repository. A custom output directory outside `artifacts/` must also be ignored if it is placed inside the repository.

## References

- [ATRW dataset, downloads, license and citation](https://lila.science/datasets/atrw)
- [Multi-Similarity Loss, CVPR 2019](https://openaccess.thecvf.com/content_CVPR_2019/html/Wang_Multi-Similarity_Loss_With_General_Pair_Weighting_for_Deep_Metric_Learning_CVPR_2019_paper.html)
- [Proxy Anchor Loss, CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/html/Kim_Proxy_Anchor_Loss_for_Deep_Metric_Learning_CVPR_2020_paper.html)
- [PyTorch Metric Learning documentation](https://kevinmusgrave.github.io/pytorch-metric-learning/)

This repository implements a controlled empirical comparison of established configurations; it does not propose a new loss or claim universally superior results.
