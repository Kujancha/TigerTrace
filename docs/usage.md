# Advanced usage

All commands use `python run.py COMMAND`. Global `--data-dir` and `--work-dir` options come **before** the command. Command-specific options come after it. Run `python run.py COMMAND --help` to inspect them.

## Paths

Raw inputs default to `data/`; generated work defaults to `artifacts/`, relative to this repository's root when using `run.py`. The wrapper also works when called by absolute path from another working directory. Optional installed console/module entry points use the current working directory as project root.

```powershell
python run.py --data-dir "D:\datasets\atrw" --work-dir "D:\work\atrw" prepare
python run.py --work-dir "D:\work\atrw" split
python run.py --work-dir "D:\work\atrw" train
```

Preparation defaults to audit-and-copy. `prepare --audit-only` checks without copying; `--train-only` and `--test-only` choose one raw side. Existing prepared copies and splits are not overwritten. After a successful preparation/split, continue with later stages rather than repeating them.

## Training options

```powershell
python run.py pilot --checkpointing
python run.py train --checkpointing
python run.py train --checkpointing --resume
```

Keep memory settings and budgets the same across methods. If a failed pilot already created outputs, choose a new work directory (and prepare/split there), or explicitly use a new pilot output:

```powershell
python run.py pilot --output artifacts/pilot_checkpointed --checkpointing
```

Override flags are supported:

```powershell
python run.py train --losses triplet_hard --seeds 42 43 44 --epochs 30
python run.py train --losses multi_similarity proxy_anchor --seeds 42 43 44 --epochs 30
```

Separate invocations can populate a consistent `artifacts/runs/` grid, but all scientific settings/source/environment must match. The old random Triplet baseline is intentionally not part of this comparison. CLI flags expose margins, Multi-Similarity parameters, Proxy Anchor parameters, learning rate, maximum epochs, patience, P/K, checkpointing and attention settings. Perform any tuning on validation only, with comparable per-method search budgets.

The run's source fingerprints guard against silently changing implementation midway through training. An interrupted epoch is replayed on resume from the last complete epoch. Do not change max epochs during resume: it defines the learning-rate schedule. If interruption occurs before the first saved epoch checkpoint, use a fresh run output or move that incomplete run aside. Load only trusted checkpoint files.

## Evaluation and plots

```powershell
python run.py evaluate --include-pretrained --batch-size 4
python run.py evaluate --include-pretrained --batch-size 4 --resume
python run.py compare
python run.py compare --examples
```

Defaults evaluate all three methods and all three seeds. Different grids can be supplied explicitly with `--methods`/`--seeds`; do not choose a seed subset based on test scores. Existing incompatible evaluation directories are not overwritten. `compare` can regenerate figures from saved results; no training or inference is needed. The contact sheet is illustrative and uses the first numerical seed per method, not the best test seed.

If running on CPU only for software debugging, append `--device cpu` to baseline, pilot, train or evaluate. Real ViT training on CPU can be slow. The synthetic tests avoid downloading weights and use tiny models for fast workflow checks.

## Optional editable installation

The default workflow does not require package installation. To expose the `tiger-reid` console command in an already configured environment:

```powershell
python -m pip install -e . --no-deps
```

Then `tiger-reid --help` or `python -m tiger_reid --help` are available. Run these from the repository root. The plain `python run.py ...` interface remains the recommended Windows workflow.
