# Packaging verification

The unified repository passed **34 software tests** on CPU. They include a complete raw-data-to-report CLI workflow with annotations beside images, fixed splits, balanced sampling, all three objectives and proxy updates, checkpoint resume, test evaluation, progress bars, plots and input rejection. Synthetic test images and tiny models are used for end-to-end tests; they are not research results.

Additional checks performed before packaging:

- Python source formatted consistently; undefined/unused-import checks passed.
- Root help works when the entry point is invoked from another working directory.
- Git ignore rules tested in a temporary repository: raw images/annotations, generated artifacts, checkpoints and environments ignored; source, docs, tests and dataset placeholders retained.
- The previous fresh-pipeline ViT class and the packaged class have identical state-dict keys. Loading the previous class's real ViT weights into the packaged class produced identical CPU forward outputs for the checked inputs. This establishes state-dict compatibility, not guaranteed bitwise equivalence across environments or compatibility of interrupted optimizer state/source fingerprints.
- The archive is checked for corruption and excludes caches, datasets, model files, generated experiment outputs and temporary files.

Verification environment: Python 3.12.14, torch 2.15.0+cpu, torchvision 0.29.2+cpu, timm 1.0.30, pytorch-metric-learning 2.9.0. NVIDIA execution and fresh real-data training were not repeated for this packaging task. Your completed training data/model artifacts are not included in the ZIP.

Source-checkout test command: `python run.py test`.
