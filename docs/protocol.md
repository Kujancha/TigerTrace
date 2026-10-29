# Experimental protocol

## Data and evaluation scope

The raw input is `data/train` and `data/test`, with annotations directly beside the corresponding images. Training identities are taken from the official `[id, filename]` CSV, not inferred from folder order. The public test CSV contains only filenames. Its images can be audited/copied, but no scored identity evaluation is claimed for that unlabeled competition set.

The custom scored split defaults to 60/14/33 disjoint **annotation identity IDs**, using split seed 42. Each validation/test identity supplies two gallery images and the rest as queries. All training seeds share these assignments. Official annotation identity separation does not independently establish biological equivalence or encounter/camera independence beyond those labels. Exact decoded-image duplicates are checked; near-duplicate screening is only a manual-review aid.

Do not claim capture-session-disjoint performance unless that condition is independently established. The manifest records `capture_session_separation_verified: false` and pending review by default. Inspect flagged pairs and gallery/query images before training. If encounter metadata or manual grouping requires a new split, preserve the old protocol and make a new version; do not hand-edit a manifest after experiments or reseed based on accuracy.

## Model and training

- Backbone: `vit_base_patch16_224.augreg_in21k_ft_in1k` from timm.
- Last four transformer blocks and final normalization unfrozen; all earlier backbone parameters frozen.
- Head: Linear(768,256), ReLU, Linear(256,128), L2 normalization.
- Input: bicubic resize224, checkpoint mean/std 0.5; modest train-only color jitter, no horizontal flip.
- P x K sampling: 4 identities x 2 distinct images by default; steps=ceil(train images / batch size). Balanced random sampling does not guarantee every image is visited each epoch.
- Adam lr=1e-4 for all trainable parameters, cosine schedule, maximum 30 epochs, patience 8 on validation macro mAP.
- Mixed precision on CUDA; loss computations in float32; gradient clipping norm 5.
- Seeds 42,43,44. Independent pretrained starts and matching initial head weights for each seed; no loss starts from another loss's trained model.
- Unfused attention by default; CuBLAS workspace set before CUDA initialization; deterministic algorithms requested in warning mode. Bitwise cross-device reproducibility is not guaranteed.

| Configuration | Loss and mining |
|---|---|
| Batch-hard Triplet | Normalized Euclidean distance; margin .3; hardest positive/negative per anchor; mean including zero losses |
| Multi-Similarity | alpha=2, beta=50, base=.5; MultiSimilarityMiner epsilon=.1 |
| Proxy Anchor | alpha=32, margin=.1; trainable identity proxies included in optimizer |

These defaults are a shared experimental starting point, not equally optimized methods. A bounded, comparable validation-only tuning budget can be used before test evaluation. Report the actual configs. A different Triplet sampling/mining strategy changes the baseline and must be named explicitly.

## Metrics

Rank-k ranks unique identities using maximum cosine similarity across their reference images. AP ranks gallery images. Macro metrics average queries within each identity, then identities equally; micro metrics average all queries directly. Two correct gallery images per identity make AP distinct from reciprocal rank.

Report mean and sample SD across training seeds. SD is not a population confidence interval and three seeds alone do not establish statistical significance. Similarity pair distributions are descriptive, contain dependent pairs and are not calibrated confidence probabilities. No unknown-rejection evaluation is implemented.

Raw training loss magnitudes cannot be compared across objectives. Total recorded training time excludes validation and checkpoint writes and varies with early stopping/system load; inspect seconds/batch as well. Peak allocated and reserved CUDA memory differ and do not measure other applications' GPU use.

The unfine-tuned reference uses normalized raw 768D backbone features. Learned embeddings use 128D, so differences combine fine-tuning and projection changes. Its results must be labeled separately. Fewer dimensions reduce feature storage at fixed dtype, not backbone parameter count or inference time by the same factor.
