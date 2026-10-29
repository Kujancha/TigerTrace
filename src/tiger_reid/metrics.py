"""Image-level AP and identity-level Rank-k, averaged equally over identities."""

from collections import defaultdict
import numpy as np


def retrieval(z, rows):
    z = np.asarray(z, dtype=np.float32)
    if z.ndim != 2 or len(z) != len(rows) or not np.isfinite(z).all():
        raise ValueError("Invalid embeddings.")
    norms = np.linalg.norm(z, axis=1, keepdims=True)
    if np.any(norms < 1e-8):
        raise ValueError("Zero embedding.")
    z = z / norms
    gi = [i for i, r in enumerate(rows) if r["role"] == "gallery"]
    qi = [i for i, r in enumerate(rows) if r["role"] == "query"]
    if not gi or not qi:
        raise ValueError("Need gallery and query.")
    gids = np.array([rows[i]["identity"] for i in gi])
    ids = sorted(set(gids))
    per_query = []
    by_id = defaultdict(list)
    for q in qi:
        target = rows[q]["identity"]
        scores = np.clip(z[gi] @ z[q], -1, 1)
        if target not in ids:
            raise ValueError("Query identity missing from gallery.")
        order = np.argsort(-scores, kind="stable")
        relevant = gids[order] == target
        ap = float(
            ((np.cumsum(relevant) / np.arange(1, len(order) + 1)) * relevant).sum()
            / relevant.sum()
        )
        identity_scores = {i: float(scores[gids == i].max()) for i in ids}
        ranking = sorted(ids, key=lambda i: (-identity_scores[i], i))
        rank = ranking.index(target) + 1
        row = {
            "filename": rows[q]["filename"],
            "identity": target,
            "predicted": ranking[0],
            "rank": rank,
            "ap": ap,
            "top5": ranking[:5],
            "best_similarity": identity_scores[ranking[0]],
        }
        by_id[target].append(row)
        per_query.append(row)
    per_identity = {
        i: {
            "rank1": float(np.mean([r["rank"] == 1 for r in rs])),
            "rank5": float(np.mean([r["rank"] <= 5 for r in rs])),
            "map": float(np.mean([r["ap"] for r in rs])),
        }
        for i, rs in by_id.items()
    }
    summary = {
        k: float(np.mean([r[k] for r in per_identity.values()]))
        for k in ["rank1", "rank5", "map"]
    }
    summary.update(
        queries=len(qi),
        identities=len(ids),
        gallery_images=len(gi),
        micro_rank1=float(np.mean([r["rank"] == 1 for r in per_query])),
        micro_map=float(np.mean([r["ap"] for r in per_query])),
    )
    return {"summary": summary, "per_identity": per_identity, "queries": per_query}
