"""Per-user NDCG@K and Recall@K from full-score vectors (binary relevance)."""

from __future__ import annotations

import numpy as np


def ndcg_recall_at_k(
    scores: np.ndarray,
    positive_items: np.ndarray,
    k: int,
) -> tuple[float, float]:
    """
    ``scores`` length ``n_items`` (higher = better). ``positive_items`` internal item ids.
    Returns (NDCG@k, Recall@k). Users with no positives get (0.0, 0.0).
    """
    pos = np.unique(positive_items.astype(np.int64, copy=False))
    n_pos = int(pos.size)
    if n_pos == 0:
        return 0.0, 0.0

    order = np.argsort(-scores, kind="mergesort")
    top = order[: min(k, order.shape[0])]
    pos_set = set(pos.tolist())

    dcg = 0.0
    for rank, item in enumerate(top, start=1):
        if item in pos_set:
            dcg += 1.0 / np.log2(rank + 1)

    ideal_n = min(n_pos, k)
    idcg = sum(1.0 / np.log2(i + 1) for i in range(1, ideal_n + 1))
    ndcg = float(dcg / idcg) if idcg > 0 else 0.0

    hits = sum(1 for item in top if item in pos_set)
    recall = float(hits / n_pos)
    return ndcg, recall


def tail_fraction_in_items(item_ids: np.ndarray, item_lab: np.ndarray, tail_group_id: int) -> float:
    """Fraction of ``item_ids`` whose label equals ``tail_group_id``."""
    if item_ids.size == 0:
        return 0.0
    lab = item_lab[item_ids.astype(np.int64, copy=False)]
    return float(np.mean(lab == tail_group_id))
