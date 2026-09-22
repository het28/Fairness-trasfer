"""User-group utility fairness from EvalInputs (macro means per group)."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, Sequence

from cikm_eval.types import EvalInputs, GroupName, UserGroupUtilityResult, UserId


def ndcg_at_k(recommended: Sequence[int], relevant: set[int], k: int) -> float:
    recs = list(recommended[:k])
    if not relevant:
        return 0.0

    dcg = 0.0
    for rank, item in enumerate(recs, start=1):
        if item in relevant:
            dcg += 1.0 / math.log2(rank + 1)

    ideal_hits = min(len(relevant), k)
    if ideal_hits == 0:
        return 0.0

    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(recommended: Sequence[int], relevant: set[int], k: int) -> float:
    if not relevant:
        return 0.0
    recs = set(recommended[:k])
    return len(recs & relevant) / len(relevant)


def compute_per_user_utility(inputs: EvalInputs) -> dict[UserId, dict[str, float]]:
    """
    Per evaluated user: ndcg, recall, num_test_items, user_group.
    Users are those present in ``inputs.recommendations``.
    """
    out: dict[UserId, dict[str, float]] = {}
    k = inputs.k
    for u, recs in inputs.recommendations.items():
        rel = set(inputs.test_items.get(u, ()))
        g = inputs.user_group.get(u, "")
        out[u] = {
            "ndcg": ndcg_at_k(recs, rel, k),
            "recall": recall_at_k(recs, rel, k),
            "num_test_items": int(len(rel)),
            "user_group": str(g),
        }
    return out


def aggregate_user_group_utility(
    per_user: dict[UserId, dict[str, float]],
) -> UserGroupUtilityResult:
    """Macro: mean within each user group, then gaps / std across group means."""
    sum_ndcg: dict[GroupName, float] = defaultdict(float)
    sum_rec: dict[GroupName, float] = defaultdict(float)
    cnt: dict[GroupName, int] = defaultdict(int)

    for row in per_user.values():
        gname = row["user_group"]
        if not gname:
            continue
        sum_ndcg[gname] += float(row["ndcg"])
        sum_rec[gname] += float(row["recall"])
        cnt[gname] += 1

    group_ndcg: dict[GroupName, float] = {}
    group_recall: dict[GroupName, float] = {}
    group_size: dict[GroupName, int] = {}
    for gname, n in cnt.items():
        if n <= 0:
            continue
        group_size[gname] = n
        group_ndcg[gname] = sum_ndcg[gname] / n
        group_recall[gname] = sum_rec[gname] / n

    ndcg_vals = list(group_ndcg.values())
    rec_vals = list(group_recall.values())

    if not ndcg_vals:
        return UserGroupUtilityResult(
            group_ndcg={},
            group_recall={},
            group_size={},
            ndcg_gap_max_min=0.0,
            recall_gap_max_min=0.0,
            ndcg_std=0.0,
            recall_std=0.0,
            worst_group_ndcg=0.0,
            worst_group_recall=0.0,
            best_group_ndcg=0.0,
            best_group_recall=0.0,
        )

    ndcg_gap = max(ndcg_vals) - min(ndcg_vals)
    recall_gap = max(rec_vals) - min(rec_vals) if rec_vals else 0.0
    ndcg_std = _pop_std(ndcg_vals)
    recall_std = _pop_std(rec_vals)

    worst_g_ndcg = min(group_ndcg, key=group_ndcg.get)
    worst_g_rec = min(group_recall, key=group_recall.get)
    best_g_ndcg = max(group_ndcg, key=group_ndcg.get)
    best_g_rec = max(group_recall, key=group_recall.get)

    return UserGroupUtilityResult(
        group_ndcg=dict(group_ndcg),
        group_recall=dict(group_recall),
        group_size=dict(group_size),
        ndcg_gap_max_min=float(ndcg_gap),
        recall_gap_max_min=float(recall_gap),
        ndcg_std=float(ndcg_std),
        recall_std=float(recall_std),
        worst_group_ndcg=float(group_ndcg[worst_g_ndcg]),
        worst_group_recall=float(group_recall[worst_g_rec]),
        best_group_ndcg=float(group_ndcg[best_g_ndcg]),
        best_group_recall=float(group_recall[best_g_rec]),
    )


def _pop_std(vals: list[float]) -> float:
    n = len(vals)
    if n <= 1:
        return 0.0
    m = sum(vals) / n
    var = sum((x - m) ** 2 for x in vals) / n
    return math.sqrt(var)


def compute_user_group_fairness(inputs: EvalInputs) -> UserGroupUtilityResult:
    per_user = compute_per_user_utility(inputs)
    return aggregate_user_group_utility(per_user)
