"""Transfer matrix, deltas, tail exposure, exposure–utility alignment."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Literal

import numpy as np
from scipy.stats import pearsonr, spearmanr

from cikm_eval.types import (
    AlignmentResult,
    EvalInputs,
    GroupName,
    TransferDeltaResult,
    TransferMatrixResult,
    ordered_item_groups,
    ordered_user_groups,
)
from cikm_eval.user_fairness import compute_user_group_fairness

NormalizeMode = Literal["row", "global", "count"]


def correlation_tail_ndcg_deltas(xs: list[float], ys: list[float]) -> tuple[float | None, float | None]:
    """Pearson / Spearman between aligned vectors (e.g. Δ tail exposure vs Δ NDCG per user group)."""
    if len(xs) < 2 or len(ys) < 2:
        return None, None
    if float(np.std(xs)) < 1e-12 or float(np.std(ys)) < 1e-12:
        return None, None
    pr = pearsonr(xs, ys)
    sr = spearmanr(xs, ys)
    pr_val = pr.statistic if hasattr(pr, "statistic") else pr[0]
    sr_val = sr.statistic if hasattr(sr, "statistic") else sr[0]
    pearson = float(pr_val) if pr_val is not None and math.isfinite(float(pr_val)) else None
    spearman = (
        float(sr_val)
        if sr_val is not None and math.isfinite(float(sr_val))
        else None
    )
    return pearson, spearman


def compute_transfer_matrix(
    inputs: EvalInputs,
    normalize: NormalizeMode = "row",
) -> TransferMatrixResult:
    """
    Count recommendations from user group h to item group g, then normalize.

    Row mode (default): T[h,g] = count / (|U_h| * k) so each row sums to 1.
    Global: T[h,g] = count / sum_{h',g'} count.
    Count: raw counts (row_sums / col_sums are sums of counts).
    """
    k = inputs.k
    users = list(inputs.recommendations.keys())
    seen_u: set[GroupName] = set()
    seen_i: set[GroupName] = set()
    for u in users:
        seen_u.add(inputs.user_group[u])
        for it in inputs.recommendations[u][:k]:
            seen_i.add(inputs.item_group[it])

    ug = ordered_user_groups(seen_u)
    ig = ordered_item_groups(seen_i)
    ui = {g: idx for idx, g in enumerate(ug)}
    ii = {g: idx for idx, g in enumerate(ig)}

    counts = np.zeros((len(ug), len(ig)), dtype=np.float64)
    group_size = np.zeros(len(ug), dtype=np.float64)

    for u in users:
        h = inputs.user_group[u]
        if h not in ui:
            continue
        hi = ui[h]
        group_size[hi] += 1.0
        for it in inputs.recommendations[u][:k]:
            g = inputs.item_group[it]
            if g not in ii:
                continue
            counts[hi, ii[g]] += 1.0

    if normalize == "count":
        mat = counts.tolist()
        row_sums = {ug[r]: float(counts[r].sum()) for r in range(len(ug))}
        col_sums = {ig[c]: float(counts[:, c].sum()) for c in range(len(ig))}
        return TransferMatrixResult(
            user_groups=list(ug),
            item_groups=list(ig),
            matrix=mat,
            row_sums=row_sums,
            col_sums=col_sums,
            normalize="count",
            counts=mat,
        )

    if normalize == "global":
        total = float(counts.sum()) + 1e-12
        norm = counts / total
    else:
        denom = (group_size * k)[:, np.newaxis] + 1e-12
        norm = counts / denom

    mat = norm.tolist()
    row_sums = {ug[r]: float(norm[r].sum()) for r in range(len(ug))}
    col_sums = {ig[c]: float(norm[:, c].sum()) for c in range(len(ig))}

    return TransferMatrixResult(
        user_groups=list(ug),
        item_groups=list(ig),
        matrix=mat,
        row_sums=row_sums,
        col_sums=col_sums,
        normalize=normalize,
        counts=counts.tolist(),
    )


def compute_transfer_delta(
    baseline: TransferMatrixResult,
    candidate: TransferMatrixResult,
) -> TransferDeltaResult:
    # Robust alignment: compare in a shared canonical label space instead of
    # requiring identical row/column order in serialized matrices.
    ug = ordered_user_groups(set(baseline.user_groups) | set(candidate.user_groups))
    ig = ordered_item_groups(set(baseline.item_groups) | set(candidate.item_groups))

    b_u = {g: i for i, g in enumerate(baseline.user_groups)}
    b_i = {g: j for j, g in enumerate(baseline.item_groups)}
    c_u = {g: i for i, g in enumerate(candidate.user_groups)}
    c_i = {g: j for j, g in enumerate(candidate.item_groups)}

    a_raw = np.asarray(baseline.matrix, dtype=np.float64)
    b_raw = np.asarray(candidate.matrix, dtype=np.float64)
    a = np.zeros((len(ug), len(ig)), dtype=np.float64)
    b = np.zeros((len(ug), len(ig)), dtype=np.float64)

    for r, uh in enumerate(ug):
        bi = b_u.get(uh)
        ci = c_u.get(uh)
        for c, ig_name in enumerate(ig):
            bj = b_i.get(ig_name)
            cj = c_i.get(ig_name)
            if bi is not None and bj is not None and bi < a_raw.shape[0] and bj < a_raw.shape[1]:
                a[r, c] = float(a_raw[bi, bj])
            if ci is not None and cj is not None and ci < b_raw.shape[0] and cj < b_raw.shape[1]:
                b[r, c] = float(b_raw[ci, cj])

    d = b - a
    l1 = float(np.abs(d).sum())
    max_abs = float(np.abs(d).max()) if d.size else 0.0
    return TransferDeltaResult(
        user_groups=list(ug),
        item_groups=list(ig),
        delta_matrix=d.tolist(),
        l1_shift=l1,
        max_abs_shift=max_abs,
    )


def compute_group_tail_exposure(
    inputs: EvalInputs,
    tail_group_name: str = "Tail",
) -> dict[GroupName, float]:
    """Mean fraction of top-k slots per user in h that hit ``tail_group_name`` (then mean over users in h)."""
    k = inputs.k
    by_h: dict[GroupName, list[float]] = defaultdict(list)
    for u, recs in inputs.recommendations.items():
        h = inputs.user_group[u]
        recs_k = recs[:k]
        if not recs_k:
            continue
        hit = sum(1 for it in recs_k if inputs.item_group[it] == tail_group_name)
        by_h[h].append(hit / len(recs_k))

    return {h: float(sum(v) / len(v)) for h, v in by_h.items() if v}


def compute_exposure_utility_alignment(
    baseline_inputs: EvalInputs,
    candidate_inputs: EvalInputs,
    tail_group_name: str = "Tail",
) -> AlignmentResult:
    base_uf = compute_user_group_fairness(baseline_inputs)
    cand_uf = compute_user_group_fairness(candidate_inputs)
    base_tail = compute_group_tail_exposure(baseline_inputs, tail_group_name)
    cand_tail = compute_group_tail_exposure(candidate_inputs, tail_group_name)

    groups = ordered_user_groups(set(base_uf.group_ndcg.keys()) | set(cand_uf.group_ndcg.keys()))
    d_tail: dict[GroupName, float] = {}
    d_ndcg: dict[GroupName, float] = {}
    xs: list[float] = []
    ys: list[float] = []
    for g in groups:
        bt = base_tail.get(g, 0.0)
        ct = cand_tail.get(g, 0.0)
        bn = base_uf.group_ndcg.get(g, 0.0)
        cn = cand_uf.group_ndcg.get(g, 0.0)
        d_tail[g] = ct - bt
        d_ndcg[g] = cn - bn
        xs.append(d_tail[g])
        ys.append(d_ndcg[g])

    pearson, spearman = correlation_tail_ndcg_deltas(xs, ys)

    return AlignmentResult(
        group_delta_tail_exposure=d_tail,
        group_delta_ndcg=d_ndcg,
        pearson_alignment=pearson,
        spearman_alignment=spearman,
    )
