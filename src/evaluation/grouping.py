"""Item popularity groups and user mainstreamness (train-only signals)."""

from __future__ import annotations

import numpy as np

from meg_rw.grouping import assign_popularity_groups, item_degrees_from_coo


def item_group_labels_from_train_coo(
    train_item_cols: np.ndarray,
    n_items: int,
    fracs: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4),
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (degrees per item, label 0..G-1 per item), G=len(fracs)."""
    deg = item_degrees_from_coo(train_item_cols, n_items)
    lab = assign_popularity_groups(n_items, deg, fracs=fracs)
    return deg, lab


def normalized_popularity(deg: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    dmin = float(deg.min())
    dmax = float(deg.max())
    return (deg.astype(np.float64) - dmin) / (dmax - dmin + eps)


def user_mainstreamness_labels(
    train_user_rows: np.ndarray,
    train_item_cols: np.ndarray,
    n_users: int,
    p_item: np.ndarray,
    n_quantiles: int = 4,
) -> np.ndarray:
    """
    M(u) = mean of p_item[i] over train items i of u; split users into ``n_quantiles``
    buckets by M(u) (0 = niche, n_quantiles-1 = mainstream).
    """
    u = train_user_rows.astype(np.int64, copy=False)
    i = train_item_cols.astype(np.int64, copy=False)
    sum_m = np.zeros(n_users, dtype=np.float64)
    cnt = np.bincount(u, minlength=n_users).astype(np.float64)
    np.add.at(sum_m, u, p_item[i])
    with np.errstate(invalid="ignore", divide="ignore"):
        m = np.where(cnt > 0, sum_m / np.maximum(cnt, 1.0), 0.0)
    order = np.argsort(m, kind="stable")
    labels = np.zeros(n_users, dtype=np.int64)
    qsz = max(1, n_users // n_quantiles)
    for q in range(n_quantiles):
        lo = q * qsz
        hi = n_users if q == n_quantiles - 1 else (q + 1) * qsz
        labels[order[lo:hi]] = q
    return labels
