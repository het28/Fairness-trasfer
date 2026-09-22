from __future__ import annotations

import numpy as np


def item_degrees_from_coo(item_cols: np.ndarray, n_items: int) -> np.ndarray:
    """Count training interactions per item (internal RecBole item ids 0..n_items-1)."""
    d = np.zeros(n_items, dtype=np.int64)
    np.add.at(d, item_cols.astype(np.int64, copy=False), 1)
    return d


def assign_popularity_groups(
    n_items: int,
    degrees: np.ndarray,
    fracs: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4),
) -> np.ndarray:
    """
    Label each item with group id 0..G-1, where G = len(fracs).
    Items sorted by descending degree; partition by fracs of the catalog (n_items).
    """
    if len(fracs) < 2:
        raise ValueError("fracs must have at least 2 groups")
    if abs(sum(fracs) - 1.0) > 1e-6:
        raise ValueError("fracs must sum to 1")
    order = np.argsort(-degrees, kind="stable")
    sizes: list[int] = []
    for f in fracs[:-1]:
        sizes.append(int(round(f * n_items)))
    sizes.append(max(0, n_items - sum(sizes)))
    if sum(sizes) != n_items:
        sizes[-1] = n_items - sum(sizes[:-1])

    labels = np.zeros(n_items, dtype=np.int64)
    start = 0
    for g, sz in enumerate(sizes):
        end = start + sz
        labels[order[start:end]] = g
        start = end
    return labels
