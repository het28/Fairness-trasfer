from __future__ import annotations

import numpy as np


def group_catalog_share(labels: np.ndarray, n_groups: int) -> np.ndarray:
    n_items = labels.shape[0]
    cnt = np.bincount(labels, minlength=n_groups)
    return cnt.astype(np.float64) / max(n_items, 1)


def group_interaction_mass(degrees: np.ndarray, labels: np.ndarray, n_groups: int) -> np.ndarray:
    total = float(degrees.sum())
    if total <= 0:
        return np.ones(n_groups, dtype=np.float64) / n_groups
    M = np.zeros(n_groups, dtype=np.float64)
    for g in range(n_groups):
        M[g] = float(degrees[labels == g].sum()) / total
    return M


def dominance(M: np.ndarray, C: np.ndarray, eps: float) -> np.ndarray:
    return M / (C + eps)
