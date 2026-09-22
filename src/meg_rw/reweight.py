from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from meg_rw.grouping import assign_popularity_groups, item_degrees_from_coo
from meg_rw.statistics import dominance, group_catalog_share, group_interaction_mass

# Default field name for RecBole Interaction (train split)
DEFAULT_MEG_RW_FIELD = "meg_rw_weight"


def build_item_popularity_groups(
    train_coo: sp.coo_matrix,
    fracs: tuple[float, float, float, float] = (0.1, 0.2, 0.3, 0.4),
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (degrees_per_item, group_label_per_item) for n_items = train_coo.shape[1].
    """
    n_users, n_items = train_coo.shape
    cols = train_coo.col
    deg = item_degrees_from_coo(cols, n_items)
    labels = assign_popularity_groups(n_items, deg, fracs=fracs)
    return deg, labels


def meg_rw_phi_from_groups(
    degrees: np.ndarray,
    labels: np.ndarray,
    n_groups: int,
    alpha: float,
    eps: float = 1e-8,
    normalize_phi: bool = False,
    mode: str = "dominance",
) -> np.ndarray:
    """phi[g] per group index."""
    mode = str(mode).lower()
    C = group_catalog_share(labels, n_groups)
    M = group_interaction_mass(degrees, labels, n_groups)
    if mode == "uniform":
        phi = np.ones(n_groups, dtype=np.float64)
    elif mode in {"catalog", "catalog_share"}:
        # no-dominance ablation: use only catalog share term
        phi = np.power(C + eps, -float(alpha), dtype=np.float64)
    else:
        # default MEG-RW
        D = dominance(M, C, eps)
        phi = np.power(D + eps, -float(alpha), dtype=np.float64)
    if normalize_phi:
        s = phi.sum()
        if s > 0:
            phi = phi * (n_groups / s)
    return phi


def apply_meg_rw_to_coo(
    train_coo: sp.coo_matrix,
    alpha: float,
    eps: float = 1e-8,
    fracs: tuple[float, float, float, float] = (0.1, 0.2, 0.3, 0.4),
    normalize_phi: bool = False,
    mode: str = "dominance",
) -> sp.coo_matrix:
    """
    Return a new COO with same sparsity pattern as train_coo and data[k] = phi(g(item_col)).
    alpha=0 => all ones (up to float32).
    """
    train_coo = train_coo.tocoo()
    n_items = train_coo.shape[1]
    deg, lab = build_item_popularity_groups(train_coo, fracs=fracs)
    n_groups = len(fracs)
    phi = meg_rw_phi_from_groups(
        deg, lab, n_groups=n_groups, alpha=alpha, eps=eps, normalize_phi=normalize_phi, mode=mode
    )
    w = phi[lab[train_coo.col.astype(np.int64, copy=False)]].astype(np.float32)
    return sp.coo_matrix((w, (train_coo.row, train_coo.col)), shape=train_coo.shape)


def meg_rw_weights_for_interaction_rows(
    item_internal_ids: np.ndarray,
    n_items: int,
    alpha: float,
    eps: float = 1e-8,
    fracs: tuple[float, float, float, float] = (0.1, 0.2, 0.3, 0.4),
    normalize_phi: bool = False,
    mode: str = "dominance",
) -> np.ndarray:
    """
    One MEG-RW weight per training interaction row (same order as item_internal_ids).
    Degrees / groups use counts over these rows (train-only), matching apply_meg_rw_to_coo.
    """
    ii = item_internal_ids.astype(np.int64, copy=False)
    deg = np.bincount(ii, minlength=n_items)
    labels = assign_popularity_groups(n_items, deg, fracs=fracs)
    n_groups = len(fracs)
    phi = meg_rw_phi_from_groups(
        deg, labels, n_groups=n_groups, alpha=alpha, eps=eps, normalize_phi=normalize_phi, mode=mode
    )
    return phi[labels[ii]].astype(np.float32)
