"""Build symmetric normalized bipartite adjacency (weighted) for RecBole-style GNNs."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch


def symmetric_norm_adjacency_weighted(
    inter_coo: sp.coo_matrix, n_users: int, n_items: int
) -> torch.Tensor:
    r"""
    Compute :math:`\hat{A} = D^{-1/2} A D^{-1/2}` on :math:`(n_{\text{users}}+n_{\text{items}})` nodes;
    bipartite edges carry weights :math:`w_{ui}` (symmetric in both directions).
    """
    inter_M = inter_coo.tocoo()
    w = np.asarray(inter_M.data, dtype=np.float32)
    if w.size != inter_M.nnz or np.any(w <= 0):
        raise ValueError("invalid COO weights")

    rows_u = inter_M.row
    cols_i = inter_M.col + n_users
    rows_l = inter_M.col + n_users
    cols_l = inter_M.row
    R = np.concatenate([rows_u, rows_l])
    C = np.concatenate([cols_i, cols_l])
    W = np.concatenate([w, w])
    A = sp.coo_matrix(
        (W, (R, C)), shape=(n_users + n_items, n_users + n_items)
    ).tocsr()

    deg = np.asarray(A.sum(axis=1)).flatten().astype(np.float64) + 1e-7
    d_inv_sqrt = np.power(deg, -0.5)
    D = sp.diags(d_inv_sqrt)
    L = (D @ A @ D).tocoo()
    idx = torch.LongTensor(np.vstack((L.row, L.col)))
    data = torch.FloatTensor(L.data)
    return torch.sparse_coo_tensor(
        idx, data, torch.Size(L.shape), dtype=torch.float32
    ).coalesce()
