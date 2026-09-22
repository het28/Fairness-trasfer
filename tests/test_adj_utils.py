import numpy as np
import scipy.sparse as sp
import torch

from recbole_ext.adj_utils import symmetric_norm_adjacency_weighted


def test_weighted_adj_symmetric():
    # 2 users x 3 items, single edges
    row, col, data = [0, 1], [0, 1], [2.0, 0.5]
    m = sp.coo_matrix((data, (row, col)), shape=(2, 3))
    t = symmetric_norm_adjacency_weighted(m, 2, 3)
    assert t.is_sparse
    # symmetry: compare with transpose (allow numerical tolerance)
    dense = t.to_dense()
    assert torch.allclose(dense, dense.T, atol=1e-5)
