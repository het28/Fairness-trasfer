import numpy as np
import scipy.sparse as sp

from meg_rw import apply_meg_rw_to_coo


def test_alpha_zero_is_ones():
    rows = np.array([0, 0, 1, 2], dtype=np.int32)
    cols = np.array([0, 1, 2, 0], dtype=np.int32)
    data = np.ones(4, dtype=np.float32)
    m = sp.coo_matrix((data, (rows, cols)), shape=(4, 5))
    out = apply_meg_rw_to_coo(m, alpha=0.0)
    np.testing.assert_array_almost_equal(out.data, np.ones_like(out.data))


def test_weights_positive():
    rng = np.random.default_rng(0)
    n_u, n_i = 20, 30
    nnz = 80
    rows = rng.integers(0, n_u, size=nnz)
    cols = rng.integers(0, n_i, size=nnz)
    m = sp.coo_matrix((np.ones(nnz), (rows, cols)), shape=(n_u, n_i))
    out = apply_meg_rw_to_coo(m, alpha=0.5)
    assert np.all(out.data > 0)
    assert out.nnz == m.nnz
