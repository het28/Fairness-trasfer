"""Transfer matrix delta metrics."""

import numpy as np

from cikm_eval.transfer import compute_transfer_delta
from cikm_eval.types import TransferMatrixResult


def test_compute_transfer_delta_known():
    base = TransferMatrixResult(
        user_groups=["niche", "mainstream"],
        item_groups=["Head", "Tail"],
        matrix=[[0.5, 0.5], [0.6, 0.4]],
        row_sums={"niche": 1.0, "mainstream": 1.0},
        col_sums={"Head": 1.1, "Tail": 0.9},
        normalize="row",
    )
    cand = TransferMatrixResult(
        user_groups=["niche", "mainstream"],
        item_groups=["Head", "Tail"],
        matrix=[[0.4, 0.6], [0.5, 0.5]],
        row_sums={"niche": 1.0, "mainstream": 1.0},
        col_sums={"Head": 0.9, "Tail": 1.1},
        normalize="row",
    )
    d = compute_transfer_delta(base, cand)
    np.testing.assert_allclose(d.delta_matrix, [[-0.1, 0.1], [-0.1, 0.1]], rtol=1e-5)
    assert abs(d.l1_shift - 0.4) < 1e-9
    assert abs(d.max_abs_shift - 0.1) < 1e-9
