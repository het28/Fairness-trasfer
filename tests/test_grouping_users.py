import numpy as np

from cikm_eval.grouping import user_mainstreamness_labels


def test_user_mainstreamness_shape():
    n_users = 8
    u = np.array([0, 0, 1, 1, 2, 3, 4, 5, 6, 7])
    i = np.array([0, 1, 0, 2, 3, 4, 5, 6, 7, 0])
    p = np.linspace(0, 1, num=8)
    lab = user_mainstreamness_labels(u, i, n_users, p, n_quantiles=4)
    assert lab.shape == (n_users,)
    assert lab.min() >= 0 and lab.max() <= 3
