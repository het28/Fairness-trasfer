"""Unit tests for c_ui multiplication and per-user exposure loss."""
from __future__ import annotations

import numpy as np

from evaluation.exposure import per_user_exposure_loss, summarize_per_user_exposure_loss
from evaluation.types import EvalInputs
from training.recbole_dataset import _extract_c_ui


class _Feat:
    def __init__(self, d):
        self.interaction = d

    def __getitem__(self, k):
        import torch

        return torch.as_tensor(self.interaction[k])


class _DS:
    def __init__(self, rating=None):
        self.rating_field = "weight" if rating is not None else None
        d = {"user_id": np.array([1, 2]), "item_id": np.array([1, 2])}
        if rating is not None:
            d["weight"] = np.asarray(rating, dtype=np.float64)
        self.inter_feat = _Feat(d)


def test_c_ui_default_ones_without_flag():
    c = _extract_c_ui(_DS([10.0, 20.0]), n_rows=2, multiply_c_ui=False)
    assert np.allclose(c, 1.0)


def test_c_ui_log1p_transform():
    c = _extract_c_ui(_DS([0.0, np.e - 1.0]), n_rows=2, multiply_c_ui=True, c_ui_transform="log1p")
    assert c[0] == 1.0  # zero mapped to 1 to keep edge
    assert abs(float(c[1]) - 1.0) < 1e-5


def test_l_exp_bounded_and_zero_on_target():
    # Two users, catalog equal Head/Tail; perfect match => loss 0
    inputs = EvalInputs(
        recommendations={1: [0, 1], 2: [0, 1]},
        test_items={1: [0], 2: [1]},
        item_group={0: "Head", 1: "Tail"},
        user_group={1: "niche", 2: "mainstream"},
        k=2,
    )
    losses = per_user_exposure_loss(inputs, target={"Head": 0.5, "Tail": 0.5})
    assert np.allclose(losses, 0.0)
    s = summarize_per_user_exposure_loss(losses, tau=0.1)
    assert s["l_exp_mean"] == 0.0
    assert s["l_exp_frac_gt_tau"] == 0.0


def test_l_exp_max_is_one():
    inputs = EvalInputs(
        recommendations={1: [0, 0]},
        test_items={1: [0]},
        item_group={0: "Head", 1: "Tail"},
        user_group={1: "niche"},
        k=2,
    )
    losses = per_user_exposure_loss(inputs, target={"Head": 0.0, "Tail": 1.0})
    assert losses[0] <= 1.0 + 1e-9
    assert losses[0] >= 0.0
