"""Exposure–utility alignment across user groups (baseline vs candidate)."""

from __future__ import annotations

from cikm_eval.transfer import compute_exposure_utility_alignment, correlation_tail_ndcg_deltas
from cikm_eval.types import EvalInputs


def test_correlation_perfect_line():
    p, s = correlation_tail_ndcg_deltas([0.5, 1.0, 1.5], [1.0, 2.0, 3.0])
    assert p is not None and p > 0.999
    assert s is not None and s > 0.999


def test_correlation_constant_returns_none():
    assert correlation_tail_ndcg_deltas([1.0, 1.0], [2.0, 3.0]) == (None, None)


def test_compute_exposure_utility_alignment_smoke():
    item_group = {1: "Tail", 2: "Head", 3: "Head", 4: "Tail"}
    base = EvalInputs(
        recommendations={0: [2, 3], 1: [2, 3]},
        test_items={0: [2], 1: [3]},
        item_group=item_group,
        user_group={0: "niche", 1: "mainstream"},
        k=2,
    )
    cand = EvalInputs(
        recommendations={0: [1, 2], 1: [3, 4]},
        test_items={0: [1], 1: [4]},
        item_group=item_group,
        user_group={0: "niche", 1: "mainstream"},
        k=2,
    )
    al = compute_exposure_utility_alignment(base, cand)
    assert set(al.group_delta_tail_exposure) == {"niche", "mainstream"}
    assert set(al.group_delta_ndcg) == {"niche", "mainstream"}
